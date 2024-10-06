use core::f64;
use half::f16;
use std::{
    cmp::Ordering,
    mem,
    ops::{Add, Div, Mul, Neg, Rem, Sub},
};

#[derive(Clone, Copy, PartialEq)]
pub enum Kind {
    E4M3,
    E5M2,
}

#[derive(Clone, Copy, PartialEq, Default)]
/// Saturation type. If `NoSat`, allow NaN and inf.
pub enum SaturationType {
    NoSat,
    #[default]
    SatFinite,
}

// https://gitlab.com/nvidia/headers/cuda-individual/cudart/-/blob/main/cuda_fp8.hpp?ref_type=heads#L97
const fn convert_to_fp8(x: f64, saturate: SaturationType, fp8_interpretation: Kind) -> u8 {
    let xbits: u64 = unsafe { mem::transmute::<f64, u64>(x) };

    let (
        fp8_maxnorm,
        fp8_mantissa_mask,
        fp8_exp_bias,
        fp8_significand_bits,
        fp8_mindenorm_o2,
        fp8_overflow_threshold,
        fp8_minnorm,
    ) = match fp8_interpretation {
        Kind::E4M3 => (
            0x7E_u8,
            0x7_u8,
            7_u16,
            4_u64,
            0x3F50000000000000_u64,
            0x407D000000000000_u64,
            0x3F90000000000000_u64,
        ),
        Kind::E5M2 => (
            0x7B_u8,
            0x3_u8,
            15_u16,
            3_u64,
            0x3EE0000000000000_u64,
            0x40EE000000000000_u64 - 1,
            0x3F10000000000000_u64,
        ),
    };

    const DP_INF_BITS: u64 = 0x7FF0000000000000;
    let fp8_dp_half_ulp: u64 = 1 << (53 - fp8_significand_bits - 1);
    let sign: u8 = ((xbits >> 63) << 7) as u8;
    let exp: u8 = ((((xbits >> 52) as u16) & 0x7FF)
        .wrapping_sub(1023)
        .wrapping_add(fp8_exp_bias)) as u8;
    let mantissa: u8 = ((xbits >> (53 - fp8_significand_bits)) & (fp8_mantissa_mask as u64)) as u8;
    let absx: u64 = xbits & 0x7FFFFFFFFFFFFFFF;

    let res = if absx <= fp8_mindenorm_o2 {
        // Zero or underflow
        0
    } else if absx > DP_INF_BITS {
        // Preseve NaNs
        match fp8_interpretation {
            Kind::E4M3 => 0x7F,
            Kind::E5M2 => 0x7E | mantissa,
        }
    } else if absx > fp8_overflow_threshold {
        // Saturate
        match saturate {
            SaturationType::SatFinite => fp8_maxnorm,
            SaturationType::NoSat => match fp8_interpretation {
                Kind::E4M3 => 0x7F,
                Kind::E5M2 => 0x7C,
            },
        }
    } else if absx >= fp8_minnorm {
        // Round, normal range
        let mut res = ((exp << (fp8_significand_bits - 1)) | mantissa) as u8;

        // Round off bits and round-to-nearest-even adjustment
        let round = xbits & ((fp8_dp_half_ulp << 1) - 1);
        if (round > fp8_dp_half_ulp) || ((round == fp8_dp_half_ulp) && (mantissa & 1 != 0)) {
            res = res.wrapping_add(1);
        }
        res
    } else {
        // Denormal numbers
        let shift = 1_u8.wrapping_sub(exp);
        let mantissa = mantissa | (1 << (fp8_significand_bits - 1));
        let mut res = mantissa >> shift;

        // Round off bits and round-to-nearest-even adjustment
        let round = (xbits | (1 << (53 - 1))) & ((fp8_dp_half_ulp << (shift as u64 + 1)) - 1);
        if (round > (fp8_dp_half_ulp << shift as u64))
            || ((round == (fp8_dp_half_ulp << shift as u64)) && (res & 1 != 0))
        {
            res = res.wrapping_add(1);
        }
        res
    };

    res | sign
}

// https://gitlab.com/nvidia/headers/cuda-individual/cudart/-/blob/main/cuda_fp8.hpp?ref_type=heads#L463
const fn convert_fp8_to_fp16(x: u8, fp8_interpretation: Kind) -> u16 {
    let mut ur = (x as u16) << 8;

    match fp8_interpretation {
        Kind::E5M2 => {
            if (ur & 0x7FFF) > 0x7C00 {
                // If NaN, return canonical NaN
                ur = 0x7FFF;
            }
        }
        Kind::E4M3 => {
            let sign = ur & 0x8000;
            let mut exponent = ((ur & 0x7800) >> 1).wrapping_add(0x2000);
            let mut mantissa = (ur & 0x0700) >> 1;
            let absx = 0x7F & x;

            if absx == 0x7F {
                // FP16 canonical NaN, discard sign
                ur = 0x7FFF;
            } else if exponent == 0x2000 {
                // Zero or denormal
                if mantissa != 0 {
                    // Normalize
                    mantissa <<= 1;
                    while (mantissa & 0x0400) == 0 {
                        mantissa <<= 1;
                        exponent = exponent.wrapping_sub(0x0400);
                    }
                    // Discard implicit leading bit
                    mantissa &= 0x03FF;
                } else {
                    // Zero
                    exponent = 0;
                }
                ur = sign | exponent | mantissa;
            } else {
                ur = sign | exponent | mantissa;
            }
        }
    };

    ur
}

#[derive(Copy, Clone)]
#[repr(transparent)]
/// Eight bit floating point type with 4-bit exponent and 3-bit mantissa.
pub struct F8E4M3(u8);

impl F8E4M3 {
    const INTERPRETATION: Kind = Kind::E4M3;

    /// Construct a 8-bit floating point value from the raw bits.
    pub const fn from_bits(bits: u8) -> Self {
        Self(bits)
    }

    /// Construct a 8-bit floating point value from the raw bits.
    pub const fn to_bits(&self) -> u8 {
        self.0
    }

    /// Convert a [`f64`] type into [`F8E4M3`].
    ///
    /// This operation is lossy.
    ///
    /// - If the 64-bit value is to large to fit in 8-bits, ±∞ will result.
    /// - NaN values are preserved.
    /// - 64-bit subnormal values are too tiny to be represented in 8-bits and result in ±0.
    /// - Exponents that underflow the minimum 8-bit exponent will result in 8-bit subnormals or ±0.
    /// - All other values are truncated and rounded to the nearest representable  8-bit value.
    pub const fn from_f64(x: f64) -> Self {
        Self(convert_to_fp8(
            x,
            SaturationType::SatFinite,
            Self::INTERPRETATION,
        ))
    }

    /// Convert a [`f32`] type into [`F8E4M3`].
    ///
    /// This operation is lossy.
    ///
    /// - If the 32-bit value is to large to fit in 8-bits, ±∞ will result.
    /// - NaN values are preserved.
    /// - 32-bit subnormal values are too tiny to be represented in 8-bits and result in ±0.
    /// - Exponents that underflow the minimum 8-bit exponent will result in 8-bit subnormals or ±0.
    /// - All other values are truncated and rounded to the nearest representable  8-bit value.
    pub const fn from_f32(x: f32) -> Self {
        Self::from_f64(x as f64)
    }

    /// Convert this [`F8E4M3`] type into a [`f16`] type.
    ///
    /// This operation may be lossy.
    ///
    /// - NaN and zero values are preserved.
    /// - Subnormal values are normalized.
    /// - Otherwise, the values are mapped to the appropriate 16-bit value.
    pub const fn to_f16(&self) -> f16 {
        f16::from_bits(convert_fp8_to_fp16(self.0, Self::INTERPRETATION))
    }

    /// Convert this [`F8E4M3`] type into a [`f32`] type.
    ///
    /// This operation may be lossy.
    ///
    /// - NaN and zero values are preserved.
    /// - Subnormal values are normalized.
    /// - Otherwise, the values are mapped to the appropriate 16-bit value.
    pub const fn to_f32(&self) -> f32 {
        self.to_f16().to_f32_const()
    }

    /// Convert this [`F8E4M3`] type into a [`f64`] type.
    ///
    /// This operation may be lossy.
    ///
    /// - NaN and zero values are preserved.
    /// - Subnormal values are normalized.
    /// - Otherwise, the values are mapped to the appropriate 16-bit value.
    pub const fn to_f64(&self) -> f64 {
        self.to_f16().to_f64_const()
    }

    /// Returns the ordering between `self` and `other`.
    ///
    /// - negative quiet NaN
    /// - negative signaling NaN
    /// - negative infinity
    /// - negative numbers
    /// - negative subnormal numbers
    /// - negative zero
    /// - positive zero
    /// - positive subnormal numbers
    /// - positive numbers
    /// - positive infinity
    /// - positive signaling NaN
    /// - positive quiet NaN.
    ///
    /// The ordering established by this function does not always agree with the
    /// [`PartialOrd`] and [`PartialEq`] implementations. For example,
    /// they consider negative and positive zero equal, while `total_cmp`
    /// doesn't.
    ///
    /// # Example
    /// ```
    /// # use float8::f8;
    ///
    /// let mut v: Vec<f8> = vec![];
    /// v.push(f8::ONE);
    /// v.push(f8::INFINITY);
    /// v.push(f8::NEG_INFINITY);
    /// v.push(f8::NAN);
    /// v.push(f8::MAX_SUBNORMAL);
    /// v.push(-f8::MAX_SUBNORMAL);
    /// v.push(f8::ZERO);
    /// v.push(f8::NEG_ZERO);
    /// v.push(f8::NEG_ONE);
    /// v.push(f8::MIN_POSITIVE);
    ///
    /// v.sort_by(|a, b| a.total_cmp(&b));
    ///
    /// assert!(v
    ///     .into_iter()
    ///     .zip(
    ///         [
    ///             f8::NEG_INFINITY,
    ///             f8::NEG_ONE,
    ///             -f8::MAX_SUBNORMAL,
    ///             f8::NEG_ZERO,
    ///             f8::ZERO,
    ///             f8::MAX_SUBNORMAL,
    ///             f8::MIN_POSITIVE,
    ///             f8::ONE,
    ///             f8::INFINITY,
    ///             f8::NAN
    ///         ]
    ///         .iter()
    ///     )
    ///     .all(|(a, b)| a.to_bits() == b.to_bits()));
    /// ```
    pub fn total_cmp(&self, other: &Self) -> Ordering {
        let mut left = self.to_bits() as i8;
        let mut right = other.to_bits() as i8;
        left ^= (((left >> 7) as u8) >> 1) as i8;
        right ^= (((right >> 7) as u8) >> 1) as i8;
        left.cmp(&right)
    }

    /// Returns `true` if and only if `self` has a positive sign, including +0.0, NaNs with a
    /// positive sign bit and +∞.
    pub const fn is_sign_positive(&self) -> bool {
        self.0 & 0x80u8 == 0
    }

    /// Returns `true` if and only if `self` has a negative sign, including −0.0, NaNs with a
    /// negative sign bit and −∞.
    pub const fn is_sign_negative(&self) -> bool {
        self.0 & 0x80u8 != 0
    }

    // Returns `true` if this value is NaN and `false` otherwise.
    pub const fn is_nan(self) -> bool {
        self.0 == 0x7Fu8 || self.0 == 0xFFu8
    }
}

#[derive(Copy, Clone)]
#[repr(transparent)]
/// Eight bit floating point type with 5-bit exponent and 2-bit mantissa.
pub struct F8E5M2(u8);

impl F8E5M2 {
    const INTERPRETATION: Kind = Kind::E5M2;

    /// Construct a 8-bit floating point value from the raw bits.
    pub const fn from_bits(bits: u8) -> Self {
        Self(bits)
    }

    /// Construct a 8-bit floating point value from the raw bits.
    pub const fn to_bits(&self) -> u8 {
        self.0
    }

    /// Convert a [`f64`] type into [`F8E5M2`].
    ///
    /// This operation is lossy.
    ///
    /// - If the 64-bit value is to large to fit in 8-bits, ±∞ will result.
    /// - NaN values are preserved.
    /// - 64-bit subnormal values are too tiny to be represented in 8-bits and result in ±0.
    /// - Exponents that underflow the minimum 8-bit exponent will result in 8-bit subnormals or ±0.
    /// - All other values are truncated and rounded to the nearest representable  8-bit value.
    pub const fn from_f64(x: f64) -> Self {
        Self(convert_to_fp8(
            x,
            SaturationType::SatFinite,
            Self::INTERPRETATION,
        ))
    }

    /// Convert a [`f32`] type into [`F8E5M2`].
    ///
    /// This operation is lossy.
    ///
    /// - If the 32-bit value is to large to fit in 8-bits, ±∞ will result.
    /// - NaN values are preserved.
    /// - 32-bit subnormal values are too tiny to be represented in 8-bits and result in ±0.
    /// - Exponents that underflow the minimum 8-bit exponent will result in 8-bit subnormals or ±0.
    /// - All other values are truncated and rounded to the nearest representable  8-bit value.
    pub const fn from_f32(x: f32) -> Self {
        Self::from_f64(x as f64)
    }

    /// Convert this [`F8E5M2`] type into a [`f16`] type.
    ///
    /// This operation may be lossy.
    ///
    /// - NaN and zero values are preserved.
    /// - Subnormal values are normalized.
    /// - Otherwise, the values are mapped to the appropriate 16-bit value.
    pub const fn to_f16(&self) -> f16 {
        f16::from_bits(convert_fp8_to_fp16(self.0, Self::INTERPRETATION))
    }

    /// Convert this [`F8E5M2`] type into a [`f32`] type.
    ///
    /// This operation may be lossy.
    ///
    /// - NaN and zero values are preserved.
    /// - Subnormal values are normalized.
    /// - Otherwise, the values are mapped to the appropriate 16-bit value.
    pub const fn to_f32(&self) -> f32 {
        self.to_f16().to_f32_const()
    }

    /// Convert this [`F8E5M2`] type into a [`f64`] type.
    ///
    /// This operation may be lossy.
    ///
    /// - NaN and zero values are preserved.
    /// - Subnormal values are normalized.
    /// - Otherwise, the values are mapped to the appropriate 16-bit value.
    pub const fn to_f64(&self) -> f64 {
        self.to_f16().to_f64_const()
    }

    /// Returns the ordering between `self` and `other`.
    ///
    /// - negative quiet NaN
    /// - negative signaling NaN
    /// - negative infinity
    /// - negative numbers
    /// - negative subnormal numbers
    /// - negative zero
    /// - positive zero
    /// - positive subnormal numbers
    /// - positive numbers
    /// - positive infinity
    /// - positive signaling NaN
    /// - positive quiet NaN.
    ///
    /// The ordering established by this function does not always agree with the
    /// [`PartialOrd`] and [`PartialEq`] implementations. For example,
    /// they consider negative and positive zero equal, while `total_cmp`
    /// doesn't.
    ///
    /// # Example
    /// ```
    /// # use float8::F8E5M2;
    ///
    /// let mut v: Vec<F8E5M2> = vec![];
    /// v.push(F8E5M2::ONE);
    /// v.push(F8E5M2::INFINITY);
    /// v.push(F8E5M2::NEG_INFINITY);
    /// v.push(F8E5M2::NAN);
    /// v.push(F8E5M2::MAX_SUBNORMAL);
    /// v.push(-F8E5M2::MAX_SUBNORMAL);
    /// v.push(F8E5M2::ZERO);
    /// v.push(F8E5M2::NEG_ZERO);
    /// v.push(F8E5M2::NEG_ONE);
    /// v.push(F8E5M2::MIN_POSITIVE);
    ///
    /// v.sort_by(|a, b| a.total_cmp(&b));
    ///
    /// assert!(v
    ///     .into_iter()
    ///     .zip(
    ///         [
    ///             F8E5M2::NEG_INFINITY,
    ///             F8E5M2::NEG_ONE,
    ///             -F8E5M2::MAX_SUBNORMAL,
    ///             F8E5M2::NEG_ZERO,
    ///             F8E5M2::ZERO,
    ///             F8E5M2::MAX_SUBNORMAL,
    ///             F8E5M2::MIN_POSITIVE,
    ///             F8E5M2::ONE,
    ///             F8E5M2::INFINITY,
    ///             F8E5M2::NAN
    ///         ]
    ///         .iter()
    ///     )
    ///     .all(|(a, b)| a.to_bits() == b.to_bits()));
    /// ```
    pub fn total_cmp(&self, other: &Self) -> Ordering {
        let mut left = self.to_bits() as i8;
        let mut right = other.to_bits() as i8;
        left ^= (((left >> 7) as u8) >> 1) as i8;
        right ^= (((right >> 7) as u8) >> 1) as i8;
        left.cmp(&right)
    }

    /// Returns `true` if and only if `self` has a positive sign, including +0.0, NaNs with a
    /// positive sign bit and +∞.
    pub const fn is_sign_positive(&self) -> bool {
        self.0 & 0x80u8 == 0
    }

    /// Returns `true` if and only if `self` has a negative sign, including −0.0, NaNs with a
    /// negative sign bit and −∞.
    pub const fn is_sign_negative(&self) -> bool {
        self.0 & 0x80u8 != 0
    }

    // Returns `true` if this value is NaN and `false` otherwise.
    // pub const fn is_nan(self) -> bool {
    pub fn is_nan(self) -> bool {
        self.0 == 0x7Eu8 || self.0 == 0xFEu8
    }
}

macro_rules! comparison {
    ($t:ident) => {
        impl PartialEq for $t {
            fn eq(&self, other: &Self) -> bool {
                if self.is_nan() || other.is_nan() {
                    false
                } else {
                    (self.0 == other.0) || ((self.0 | other.0) & 0x7Fu8 == 0)
                }
            }
        }

        impl PartialOrd for $t {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                if self.is_nan() || other.is_nan() {
                    None
                } else {
                    let neg = self.0 & 0x80u8 != 0;
                    let other_neg = other.0 & 0x80u8 != 0;
                    match (neg, other_neg) {
                        (false, false) => Some(self.0.cmp(&other.0)),
                        (false, true) => {
                            if (self.0 | other.0) & 0x7Fu8 == 0 {
                                Some(Ordering::Equal)
                            } else {
                                Some(Ordering::Greater)
                            }
                        }
                        (true, false) => {
                            if (self.0 | other.0) & 0x7Fu8 == 0 {
                                Some(Ordering::Equal)
                            } else {
                                Some(Ordering::Less)
                            }
                        }
                        (true, true) => Some(other.0.cmp(&self.0)),
                    }
                }
            }

            fn lt(&self, other: &Self) -> bool {
                if self.is_nan() || other.is_nan() {
                    false
                } else {
                    let neg = self.0 & 0x80u8 != 0;
                    let other_neg = other.0 & 0x80u8 != 0;
                    match (neg, other_neg) {
                        (false, false) => self.0 < other.0,
                        (false, true) => false,
                        (true, false) => (self.0 | other.0) & 0x7Fu8 != 0,
                        (true, true) => self.0 > other.0,
                    }
                }
            }

            fn le(&self, other: &Self) -> bool {
                if self.is_nan() || other.is_nan() {
                    false
                } else {
                    let neg = self.0 & 0x80u8 != 0;
                    let other_neg = other.0 & 0x80u8 != 0;
                    match (neg, other_neg) {
                        (false, false) => self.0 <= other.0,
                        (false, true) => (self.0 | other.0) & 0x7Fu8 == 0,
                        (true, false) => true,
                        (true, true) => self.0 >= other.0,
                    }
                }
            }

            fn gt(&self, other: &Self) -> bool {
                if self.is_nan() || other.is_nan() {
                    false
                } else {
                    let neg = self.0 & 0x80u8 != 0;
                    let other_neg = other.0 & 0x80u8 != 0;
                    match (neg, other_neg) {
                        (false, false) => self.0 > other.0,
                        (false, true) => (self.0 | other.0) & 0x7Fu8 != 0,
                        (true, false) => false,
                        (true, true) => self.0 < other.0,
                    }
                }
            }

            fn ge(&self, other: &Self) -> bool {
                if self.is_nan() || other.is_nan() {
                    false
                } else {
                    let neg = self.0 & 0x80u8 != 0;
                    let other_neg = other.0 & 0x80u8 != 0;
                    match (neg, other_neg) {
                        (false, false) => self.0 >= other.0,
                        (false, true) => true,
                        (true, false) => (self.0 | other.0) & 0x7Fu8 == 0,
                        (true, true) => self.0 <= other.0,
                    }
                }
            }
        }
    };
}

comparison!(F8E4M3);
comparison!(F8E5M2);

macro_rules! constants {
    ($t:ident) => {
        impl $t {
            /// π
            pub const PI: Self = Self::from_f64(f64::consts::PI);

            /// The full circle constant (τ)
            ///
            /// Equal to 2π.
            pub const TAU: Self = Self::from_f64(f64::consts::TAU);

            /// π/2
            pub const FRAC_PI_2: Self = Self::from_f64(f64::consts::FRAC_PI_2);

            /// π/3
            pub const FRAC_PI_3: Self = Self::from_f64(f64::consts::FRAC_PI_3);

            /// π/4
            pub const FRAC_PI_4: Self = Self::from_f64(f64::consts::FRAC_PI_4);

            /// π/6
            pub const FRAC_PI_6: Self = Self::from_f64(f64::consts::FRAC_PI_6);

            /// π/8
            pub const FRAC_PI_8: Self = Self::from_f64(f64::consts::FRAC_PI_8);

            /// 1/π
            pub const FRAC_1_PI: Self = Self::from_f64(f64::consts::FRAC_1_PI);

            /// 2/π
            pub const FRAC_2_PI: Self = Self::from_f64(f64::consts::FRAC_2_PI);

            /// 2/sqrt(π)
            pub const FRAC_2_SQRT_PI: Self = Self::from_f64(f64::consts::FRAC_2_SQRT_PI);

            /// sqrt(2)
            pub const SQRT_2: Self = Self::from_f64(f64::consts::SQRT_2);

            /// 1/sqrt(2)
            pub const FRAC_1_SQRT_2: Self = Self::from_f64(f64::consts::FRAC_1_SQRT_2);

            /// Euler's number (e)
            pub const E: Self = Self::from_f64(f64::consts::E);

            /// log<sub>2</sub>(10)
            pub const LOG2_10: Self = Self::from_f64(f64::consts::LOG2_10);

            /// log<sub>2</sub>(e)
            pub const LOG2_E: Self = Self::from_f64(f64::consts::LOG2_E);

            /// log<sub>10</sub>(2)
            pub const LOG10_2: Self = Self::from_f64(f64::consts::LOG10_2);

            /// log<sub>10</sub>(e)
            pub const LOG10_E: Self = Self::from_f64(f64::consts::LOG10_E);

            /// ln(2)
            pub const LN_2: Self = Self::from_f64(f64::consts::LN_2);

            /// ln(10)
            pub const LN_10: Self = Self::from_f64(f64::consts::LN_10);
        }
    };
}

constants!(F8E4M3);
constants!(F8E5M2);

#[allow(clippy::unusual_byte_groupings)]
impl F8E4M3 {
    /// Number of mantissa digits
    pub const MANTISSA_DIGITS: u32 = 3;
    /// Maxiumum possible value
    pub const MAX: Self = Self::from_bits(0x7E - 1);
    /// Minumum possible value
    pub const MIN: Self = Self::from_bits(0xFE - 1);
    /// Positive infinity ∞
    pub const INFINITY: Self = Self::from_bits(0x7E);
    /// Negative infinity -∞
    pub const NEG_INFINITY: Self = Self::from_bits(0xFE);
    /// Smallest possible normal value
    pub const MIN_POSITIVE: Self = Self::from_bits(0b0_0001_000);
    /// Smallest possible subnormal value
    pub const MIN_POSITIVE_SUBNORMAL: Self = Self::from_bits(0b0_0000_001);
    /// Smallest possible subnormal value
    pub const MAX_SUBNORMAL: Self = Self::from_bits(0b0_0000_111);
    /// This is the difference between 1.0 and the next largest representable number.
    pub const EPSILON: Self = Self::from_bits(0b0_0100_000);
    /// NaN value
    pub const NAN: Self = Self::from_bits(0x7F);
    /// 1
    pub const ONE: Self = Self::from_bits(0b0_0111_000);
    /// 0
    pub const ZERO: Self = Self::from_bits(0b0_0000_000);
    /// -1
    pub const NEG_ONE: Self = Self::from_bits(0b1_0111_000);
    /// -0
    pub const NEG_ZERO: Self = Self::from_bits(0b1_0000_000);
    /// One greater than the minimum possible normal power of 2 exponent
    pub const MIN_EXP: i32 = -5;
    /// Minimum possible normal power of 10 exponent
    pub const MIN_10_EXP: i32 = -1;
    /// Maximum possible normal power of 2 exponent
    pub const MAX_EXP: i32 = 7;
    /// Maximum possible normal power of 10 exponent
    pub const MAX_10_EXP: i32 = 2;
    /// Approximate number of significant digits in base 10
    pub const DIGITS: u32 = 0;
}

#[allow(clippy::unusual_byte_groupings)]
impl F8E5M2 {
    /// Number of mantissa digits
    pub const MANTISSA_DIGITS: u32 = 2;
    /// Maxiumum possible value
    pub const MAX: Self = Self::from_bits(0x7B - 1);
    /// Minumum possible value
    pub const MIN: Self = Self::from_bits(0xFB - 1);
    /// Positive infinity ∞
    pub const INFINITY: Self = Self::from_bits(0x7B);
    /// Negative infinity -∞
    pub const NEG_INFINITY: Self = Self::from_bits(0xFB);
    /// Smallest possible normal value
    pub const MIN_POSITIVE: Self = Self::from_bits(0b0_00001_00);
    /// Smallest possible subnormal value
    pub const MIN_POSITIVE_SUBNORMAL: Self = Self::from_bits(0b0_00000_01);
    /// Smallest possible subnormal value
    pub const MAX_SUBNORMAL: Self = Self::from_bits(0b0_00000_11);
    /// This is the difference between 1.0 and the next largest representable number.
    pub const EPSILON: Self = Self::from_bits(0b0_01101_00);
    /// NaN value
    pub const NAN: Self = Self::from_bits(0x7E);
    /// 1
    pub const ONE: Self = Self::from_bits(0b0_01111_00);
    /// 0
    pub const ZERO: Self = Self::from_bits(0b0_00000_00);
    /// -1
    pub const NEG_ONE: Self = Self::from_bits(0b1_01111_00);
    /// -0
    pub const NEG_ZERO: Self = Self::from_bits(0b1_00000_00);
    /// One greater than the minimum possible normal power of 2 exponent
    pub const MIN_EXP: i32 = -13;
    /// Minimum possible normal power of 10 exponent
    pub const MIN_10_EXP: i32 = -4;
    /// Maximum possible normal power of 2 exponent
    pub const MAX_EXP: i32 = 15;
    /// Maximum possible normal power of 10 exponent
    pub const MAX_10_EXP: i32 = 4;
    /// Approximate number of significant digits in base 10
    pub const DIGITS: u32 = 0;
}

macro_rules! display {
    ($t:ident) => {
        impl std::fmt::Display for $t {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                std::fmt::Display::fmt(&self.to_f32(), f)
            }
        }
        impl std::fmt::Debug for $t {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                std::fmt::Debug::fmt(&self.to_f32(), f)
            }
        }
    };
}

display!(F8E4M3);
display!(F8E5M2);

macro_rules! binary {
    ($trait:ident, $fn_name:ident, $t:ident, $op:tt) => {
        impl $trait for $t {
            type Output = Self;

            fn $fn_name(self, rhs: Self) -> Self::Output {
                Self::from_f32(self.to_f32() $op rhs.to_f32())
            }
        }
    };
}

macro_rules! unary {
    ($trait:ident, $fn_name:ident, $t:ident, $op:tt) => {
        impl $trait for $t {
            type Output = Self;

            fn $fn_name(self) -> Self::Output {
                Self::from_f32($op self.to_f32())
            }
        }
    };
}

binary!(Add, add, F8E4M3, +);
binary!(Sub, sub, F8E4M3, -);
binary!(Mul, mul, F8E4M3, *);
binary!(Div, div, F8E4M3, /);
binary!(Rem, rem, F8E4M3, %);
unary!(Neg, neg, F8E4M3, -);

binary!(Add, add, F8E5M2, +);
binary!(Sub, sub, F8E5M2, -);
binary!(Mul, mul, F8E5M2, *);
binary!(Div, div, F8E5M2, /);
binary!(Rem, rem, F8E5M2, %);
unary!(Neg, neg, F8E5M2, -);

#[allow(non_camel_case_types)]
/// An alias for [`F8E4M3`].
pub type f8 = F8E4M3;

macro_rules! from_t {
    ($t:ident) => {
        impl From<$t> for f64 {
            fn from(value: $t) -> Self {
                value.to_f64()
            }
        }

        impl From<$t> for f32 {
            fn from(value: $t) -> Self {
                value.to_f32()
            }
        }

        impl From<$t> for f16 {
            fn from(value: $t) -> Self {
                value.to_f16()
            }
        }
    };
}

from_t!(F8E4M3);
from_t!(F8E5M2);

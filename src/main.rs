use half::f16;

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
#[repr(transparent)]
// F8 with 4-bit exponent and 3-bit mantissa
pub struct F8(u8);

impl F8 {
    // https://gitlab.com/nvidia/headers/cuda-individual/cudart/-/blob/main/cuda_fp8.hpp?ref_type=heads#L97
    pub fn from_f64(x: f64) -> Self {
        let xbits: u64 = unsafe { std::mem::transmute(x) };

        const FP8_EXP_BIAS: u32 = 7;
        const FP8_SIGNIFICAND_BITS: u64 = 4;
        const FP8_MANTISSA_MASK: u8 = 0x7;
        const FP8_MINDENORM_O2: u64 = 0x3F50000000000000; // mindenorm/2 = 2^-10
        const FP8_OVERFLOW_THRESHOLD: u64 = 0x407D000000000000; // maxnorm + 1/2ulp = 0x1.Cp+8 + 0x1p+4
        const FP8_MAXNORM: u32 = 0x7E;
        const FP8_MINNORM: u64 = 0x3F90000000000000; // minnorm = 2^-6
        const DP_INF_BITS: u64 = 0x7FF0000000000000;

        // 1/2 LSB of the target format, positioned in double precision mantissa
        // helpful in midpoints detection during round-to-nearest-even step
        const FP8_DP_HALF_ULP: u64 = 1u64.wrapping_shl((53u64 - FP8_SIGNIFICAND_BITS - 1) as u32);
        // prepare sign bit in target format
        let sign = ((xbits.wrapping_shr(64)).wrapping_shl(7)) as u8;
        // prepare exponent field in target format
        let exp = ((((xbits.wrapping_shr(52)) as u16) & 0x7FF) as u32)
            .wrapping_sub(1023u32)
            .wrapping_add(FP8_EXP_BIAS) as u8;
        // round mantissa to target format width, rounding towards zero
        let mantissa =
            (xbits.wrapping_shr((55u64 - FP8_SIGNIFICAND_BITS) as u32)) as u8 & FP8_MANTISSA_MASK;
        let absx = xbits & 0x7FFFFFFFFFFFFFFFu64;

        let res = if absx <= FP8_MINDENORM_O2 {
            // zero or underflow
            0u8
        } else if absx > DP_INF_BITS {
            // NaN
            0x7F
        } else if absx > FP8_OVERFLOW_THRESHOLD {
            // Saturate to infinite
            FP8_MAXNORM as u8
        } else if absx >= FP8_MINNORM {
            let mut res =
                (exp.wrapping_shl(FP8_SIGNIFICAND_BITS.wrapping_sub(1) as u32)) | mantissa;
            // rounded-off bits
            let round = xbits & (FP8_DP_HALF_ULP.wrapping_shl(1)).wrapping_sub(1);
            // round-to-nearest-even adjustment
            if round > FP8_DP_HALF_ULP || (round == FP8_DP_HALF_ULP && (mantissa & 1) != 0) {
                res = res + 1;
            }
            res
        } else {
            // Denormal range
            let shift = 1u8.wrapping_sub(exp);
            // add implicit leading bit
            let mantissa =
                mantissa | (1u8.wrapping_shl(FP8_SIGNIFICAND_BITS.wrapping_sub(1) as u32));
            // additional round-off due to denormalization
            let mut res = mantissa.wrapping_shr(shift as u32);

            // rounded-off bits, including implicit leading bit
            let round = (xbits | (1u64.wrapping_shl(53 - 1)))
                & ((FP8_DP_HALF_ULP.wrapping_shl((shift + 1) as u32)).wrapping_sub(1));
            if round > (FP8_DP_HALF_ULP.wrapping_shl(shift as u32))
                || (round == (FP8_DP_HALF_ULP.wrapping_shl(shift as u32)) && (res & 1) != 0)
            {
                res = res + 1;
            }
            res
        };

        Self(res | sign)
    }

    // https://gitlab.com/nvidia/headers/cuda-individual/cudart/-/blob/main/cuda_fp8.hpp?ref_type=heads#L463
    pub fn to_f16(&self) -> f16 {
        let mut ur: u16 = self.0.into();
        ur = ur.wrapping_shl(8);

        let sign = ur & 0x8000;
        let mut exponent = ((ur & 0x7800).wrapping_shr(1)) + 0x2000;
        let mut mantissa = (ur & 0x0700).wrapping_shr(1);
        let absx = 0x7F & self.0 as u8;

        if absx == 0x7F {
            // NaN -> fp16 canonical NaN
            ur = 0x7FFF;
        } else if exponent == 0x2000 {
            // zero or denorm
            if mantissa != 0 {
                // Noramlize
                mantissa = mantissa.wrapping_shl(1);
                while mantissa & 0x0400 == 0 {
                    mantissa = mantissa.wrapping_shl(1);
                    exponent = exponent - 0x0400;
                }
                // Discard implicit leading bit
                mantissa &= 0x03FF;
            } else {
                exponent = 0;
            }

            ur = (sign | exponent) | mantissa;
        } else {
            ur = (sign | exponent) | mantissa;
        }

        f16::from_bits(ur)
    }
}

fn main() {
    for n in 0..10 {
        let inp = (n as f64) * 0.1;
        let num = F8::from_f64(inp);
        println!("out = {}, inp = {}", num.to_f16(), inp);
    }
}

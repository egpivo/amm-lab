//! Exact Uniswap v3 arithmetic for the LVR deep-subset calibration
//! (.local/lvr, round 17). Integer-only: no floating point anywhere in
//! this module. Q64.96 sqrt prices in `U256`; 512-bit intermediates for
//! mulDiv chains and the epsilon-boundary integer square root.
//!
//! Ports:
//! - `TickMath::getSqrtRatioAtTick` (magic-constant ladder). The
//!   constants are additionally validated EMPIRICALLY at run time by the
//!   layer-0 containment check in the deep-subset binary
//!   (sqrt(tick) <= s_post < sqrt(tick+1) for every observed swap).
//! - `SqrtPriceMath::getAmount0Delta` / `getAmount1Delta` with exact
//!   rounding semantics: curve INPUT rounds up, curve OUTPUT rounds down.

use ruint::aliases::{U256, U512};

pub const MAX_TICK: i32 = 887_272;
pub const MIN_TICK: i32 = -887_272;

fn q96() -> U256 {
    U256::from(1u8) << 96
}

const MAGIC_HEX: [(u32, &str); 19] = [
    (0x2, "FFF97272373D413259A46990580E213A"),
    (0x4, "FFF2E50F5F656932EF12357CF3C7FDCC"),
    (0x8, "FFE5CACA7E10E4E61C3624EAA0941CD0"),
    (0x10, "FFCB9843D60F6159C9DB58835C926644"),
    (0x20, "FF973B41FA98C081472E6896DFB254C0"),
    (0x40, "FF2EA16466C96A3843EC78B326B52861"),
    (0x80, "FE5DEE046A99A2A811C461F1969C3053"),
    (0x100, "FCBE86C7900A88AEDCFFC83B479AA3A4"),
    (0x200, "F987A7253AC413176F2B074CF7815E54"),
    (0x400, "F3392B0822B70005940C7A398E4B70F3"),
    (0x800, "E7159475A2C29B7443B29C7FA6E889D9"),
    (0x1000, "D097F3BDFD2022B8845AD8F792AA5825"),
    (0x2000, "A9F746462D870FDF8A65DC1F90E061E5"),
    (0x4000, "70D869A156D2A1B890BB3DF62BAF32F7"),
    (0x8000, "31BE135F97D08FD981231505542FCFA6"),
    (0x10000, "9AA508B5B7A84E1C677DE54F3E99BC9"),
    (0x20000, "5D6AF8DEDB81196699C329225EE604"),
    (0x40000, "2216E584F5FA1EA926041BEDFE98"),
    (0x80000, "48A170391F7DC42444E8FA2"),
];

/// Exact integer `TickMath.getSqrtRatioAtTick`. Panics on out-of-range
/// ticks (callers validate event ticks first).
pub fn get_sqrt_ratio_at_tick(tick: i32) -> U256 {
    assert!(
        (MIN_TICK..=MAX_TICK).contains(&tick),
        "tick out of range: {tick}"
    );
    let at = tick.unsigned_abs();
    let mut ratio = if at & 1 != 0 {
        U256::from_str_radix("FFFCB933BD6FAD37AA2D162D1A594001", 16).unwrap()
    } else {
        U256::from(1u8) << 128
    };
    for (bit, hex) in MAGIC_HEX {
        if at & bit != 0 {
            let magic = U256::from_str_radix(hex, 16).unwrap();
            // 256x256 -> 512, then >> 128 fits back into 256 bits.
            let wide: U512 = ratio.widening_mul(magic);
            ratio = U256::from(wide >> 128);
        }
    }
    if tick > 0 {
        ratio = U256::MAX / ratio;
    }
    // Q128.128 -> Q64.96, rounding up.
    let down = ratio >> 32;
    let mask = (U256::from(1u8) << 32) - U256::from(1u8);
    if ratio & mask != U256::ZERO {
        down + U256::from(1u8)
    } else {
        down
    }
}

fn div_512(n: U512, d: U512, round_up: bool) -> U512 {
    let q = n / d;
    if round_up && q * d != n {
        q + U512::from(1u8)
    } else {
        q
    }
}

/// token1 amount for the sqrt range [s1, s2] (s1 < s2) at liquidity `liq`.
/// Round up for curve input, down for curve output.
pub fn amount1_delta(liq: u128, s1: U256, s2: U256, round_up: bool) -> U256 {
    debug_assert!(s1 < s2);
    let n: U512 = U256::from(liq).widening_mul(s2 - s1);
    U256::from(div_512(n, U512::from(q96()), round_up))
}

/// token0 amount for the sqrt range [s1, s2] (s1 < s2) at liquidity `liq`,
/// exact SqrtPriceMath mulDiv chain:
/// round up  = divRoundingUp(mulDivRoundingUp(liq << 96, s2 - s1, s2), s1)
/// round down= mulDiv(liq << 96, s2 - s1, s2) / s1
pub fn amount0_delta(liq: u128, s1: U256, s2: U256, round_up: bool) -> U256 {
    debug_assert!(s1 < s2);
    let shifted: U256 = U256::from(liq) << 96;
    let num: U512 = shifted.widening_mul(s2 - s1);
    let step1 = div_512(num, U512::from(s2), round_up);
    U256::from(div_512(step1, U512::from(s1), round_up))
}

/// Integer square root of a U512 (Newton's method).
pub fn isqrt_512(n: U512) -> U512 {
    if n == U512::ZERO {
        return U512::ZERO;
    }
    // initial guess: 2^(ceil(bits/2))
    let bits = 512 - n.leading_zeros();
    let mut x = U512::from(1u8) << bits.div_ceil(2);
    loop {
        let y = (x + n / x) >> 1;
        if y >= x {
            return x;
        }
        x = y;
    }
}

/// Epsilon sqrt-price boundary under the "at least an epsilon price
/// move" definition (round 18): the UP boundary rounds the square root
/// UP (ceil), the DOWN boundary rounds DOWN (floor), so the boundary is
/// never inside the epsilon band on either side:
///   s_up   = ceil( sqrt(s_pre^2 (1 + eps)) )
///   s_down = floor( sqrt(s_pre^2 (1 - eps)) )
/// The interior division by 10^4 is exact in the composed rational
/// before rounding: ceil variant uses ceil at both stages.
pub fn eps_boundary(s_pre: U256, eps_bps: u32, up: bool) -> U256 {
    let sq: U512 = s_pre.widening_mul(s_pre);
    if up {
        let k = U512::from(10_000 + eps_bps);
        let scaled = div_512(sq * k, U512::from(10_000u32), true);
        let r = isqrt_512(scaled);
        // ceil sqrt: bump unless r is an exact root
        if r * r == scaled {
            U256::from(r)
        } else {
            U256::from(r + U512::from(1u8))
        }
    } else {
        let k = U512::from(10_000 - eps_bps);
        let scaled = sq * k / U512::from(10_000u32);
        U256::from(isqrt_512(scaled))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Solidity TickMath reference values: MIN_SQRT_RATIO, MAX_SQRT_RATIO
    /// (from the deployed library), and tick 0 = 2^96 exactly.
    #[test]
    fn tickmath_reference_values() {
        assert_eq!(get_sqrt_ratio_at_tick(0), U256::from(1u8) << 96);
        assert_eq!(
            get_sqrt_ratio_at_tick(MIN_TICK),
            U256::from(4295128739u64),
            "MIN_SQRT_RATIO"
        );
        assert_eq!(
            get_sqrt_ratio_at_tick(MAX_TICK),
            U256::from_str_radix("1461446703485210103287273052203988822378723970342", 10).unwrap(),
            "MAX_SQRT_RATIO"
        );
    }

    /// Monotonicity and adjacent-tick ratio sanity around zero.
    #[test]
    fn tickmath_monotone_and_ratio() {
        let mut prev = get_sqrt_ratio_at_tick(-1000);
        for t in -999..=1000 {
            let cur = get_sqrt_ratio_at_tick(t);
            assert!(cur > prev, "sqrt ratio must be strictly increasing at {t}");
            prev = cur;
        }
        // sqrt(1.0001) ~ 1.00004999875..., check tick 1 / tick 0 to 1e-10
        let s0 = get_sqrt_ratio_at_tick(0);
        let s1 = get_sqrt_ratio_at_tick(1);
        let num = U512::from(s1) * U512::from(10u8).pow(U512::from(12u8));
        let ratio: U512 = num / U512::from(s0);
        let r: u128 = ratio.try_into().unwrap();
        assert!((r as i128 - 1_000_049_998_750).abs() < 10, "got {r}");
    }

    /// Rounding semantics: up >= down, difference at most 1 per division
    /// stage; amount1 exact when divisible.
    #[test]
    fn amount_delta_rounding() {
        let s1 = get_sqrt_ratio_at_tick(-100);
        let s2 = get_sqrt_ratio_at_tick(100);
        let liq = 1_000_000_000_000u128;
        let a1u = amount1_delta(liq, s1, s2, true);
        let a1d = amount1_delta(liq, s1, s2, false);
        assert!(a1u >= a1d && a1u - a1d <= U256::from(1u8));
        let a0u = amount0_delta(liq, s1, s2, true);
        let a0d = amount0_delta(liq, s1, s2, false);
        assert!(a0u >= a0d && a0u - a0d <= U256::from(2u8)); // two stages
        // exact divisibility: liq = Q96 multiple makes amount1 exact
        let liq2 = 1u128 << 96;
        let e_up = amount1_delta(liq2, s1, s2, true);
        let e_dn = amount1_delta(liq2, s1, s2, false);
        assert_eq!(e_up, e_dn);
    }

    #[test]
    fn isqrt_exact_and_floor() {
        for v in [0u128, 1, 2, 3, 4, 15, 16, 17, u128::MAX] {
            let n = U512::from(v);
            let r = isqrt_512(n);
            assert!(r * r <= n);
            let r1 = r + U512::from(1u8);
            assert!(r1 * r1 > n);
        }
        // large: (2^200)^2 = 2^400
        let big = U512::from(1u8) << 400;
        assert_eq!(isqrt_512(big), U512::from(1u8) << 200);
    }

    /// Round-18 rounding semantics: the boundary must never sit strictly
    /// INSIDE the epsilon band — up-boundary^2 >= s^2(1+eps),
    /// down-boundary^2 <= s^2(1-eps).
    #[test]
    fn eps_boundary_never_inside_band() {
        for tick in [-100_000, -1, 0, 7, 100_000] {
            let s = get_sqrt_ratio_at_tick(tick);
            let s2: U512 = s.widening_mul(s);
            for eps in [10u32, 50, 100] {
                let up = eps_boundary(s, eps, true);
                let up2: U512 = up.widening_mul(up);
                assert!(
                    up2 * U512::from(10_000u32) >= s2 * U512::from(10_000 + eps),
                    "up boundary inside band at tick {tick} eps {eps}"
                );
                let dn = eps_boundary(s, eps, false);
                let dn2: U512 = dn.widening_mul(dn);
                assert!(
                    dn2 * U512::from(10_000u32) <= s2 * U512::from(10_000 - eps),
                    "down boundary inside band at tick {tick} eps {eps}"
                );
            }
        }
    }

    #[test]
    fn eps_boundary_brackets_price() {
        let s = get_sqrt_ratio_at_tick(0); // price 1.0
        let up = eps_boundary(s, 100, true); // +1% price
        let dn = eps_boundary(s, 100, false); // -1% price
        // (up/s)^2 ~ 1.01, (dn/s)^2 ~ 0.99 to within isqrt floor
        let up2: U512 = up.widening_mul(up);
        let s2: U512 = s.widening_mul(s);
        let num = up2 * U512::from(10u8).pow(U512::from(6u8)) / s2;
        let r: u128 = num.try_into().unwrap();
        assert!((r as i128 - 1_010_000).abs() <= 1, "up ratio {r}");
        let dn2: U512 = dn.widening_mul(dn);
        let num = dn2 * U512::from(10u8).pow(U512::from(6u8)) / s2;
        let r: u128 = num.try_into().unwrap();
        assert!((r as i128 - 990_000).abs() <= 1, "down ratio {r}");
    }
}

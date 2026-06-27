use crate::error::AmmError;

pub type TokenAmount = u128;
pub type BasisPoints = u16;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TokenId {
    X,
    Y,
}

pub fn mul_div(a: u128, b: u128, c: u128) -> Result<u128, AmmError> {
    if c == 0 {
        return Err(AmmError::DivisionByZero);
    }
    match a.checked_mul(b) {
        Some(ab) => Ok(ab / c),
        None => {
            let part1 = (a / c).checked_mul(b);
            let part2 = (a % c).checked_mul(b).map(|x| x / c);
            match (part1, part2) {
                (Some(p1), Some(p2)) => p1.checked_add(p2).ok_or(AmmError::Overflow),
                _ => Err(AmmError::Overflow),
            }
        }
    }
}

pub fn isqrt(n: u128) -> u128 {
    if n == 0 {
        return 0;
    }
    let mut x = n;
    let mut x1 = (x + n / x) / 2;

    while x > x1 {
        x = x1;
        x1 = (x + n / x) / 2;
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mul_div_basic() {
        // 6 * 4 / 3 = 8
        assert_eq!(mul_div(6, 4, 3).unwrap(), 8);
    }

    #[test]
    fn test_mul_div_zero_denominator() {
        assert!(mul_div(6, 4, 0).is_err());
    }

    #[test]
    fn test_mul_div_overflow_fallback() {
        assert_eq!(mul_div(u128::MAX, 2, u128::MAX).unwrap(), 2);
    }

    #[test]
    fn test_isqrt_basic() {
        assert_eq!(isqrt(0), 0);
        assert_eq!(isqrt(1), 1);
        assert_eq!(isqrt(4), 2);
        assert_eq!(isqrt(9), 3);
        assert_eq!(isqrt(16), 4);
    }

    #[test]
    fn test_isqrt_non_perfect_square() {
        assert_eq!(isqrt(2), 1);
        assert_eq!(isqrt(8), 2);
        assert_eq!(isqrt(10), 3);
    }
}

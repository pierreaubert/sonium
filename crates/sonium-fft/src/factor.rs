//! Length factorisation: choose the radix sequence for a given FFT size.
//!
//! Small "smooth" lengths (factors limited to {2, 3, 4, 5}) decompose into
//! a mixed-radix Stockham sequence. Anything else falls back to Bluestein
//! (which itself wraps a smooth-length FFT internally). Radix-4 is
//! preferred over radix-2 for power-of-2 sizes because it halves the
//! twiddle count.

/// Strategy chosen by [`plan_length`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Strategy {
    /// Smooth length: ordered list of radices whose product is `N`.
    MixedRadix(Vec<usize>),
    /// Non-smooth: Bluestein wrap. The padded inner length is the next
    /// power-of-two `M >= 2N - 1`.
    Bluestein { padded_len: usize },
}

/// Failure while validating an FFT length before plan construction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlanLengthError {
    /// A zero-length transform has no meaningful DFT contract.
    ZeroLength,
    /// Bluestein's padded convolution length cannot be represented by `usize`.
    LengthOverflow { length: usize },
}

impl std::fmt::Display for PlanLengthError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ZeroLength => f.write_str("FFT length must be positive"),
            Self::LengthOverflow { length } => {
                write!(f, "FFT length {length} overflows Bluestein convolution")
            }
        }
    }
}

impl std::error::Error for PlanLengthError {}

// Radix-8 stages exist (scalar codelet; AVX2 streaming kernels were
// measured far slower than the scalar loop on x86_64 and removed), but
// the default factorisation stays radix-4-first on every backend:
// A/B measurements showed radix-8-first loses more at 1024/4096/2M
// (fused radix-4 audio paths, stream thrashing) than it gains at
// 512/2048. Radix-8 remains reachable through explicit stage
// construction.
const SUPPORTED_RADICES: &[usize] = &[5, 4, 3, 2];

/// Pick a factorisation of `n` for the Stockham driver. Returns
/// `Strategy::MixedRadix` when `n` is a product of `{2, 3, 4, 5}`,
/// otherwise `Strategy::Bluestein` with the next pow2 of `2n - 1`.
pub fn plan_length(n: usize) -> Strategy {
    try_plan_length(n).unwrap_or_else(|error| panic!("{error}"))
}

/// Checked form of [`plan_length`] suitable for fallible planner boundaries.
pub fn try_plan_length(n: usize) -> Result<Strategy, PlanLengthError> {
    if n == 0 {
        return Err(PlanLengthError::ZeroLength);
    }
    if n == 1 {
        return Ok(Strategy::MixedRadix(Vec::new()));
    }
    if let Some(factors) = try_smooth(n) {
        return Ok(Strategy::MixedRadix(factors));
    }
    let convolution_len = n
        .checked_mul(2)
        .and_then(|value| value.checked_sub(1))
        .ok_or(PlanLengthError::LengthOverflow { length: n })?;
    let m = convolution_len
        .checked_next_power_of_two()
        .ok_or(PlanLengthError::LengthOverflow { length: n })?;
    Ok(Strategy::Bluestein { padded_len: m })
}

fn try_smooth(mut n: usize) -> Option<Vec<usize>> {
    let mut out = Vec::new();
    for &r in SUPPORTED_RADICES {
        while n.is_multiple_of(r) {
            out.push(r);
            n /= r;
        }
    }
    if n == 1 { Some(out) } else { None }
}

/// Smallest power of two `>= n`. `next_pow2(0) == next_pow2(1) == 1`.
///
/// Panics if the result overflows `usize`. Fallible planner boundaries must
/// use [`try_next_pow2_pub`] (or [`try_plan_length`]), which reports the same
/// overflow as [`PlanLengthError::LengthOverflow`] instead of panicking.
pub fn next_pow2_pub(n: usize) -> usize {
    next_pow2(n)
}

/// Checked form of [`next_pow2_pub`]: the overflow that panics there is
/// reported as [`PlanLengthError::LengthOverflow`] here, so fallible plan
/// constructors can route every arithmetic failure through one error type.
pub fn try_next_pow2_pub(n: usize) -> Result<usize, PlanLengthError> {
    if n <= 1 {
        return Ok(1);
    }
    let shift = usize::BITS as usize - (n - 1).leading_zeros() as usize;
    1usize
        .checked_shl(shift as u32)
        .ok_or(PlanLengthError::LengthOverflow { length: n })
}

fn next_pow2(n: usize) -> usize {
    try_next_pow2_pub(n).expect("FFT power-of-two length overflows usize")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn powers_of_two_use_radix_4_first() {
        match plan_length(1024) {
            Strategy::MixedRadix(f) => {
                // 1024 = 4^5 = five radix-4 stages.
                assert!(f.iter().all(|&r| r == 4), "got {f:?}");
                assert_eq!(f.len(), 5);
            }
            other => panic!("expected MixedRadix, got {other:?}"),
        }
    }

    #[test]
    fn radix_2_picks_up_residual_factor() {
        // 128 = 4^3 * 2: the residual 2 must be picked up after the
        // larger radices are exhausted.
        match plan_length(128) {
            Strategy::MixedRadix(f) => {
                let prod: usize = f.iter().product();
                assert_eq!(prod, 128);
                assert!(f.contains(&2));
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn smooth_mixed_factors() {
        // 60 = 5 * 4 * 3
        match plan_length(60) {
            Strategy::MixedRadix(f) => {
                let prod: usize = f.iter().product();
                assert_eq!(prod, 60);
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn prime_uses_bluestein() {
        // 7 has no factor in {2,3,4,5}.
        match plan_length(7) {
            Strategy::Bluestein { padded_len } => {
                assert!(padded_len >= 2 * 7 - 1);
                assert!(padded_len.is_power_of_two());
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn length_one_is_identity() {
        match plan_length(1) {
            Strategy::MixedRadix(v) => assert!(v.is_empty()),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn checked_length_rejects_zero_without_panicking() {
        assert_eq!(try_plan_length(0), Err(PlanLengthError::ZeroLength));
    }

    #[test]
    fn next_pow2_works() {
        assert_eq!(next_pow2(1), 1);
        assert_eq!(next_pow2(2), 2);
        assert_eq!(next_pow2(3), 4);
        assert_eq!(next_pow2(5), 8);
        assert_eq!(next_pow2(13), 16);
        assert_eq!(next_pow2(1024), 1024);
    }

    #[test]
    fn checked_next_pow2_reports_overflow_without_panicking() {
        assert_eq!(try_next_pow2_pub(0), Ok(1));
        assert_eq!(try_next_pow2_pub(1), Ok(1));
        assert_eq!(try_next_pow2_pub(13), Ok(16));
        // `usize::MAX` needs a power of two that does not fit in `usize`.
        assert_eq!(
            try_next_pow2_pub(usize::MAX),
            Err(PlanLengthError::LengthOverflow { length: usize::MAX })
        );
        // The panicking wrapper agrees with the checked form wherever the
        // result is representable.
        for n in [0usize, 1, 2, 3, 5, 13, 1024, 1 << 40] {
            assert_eq!(next_pow2_pub(n), try_next_pow2_pub(n).unwrap());
        }
    }
}

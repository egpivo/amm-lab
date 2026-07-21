//! Running per-tick liquidity book with a bisect-cached cumulative.
//!
//! Port of the `Book` class in the reference `build_outcomes.py` v3. Depth queries at
//! swaps must be cheap (there can be hundreds of thousands of swaps between two
//! liquidity events), so the cumulative `sum(net) for tick <= t` is cached and rebuilt
//! only when a mint/burn dirties the book. `active_L(t)` is then a binary search.
//!
//! Ticks are Uniswap v3 ticks (fit in `i32`; range +/-887272). `liquidityNet` per tick
//! is signed and accumulated in `i128` (a position contributes +L at its lower tick and
//! -L at its upper tick); active in-range liquidity below a price is the running sum and
//! is clamped at zero.

use std::collections::HashMap;

#[derive(Default)]
pub struct Book {
    net: HashMap<i32, i128>,
    dirty: bool,
    ticks: Vec<i32>, // sorted ticks with nonzero net
    cum: Vec<i128>,  // prefix cumulative of net over `ticks`
}

impl Book {
    pub fn new() -> Self {
        Book {
            net: HashMap::new(),
            dirty: true,
            ticks: Vec::new(),
            cum: Vec::new(),
        }
    }

    /// Apply a signed liquidity delta at a tick (mint: +L at lower / -L at upper).
    pub fn apply(&mut self, tick: i32, delta: i128) {
        *self.net.entry(tick).or_insert(0) += delta;
        self.dirty = true;
    }

    fn rebuild(&mut self) {
        self.ticks = self
            .net
            .iter()
            .filter(|&(_, &v)| v != 0)
            .map(|(&t, _)| t)
            .collect();
        self.ticks.sort_unstable();
        self.cum.clear();
        let mut s: i128 = 0;
        for &t in &self.ticks {
            s += self.net[&t];
            self.cum.push(s);
        }
        self.dirty = false;
    }

    /// Initialized ticks `t` with `lo < t <= hi` (ascending) and their net
    /// liquidity — the crossing ladder for piecewise band integration
    /// (LVR deep-subset calibration; additive, does not affect panel code).
    pub fn crossings(&mut self, lo: i32, hi: i32) -> Vec<(i32, i128)> {
        if self.dirty {
            self.rebuild();
        }
        let i = self.ticks.partition_point(|&t| t <= lo);
        let j = self.ticks.partition_point(|&t| t <= hi);
        self.ticks[i..j]
            .iter()
            .map(|&t| (t, self.net[&t]))
            .collect()
    }

    /// Active in-range liquidity at `tick` = sum of net over all ticks <= `tick`,
    /// clamped at zero. Rebuilds the cumulative only if the book changed since last call.
    pub fn active_l(&mut self, tick: i32) -> i128 {
        if self.dirty {
            self.rebuild();
        }
        if self.ticks.is_empty() {
            return 0;
        }
        // index of first tick strictly greater than `tick` (upper_bound / bisect_right)
        let i = self.ticks.partition_point(|&t| t <= tick);
        if i == 0 { 0 } else { self.cum[i - 1].max(0) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overlapping_ranges_and_out_of_range() {
        // position spanning [-100, 100] with L=1000
        let mut b = Book::new();
        b.apply(-100, 1000);
        b.apply(100, -1000);
        assert_eq!(b.active_l(-200), 0); // below lower
        assert_eq!(b.active_l(0), 1000); // in range
        assert_eq!(b.active_l(50), 1000);
        assert_eq!(b.active_l(150), 0); // above upper: net back to 0

        // add overlapping [0, 200] with L=500
        b.apply(0, 500);
        b.apply(200, -500);
        assert_eq!(b.active_l(-50), 1000);
        assert_eq!(b.active_l(50), 1500); // both ranges active
        assert_eq!(b.active_l(150), 500);
    }

    #[test]
    fn cache_invalidates_on_apply() {
        let mut b = Book::new();
        b.apply(0, 100);
        b.apply(10, -100);
        assert_eq!(b.active_l(5), 100);
        // mutate after a query -> cache must rebuild
        b.apply(0, 50);
        assert_eq!(b.active_l(5), 150);
    }

    #[test]
    fn empty_book_is_zero() {
        let mut b = Book::new();
        assert_eq!(b.active_l(0), 0);
    }
}

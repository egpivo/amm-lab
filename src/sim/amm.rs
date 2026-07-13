//! Constant-product pool with directional (buy/sell) fees.
//!
//! Fee convention: fees accrue to a separate LP fee bucket, never to reserves,
//! so the invariant k = reserve_x * reserve_y is preserved exactly by every
//! trade. This makes conservation testable and keeps the arbitrage closed
//! form exact.
//!
//! Units: X is the numeraire, Y the risky asset. Prices are X per Y.

#[derive(Debug, Clone)]
pub struct Pool {
    pub reserve_x: f64,
    pub reserve_y: f64,
    /// Fee charged (in X) when a trader buys Y from the pool.
    pub fee_buy: f64,
    /// Fee charged (in X) when a trader sells Y to the pool.
    pub fee_sell: f64,
    /// Cumulative fee revenue in X units (LP bucket, outside reserves).
    pub fee_revenue_x: f64,
}

/// Result of an executed trade, from the trader's perspective.
#[derive(Debug, Clone, Copy)]
pub struct Fill {
    /// Y received (buy) or delivered (sell).
    pub qty_y: f64,
    /// X paid (buy) or received (sell), fee included.
    pub amount_x: f64,
    /// Fee portion of `amount_x`.
    pub fee_x: f64,
}

impl Pool {
    pub fn new(reserve_x: f64, reserve_y: f64, fee_buy: f64, fee_sell: f64) -> Self {
        Self {
            reserve_x,
            reserve_y,
            fee_buy,
            fee_sell,
            fee_revenue_x: 0.0,
        }
    }

    /// Marginal (mid) price, X per Y, ignoring fees.
    pub fn mid_price(&self) -> f64 {
        self.reserve_x / self.reserve_y
    }

    /// Signed inventory imbalance in [-1, 1]: positive when the pool is
    /// X-heavy relative to the oracle valuation.
    pub fn inventory_imbalance(&self, oracle_price: f64) -> f64 {
        let y_value = self.reserve_y * oracle_price;
        (self.reserve_x - y_value) / (self.reserve_x + y_value)
    }

    /// Total X a trader must pay to buy `qty_y` of Y (fee included).
    /// Returns None if the trade is infeasible.
    pub fn buy_cost(&self, qty_y: f64) -> Option<Fill> {
        if qty_y <= 0.0 || qty_y >= self.reserve_y {
            return None;
        }
        let dx_net = self.reserve_x * qty_y / (self.reserve_y - qty_y);
        let dx_total = dx_net / (1.0 - self.fee_buy);
        Some(Fill {
            qty_y,
            amount_x: dx_total,
            fee_x: dx_total - dx_net,
        })
    }

    /// Total X a trader receives for selling `qty_y` of Y (fee deducted).
    pub fn sell_proceeds(&self, qty_y: f64) -> Option<Fill> {
        if qty_y <= 0.0 {
            return None;
        }
        let dx_gross = self.reserve_x * qty_y / (self.reserve_y + qty_y);
        let fee = dx_gross * self.fee_sell;
        Some(Fill {
            qty_y,
            amount_x: dx_gross - fee,
            fee_x: fee,
        })
    }

    /// Execute a buy of `qty_y`; reserves move by the net amounts, fee goes
    /// to the LP bucket.
    pub fn buy(&mut self, qty_y: f64) -> Option<Fill> {
        let fill = self.buy_cost(qty_y)?;
        self.reserve_x += fill.amount_x - fill.fee_x;
        self.reserve_y -= qty_y;
        self.fee_revenue_x += fill.fee_x;
        Some(fill)
    }

    /// Execute a sell of `qty_y`.
    pub fn sell(&mut self, qty_y: f64) -> Option<Fill> {
        let fill = self.sell_proceeds(qty_y)?;
        self.reserve_x -= fill.amount_x + fill.fee_x;
        self.reserve_y += qty_y;
        self.fee_revenue_x += fill.fee_x;
        Some(fill)
    }

    /// Effective per-unit buy price for a trade of `qty_y` (fee included).
    pub fn effective_buy_price(&self, qty_y: f64) -> Option<f64> {
        self.buy_cost(qty_y).map(|f| f.amount_x / f.qty_y)
    }

    /// Effective per-unit sell price for a trade of `qty_y` (fee deducted).
    pub fn effective_sell_price(&self, qty_y: f64) -> Option<f64> {
        self.sell_proceeds(qty_y).map(|f| f.amount_x / f.qty_y)
    }

    pub fn pool_value(&self, oracle_price: f64) -> f64 {
        self.reserve_x + self.reserve_y * oracle_price
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buy_preserves_invariant() {
        let mut p = Pool::new(1_000_000.0, 1_000.0, 0.003, 0.003);
        let k = p.reserve_x * p.reserve_y;
        p.buy(10.0).unwrap();
        assert!((p.reserve_x * p.reserve_y - k).abs() / k < 1e-12);
    }

    #[test]
    fn sell_preserves_invariant() {
        let mut p = Pool::new(1_000_000.0, 1_000.0, 0.003, 0.003);
        let k = p.reserve_x * p.reserve_y;
        p.sell(10.0).unwrap();
        assert!((p.reserve_x * p.reserve_y - k).abs() / k < 1e-12);
    }

    #[test]
    fn larger_buys_cost_more_per_unit() {
        let p = Pool::new(1_000_000.0, 1_000.0, 0.003, 0.003);
        let small = p.effective_buy_price(1.0).unwrap();
        let large = p.effective_buy_price(100.0).unwrap();
        assert!(large > small);
        assert!(small > p.mid_price());
    }

    #[test]
    fn round_trip_loses_fees() {
        let mut p = Pool::new(1_000_000.0, 1_000.0, 0.003, 0.003);
        let buy = p.buy(10.0).unwrap();
        let sell = p.sell(10.0).unwrap();
        assert!(sell.amount_x < buy.amount_x);
    }
}

pub struct CampbellPool {
    pub reserve_x: f64,
    pub reserve_y: f64,
    pub amm_fee: f64,
    pub cumulative_fee_revenue: f64,
}

impl CampbellPool {
    pub fn new(reserve_x: f64, reserve_y: f64, amm_fee: f64) -> Self {
        Self {
            reserve_x,
            reserve_y,
            amm_fee,
            cumulative_fee_revenue: 0.0,
        }
    }
    pub fn marginal_price(&self) -> f64 {
        self.reserve_x / self.reserve_y
    }
    pub fn pool_value(&self, cex_price: f64) -> f64 {
        self.reserve_x + self.reserve_y * cex_price
    }
    pub fn apply_delta(&mut self, delta_y: f64) -> f64 {
        if delta_y == 0.0 {
            return 0.0;
        }
        let exec_price = self.reserve_x / (self.reserve_y - delta_y);
        let x_change = exec_price * delta_y;
        let fee = x_change.abs() * self.amm_fee;

        self.reserve_x += x_change;
        self.reserve_y -= delta_y;
        self.cumulative_fee_revenue += fee;
        fee
    }
}

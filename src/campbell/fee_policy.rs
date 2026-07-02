pub struct FeeObservation {
    pub step: usize,
    pub external_price: f64,
    pub amm_price: f64,
    pub oracle_gap_bps: f64,
    pub inventory_skew: f64,
    pub recent_vol: f64,
}

pub trait FeePolicy {
    fn name(&self) -> &'static str;
    fn fee(&mut self, obs: &FeeObservation) -> f64;
}

pub struct FixedFeePolicy {
    pub fee: f64,
}

impl FixedFeePolicy {
    pub fn new(fee: f64) -> Self {
        Self { fee }
    }
}

impl FeePolicy for FixedFeePolicy {
    fn name(&self) -> &'static str {
        "fixed"
    }
    fn fee(&mut self, _obs: &FeeObservation) -> f64 {
        self.fee
    }
}

pub struct OracleGapFeePolicy {
    pub base_fee: f64,
    pub gap_multiplier: f64,
    pub min_fee: f64,
    pub max_fee: f64,
}

impl FeePolicy for OracleGapFeePolicy {
    fn name(&self) -> &'static str {
        "oracle_gap"
    }
    fn fee(&mut self, obs: &FeeObservation) -> f64 {
        let f = self.base_fee + self.gap_multiplier * obs.oracle_gap_bps.abs() / 10_000.0;
        f.clamp(self.min_fee, self.max_fee)
    }
}

pub struct InventoryGapFeePolicy {
    pub base_fee: f64,
    pub gap_multiplier: f64,
    pub min_fee: f64,
    pub max_fee: f64,
}

impl FeePolicy for InventoryGapFeePolicy {
    fn name(&self) -> &'static str {
        "inventory_gap"
    }
    fn fee(&mut self, obs: &FeeObservation) -> f64 {
        let f = self.base_fee + self.gap_multiplier * obs.inventory_skew.abs();
        f.clamp(self.min_fee, self.max_fee)
    }
}

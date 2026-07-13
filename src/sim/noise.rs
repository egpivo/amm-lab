//! Exogenous noise-trader flow with weak fee/price sensitivity.
//!
//! Each step draws a total buy volume and sell volume (lognormal around a
//! base intensity), then splits each between the live pools by a softmax
//! over effective execution cost, so cheaper pools attract more flow but
//! routing is not perfectly efficient.

use crate::sim::amm::Pool;
use rand::Rng;
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoiseConfig {
    /// Mean Y volume bought per step across pools.
    pub buy_intensity: f64,
    /// Mean Y volume sold per step across pools.
    pub sell_intensity: f64,
    /// Lognormal sigma of per-step volume.
    pub volume_sigma: f64,
    /// Routing sensitivity to effective-cost differences (per bps).
    pub route_sensitivity: f64,
    /// Reference clip size used to probe effective prices.
    pub probe_size: f64,
}

impl Default for NoiseConfig {
    fn default() -> Self {
        Self {
            buy_intensity: 2.0,
            sell_intensity: 2.0,
            volume_sigma: 0.5,
            route_sensitivity: 0.05,
            probe_size: 1.0,
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct NoiseFlows {
    /// Buy volume routed to each pool (Y units).
    pub buys: [f64; 2],
    /// Sell volume routed to each pool (Y units).
    pub sells: [f64; 2],
}

/// Softmax weight on pool 0 given per-unit costs (lower cost => higher weight).
fn route_share(cost_a: f64, cost_b: f64, sensitivity: f64, oracle: f64) -> f64 {
    let gap_bps = (cost_b - cost_a) / oracle * 10_000.0;
    1.0 / (1.0 + (-sensitivity * gap_bps).exp())
}

pub fn noise_flows(
    cfg: &NoiseConfig,
    pools: &[Pool],
    oracle_price: f64,
    rng: &mut StdRng,
) -> NoiseFlows {
    let draw = |rng: &mut StdRng, mean: f64| -> f64 {
        let z: f64 = rng.gen_range(-1.0..1.0);
        (mean * (cfg.volume_sigma * z).exp()).max(0.0)
    };
    let total_buy = draw(rng, cfg.buy_intensity);
    let total_sell = draw(rng, cfg.sell_intensity);

    let mut flows = NoiseFlows::default();
    if pools.len() == 1 {
        flows.buys[0] = total_buy;
        flows.sells[0] = total_sell;
        return flows;
    }

    let buy_a = pools[0]
        .effective_buy_price(cfg.probe_size)
        .unwrap_or(f64::INFINITY);
    let buy_b = pools[1]
        .effective_buy_price(cfg.probe_size)
        .unwrap_or(f64::INFINITY);
    let share_a = route_share(buy_a, buy_b, cfg.route_sensitivity, oracle_price);
    flows.buys = [total_buy * share_a, total_buy * (1.0 - share_a)];

    // For sells, a *higher* effective sell price is better, so flip the sign.
    let sell_a = pools[0].effective_sell_price(cfg.probe_size).unwrap_or(0.0);
    let sell_b = pools[1].effective_sell_price(cfg.probe_size).unwrap_or(0.0);
    let share_a = route_share(-sell_a, -sell_b, cfg.route_sensitivity, oracle_price);
    flows.sells = [total_sell * share_a, total_sell * (1.0 - share_a)];
    flows
}

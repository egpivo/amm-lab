//! External CEX / oracle mid-price process (seeded GBM) plus rolling
//! realized volatility used as a state feature.

use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};

pub fn gbm_path(n_steps: usize, s0: f64, mu: f64, sigma: f64, dt: f64, seed: u64) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let z = Normal::new(0.0, 1.0).unwrap();
    let mut prices = Vec::with_capacity(n_steps + 1);
    prices.push(s0);
    for _ in 0..n_steps {
        let shock: f64 = z.sample(&mut rng);
        let next = prices.last().unwrap()
            * ((mu - 0.5 * sigma * sigma) * dt + sigma * dt.sqrt() * shock).exp();
        prices.push(next);
    }
    prices
}

/// Realized volatility of log returns over the trailing `window` steps,
/// annualized by 1/sqrt(dt). Returns 0.0 until enough history exists.
pub fn rolling_vol(prices: &[f64], t: usize, window: usize, dt: f64) -> f64 {
    if t < 2 {
        return 0.0;
    }
    let start = t.saturating_sub(window);
    let rets: Vec<f64> = (start + 1..=t)
        .map(|i| (prices[i] / prices[i - 1]).ln())
        .collect();
    if rets.len() < 2 {
        return 0.0;
    }
    let mean = rets.iter().sum::<f64>() / rets.len() as f64;
    let var = rets.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (rets.len() - 1) as f64;
    (var / dt).sqrt()
}

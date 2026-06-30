use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};

// S_{t+1} = S_t * exp((μ - σ²/2) * dt + σ * sqrt(dt) * Z)
pub fn generate_gbm(n: usize, s0: f64, mu: f64, sigma: f64, dt: f64, seed: u64) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let z = Normal::new(0.0, 1.0).unwrap();
    let mut prices = Vec::with_capacity(n + 1);
    prices.push(s0);
    for _ in 0..n {
        let z_val = z.sample(&mut rng);
        let s_next = prices.last().unwrap()
            * ((mu - 0.5 * sigma * sigma) * dt + sigma * dt.sqrt() * z_val).exp();
        prices.push(s_next);
    }
    prices
}

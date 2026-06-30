use crate::campbell::pool::CampbellPool;

pub fn arb_delta(pool: &CampbellPool, cex_price: f64, cex_fee: f64) -> f64 {
    let p = pool.reserve_x / pool.reserve_y;
    let ratio1 = p * (1.0 - pool.amm_fee) / (cex_price * (1.0 + cex_fee));
    let ratio2 = p * (1.0 + pool.amm_fee) / (cex_price * (1.0 - cex_fee));
    let val1 = pool.reserve_y * (1.0 - ratio1.sqrt());
    let val2 = pool.reserve_y * (1.0 - ratio2.sqrt());

    val1.min(0.0) + val2.max(0.0)
}

pub fn fundamental_buy_delta(
    demand: f64,
    pool: &CampbellPool,
    cex_price: f64,
    cex_fee: f64,
) -> f64 {
    let p = pool.reserve_x / pool.reserve_y;
    let ratio = p * (1.0 + pool.amm_fee) / (cex_price * (1.0 + cex_fee));
    let val = pool.reserve_y * (1.0 - ratio.sqrt());
    demand.min(val).max(0.0)
}

pub fn fundamental_sell_delta(
    demand: f64,
    pool: &CampbellPool,
    cex_price: f64,
    cex_fee: f64,
) -> f64 {
    let p = pool.reserve_x / pool.reserve_y;
    let ratio = p * (1.0 - pool.amm_fee) / (cex_price * (1.0 - cex_fee));
    let val = pool.reserve_y * (1.0 - ratio.sqrt());
    demand.max(val).min(0.0)
}

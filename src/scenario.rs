use serde::{Deserialize, Serialize};
use std::path::Path;

use crate::arbitrage::{ArbitrageStep, run_arbitrage};
use crate::error::AmmError;
use crate::liquidity::{add_liquidity, remove_liquidity};
use crate::lp_accounting::{LiquidityPosition, compute_lp_performance};
use crate::pool::Pool;
use crate::swap::swap_with_slippage;
use crate::transaction::Transaction;

// ── input ────────────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct Scenario {
    pub name: String,
    pub description: String,
    pub transactions: Vec<Transaction>,
}

// ── rich event types (Serialize → JSON / CSV) ────────────────────────────────

#[derive(Debug, Clone, Serialize)]
pub struct SwapEvent {
    pub step: usize,
    pub actor: String,
    pub direction: String,
    pub amount_in: u128,
    pub amount_out: u128,
    pub fee_amount: u128,
    pub exec_price: f64,
    pub spot_price_before: f64,
    pub price_impact_pct: f64,
    pub reserve_x_after: u128,
    pub reserve_y_after: u128,
    pub invariant_before: u128,
    pub invariant_after: u128,
}

/// One row in the arbitrage-steps CSV; mirrors `ArbitrageStep` with a tx_step prefix.
#[derive(Debug, Clone, Serialize)]
pub struct ArbitrageStepRecord {
    pub tx_step: usize,
    pub step_index: u32,
    pub direction: String,
    pub amount_in: u128,
    pub amount_out: u128,
    pub fee_paid: u128,
    pub profit_estimate: f64,
    pub pool_price_before: f64,
    pub pool_price_after: f64,
    pub external_price: f64,
    pub price_gap_before: f64,
    pub price_gap_after: f64,
    pub reserve_x_after: u128,
    pub reserve_y_after: u128,
}

impl ArbitrageStepRecord {
    fn from_step(tx_step: usize, s: &ArbitrageStep) -> Self {
        ArbitrageStepRecord {
            tx_step,
            step_index: s.step_index,
            direction: format!("{:?}", s.direction),
            amount_in: s.amount_in,
            amount_out: s.amount_out,
            fee_paid: s.fee_paid,
            profit_estimate: s.profit_estimate,
            pool_price_before: s.pool_price_before,
            pool_price_after: s.pool_price_after,
            external_price: s.external_price,
            price_gap_before: s.price_gap_before,
            price_gap_after: s.price_gap_after,
            reserve_x_after: s.reserve_x_after,
            reserve_y_after: s.reserve_y_after,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ArbitrageEvent {
    pub step: usize,
    pub steps_executed: usize,
    pub pool_price_before: f64,
    pub pool_price_after: f64,
    pub external_price: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct LpPerformanceEvent {
    pub actor: String,
    pub withdraw_x: u128,
    pub withdraw_y: u128,
    /// Pro-rata share of total fees valued at external_price. Proxy only.
    pub fee_income_value_in_y: f64,
    pub hold_value_in_y: f64,
    pub lp_value_in_y: f64,
    pub impermanent_loss_pct: f64,
    pub net_profit_loss_in_y: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct PoolSnapshot {
    pub reserve_x: u128,
    pub reserve_y: u128,
    pub lp_supply: u128,
    pub spot_price: f64,
    pub invariant: u128,
    pub fee_x_accumulated: u128,
    pub fee_y_accumulated: u128,
}

impl PoolSnapshot {
    pub fn from_pool(pool: &Pool) -> Self {
        PoolSnapshot {
            reserve_x: pool.reserve_x,
            reserve_y: pool.reserve_y,
            lp_supply: pool.lp_supply,
            spot_price: pool.spot_price(),
            invariant: pool.invariant(),
            fee_x_accumulated: pool.fee_x_accumulated,
            fee_y_accumulated: pool.fee_y_accumulated,
        }
    }
}

// ── report ───────────────────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
pub struct ScenarioReport {
    pub scenario_name: String,
    pub description: String,
    pub run_timestamp: String,
    pub final_pool: Option<PoolSnapshot>,
    pub swap_events: Vec<SwapEvent>,
    pub arbitrage_events: Vec<ArbitrageEvent>,
    /// All individual arbitrage steps across all ArbitrageUntilNoProfit transactions.
    pub arbitrage_steps: Vec<ArbitrageStepRecord>,
    pub lp_performance: Option<LpPerformanceEvent>,
    pub log: Vec<String>,
}

// ── runner ───────────────────────────────────────────────────────────────────

pub fn run_scenario(scenario: &Scenario) -> Result<ScenarioReport, AmmError> {
    let mut pool: Option<Pool> = None;
    let mut external_price: f64 = 1.0;
    let mut lp_position: Option<(String, LiquidityPosition)> = None;

    let mut swap_events: Vec<SwapEvent> = Vec::new();
    let mut arbitrage_events: Vec<ArbitrageEvent> = Vec::new();
    let mut arbitrage_steps: Vec<ArbitrageStepRecord> = Vec::new();
    let mut lp_performance: Option<LpPerformanceEvent> = None;
    let mut log: Vec<String> = Vec::new();
    let mut step = 0usize;

    for tx in &scenario.transactions {
        step += 1;
        match tx {
            Transaction::CreatePool {
                reserve_x,
                reserve_y,
                fee_bps,
            } => {
                pool = Some(Pool::new(*reserve_x, *reserve_y, *fee_bps)?);
                lp_position = None;
                log.push(format!(
                    "[{step}] CreatePool reserve_x={reserve_x} reserve_y={reserve_y} fee_bps={fee_bps}"
                ));
            }

            Transaction::AddLiquidity {
                actor,
                amount_x,
                amount_y,
            } => {
                let p = pool.as_mut().ok_or(AmmError::EmptyPool)?;
                let snap_before = PoolSnapshot::from_pool(p);
                let result = add_liquidity(p, *amount_x, *amount_y)?;
                if lp_position.is_none() {
                    lp_position = Some((
                        format!("{actor:?}"),
                        LiquidityPosition {
                            lp_shares: result.lp_minted,
                            entry_reserve_x: snap_before.reserve_x,
                            entry_reserve_y: snap_before.reserve_y,
                            entry_lp_supply: snap_before.lp_supply,
                        },
                    ));
                }
                log.push(format!(
                    "[{step}] AddLiquidity actor={actor:?} lp_minted={}",
                    result.lp_minted
                ));
            }

            Transaction::RemoveLiquidity { actor, lp_shares } => {
                let p = pool.as_mut().ok_or(AmmError::EmptyPool)?;
                let result = remove_liquidity(p, *lp_shares)?;
                log.push(format!(
                    "[{step}] RemoveLiquidity actor={actor:?} x={} y={}",
                    result.amount_x, result.amount_y
                ));
            }

            Transaction::SwapExactIn {
                actor,
                direction,
                amount_in,
                min_amount_out,
            } => {
                let p = pool.as_mut().ok_or(AmmError::EmptyPool)?;
                let receipt = swap_with_slippage(p, *direction, *amount_in, *min_amount_out)?;
                let ev = SwapEvent {
                    step,
                    actor: format!("{actor:?}"),
                    direction: format!("{direction:?}"),
                    amount_in: receipt.quote.amount_in,
                    amount_out: receipt.quote.amount_out,
                    fee_amount: receipt.quote.fee_amount,
                    exec_price: receipt.quote.exec_price,
                    spot_price_before: receipt.quote.spot_price_before,
                    price_impact_pct: receipt.quote.price_impact_pct,
                    reserve_x_after: receipt.reserve_x_after,
                    reserve_y_after: receipt.reserve_y_after,
                    invariant_before: receipt.quote.invariant_before,
                    invariant_after: receipt.quote.invariant_after,
                };
                log.push(format!(
                    "[{step}] SwapExactIn actor={actor:?} direction={direction:?} \
                     amount_in={} amount_out={} exec_price={:.6} impact={:.4}%",
                    ev.amount_in, ev.amount_out, ev.exec_price, ev.price_impact_pct
                ));
                swap_events.push(ev);
            }

            Transaction::ExternalPriceMove { new_price } => {
                external_price = *new_price;
                log.push(format!(
                    "[{step}] ExternalPriceMove new_price={new_price:.6}"
                ));
            }

            Transaction::ArbitrageUntilNoProfit { max_steps } => {
                let p = pool.as_mut().ok_or(AmmError::EmptyPool)?;
                let price_before = p.spot_price();
                let arb_steps = run_arbitrage(p, external_price, *max_steps);
                let price_after = p.spot_price();

                for s in &arb_steps {
                    arbitrage_steps.push(ArbitrageStepRecord::from_step(step, s));
                }

                let ev = ArbitrageEvent {
                    step,
                    steps_executed: arb_steps.len(),
                    pool_price_before: price_before,
                    pool_price_after: price_after,
                    external_price,
                };
                log.push(format!(
                    "[{step}] ArbitrageUntilNoProfit steps={} price_before={:.6} price_after={:.6}",
                    ev.steps_executed, ev.pool_price_before, ev.pool_price_after
                ));
                arbitrage_events.push(ev);
            }

            Transaction::ReportLpPerformance { actor } => {
                let p = pool.as_ref().ok_or(AmmError::EmptyPool)?;
                if let Some((ref lp_actor, ref pos)) = lp_position {
                    let report = compute_lp_performance(pos, p, external_price)?;
                    let ev = LpPerformanceEvent {
                        actor: lp_actor.clone(),
                        withdraw_x: report.withdraw_x,
                        withdraw_y: report.withdraw_y,
                        fee_income_value_in_y: report.fee_income_value_in_y,
                        hold_value_in_y: report.hold_value_in_y,
                        lp_value_in_y: report.lp_value_in_y,
                        impermanent_loss_pct: report.impermanent_loss_pct,
                        net_profit_loss_in_y: report.net_profit_loss_in_y,
                    };
                    log.push(format!(
                        "[{step}] LpPerformance actor={} fee_proxy={:.2} hold={:.2} lp={:.2} il={:.4}% net={:.2}",
                        ev.actor,
                        ev.fee_income_value_in_y,
                        ev.hold_value_in_y,
                        ev.lp_value_in_y,
                        ev.impermanent_loss_pct,
                        ev.net_profit_loss_in_y
                    ));
                    lp_performance = Some(ev);
                } else {
                    log.push(format!(
                        "[{step}] LpPerformance actor={actor:?} — no position recorded"
                    ));
                }
            }
        }
    }

    Ok(ScenarioReport {
        scenario_name: scenario.name.clone(),
        description: scenario.description.clone(),
        run_timestamp: chrono::Utc::now().to_rfc3339(),
        final_pool: pool.as_ref().map(PoolSnapshot::from_pool),
        swap_events,
        arbitrage_events,
        arbitrage_steps,
        lp_performance,
        log,
    })
}

// ── I/O ──────────────────────────────────────────────────────────────────────

pub fn load_scenario(path: &Path) -> Result<Scenario, Box<dyn std::error::Error>> {
    let content = std::fs::read_to_string(path)?;
    let scenario: Scenario = toml::from_str(&content)?;
    Ok(scenario)
}

pub fn write_json(
    report: &ScenarioReport,
    dir: &Path,
) -> Result<String, Box<dyn std::error::Error>> {
    std::fs::create_dir_all(dir)?;
    let path = dir.join(format!("{}.json", report.scenario_name));
    let json = serde_json::to_string_pretty(report)?;
    std::fs::write(&path, &json)?;
    Ok(path.to_string_lossy().into_owned())
}

pub fn write_csv_swaps(
    report: &ScenarioReport,
    dir: &Path,
) -> Result<String, Box<dyn std::error::Error>> {
    std::fs::create_dir_all(dir)?;
    let path = dir.join(format!("{}_swaps.csv", report.scenario_name));
    let mut wtr = csv::Writer::from_path(&path)?;
    for ev in &report.swap_events {
        wtr.serialize(ev)?;
    }
    wtr.flush()?;
    Ok(path.to_string_lossy().into_owned())
}

pub fn write_csv_arbitrage(
    report: &ScenarioReport,
    dir: &Path,
) -> Result<String, Box<dyn std::error::Error>> {
    std::fs::create_dir_all(dir)?;
    let path = dir.join(format!("{}_arbitrage.csv", report.scenario_name));
    let mut wtr = csv::Writer::from_path(&path)?;
    for row in &report.arbitrage_steps {
        wtr.serialize(row)?;
    }
    wtr.flush()?;
    Ok(path.to_string_lossy().into_owned())
}

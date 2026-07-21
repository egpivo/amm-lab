//! M2 (lvr paper) summary aggregation over per-step records.
//!
//! Everything here is a pure post-processing function of `Vec<StepRecord>`;
//! the engine itself only records primitive per-fill quantities. Metric
//! definitions are frozen in `.local/lvr/m1_theory.md` (Sec. 8):
//! - A/B are the positive/negative variations of the tracking-gap measure
//!   on PRIMITIVE FILL EVENTS (fill-level, primary); event-netted and
//!   fill-type-netted variants are aggregation-granularity robustness.
//! - Markout is LP-side, MO_{k,h} = delta_k (P_{t_k+h} - pbar_k) against
//!   the CEX mid, size-weighted, fills past the episode end dropped with
//!   fill-count AND volume coverage reported.
//! - Ratios with zero denominators are `None`, never 0.0: a shutdown
//!   policy must not display "adverse loss per served unit = 0".

use crate::campbell::simulation::StepRecord;
use serde::Serialize;

fn ratio(num: f64, den: f64) -> Option<f64> {
    if den > 0.0 { Some(num / den) } else { None }
}

#[derive(Debug, Serialize)]
pub struct LvrSummary {
    // Tracking difference and its Jordan components (fill-level, primary).
    pub l_total: f64,
    pub a_fill: f64,
    pub b_fill: f64,
    // Event-netted granularity (per-step net of all three legs).
    pub a_event: f64,
    pub b_event: f64,
    // Fill-type-netted granularity (arb kept separate; buy/sell netted
    // into ell_fund per step). Ordering: event <= type <= fill for both
    // A and B; all three satisfy A - B = L.
    pub a_type: f64,
    pub b_type: f64,
    // Per-leg splits (arb leg is per-fill nonnegative: l_arb = a_arb).
    pub a_arb: f64,
    pub a_fund: f64,
    pub b_fund: f64,
    // Fees and LP relative surplus (rebates/costs are zero in this engine).
    pub fees_total: f64,
    pub fees_arb: f64,
    pub fees_fund: f64,
    pub u_lp_rel: f64,
    // Service and destination allocation (risky units). None = no
    // potential demand (denominator zero), NOT "share of zero".
    pub served_fund_volume: f64,
    pub potential_volume: f64,
    pub alloc_amm_share: Option<f64>,
    pub alloc_cex_share: Option<f64>,
    pub alloc_unserved_share: Option<f64>,
    // Incidence. `incidence_any_fund_step` is the descriptive "any
    // fundamental leg filled this step" rate; the side-event incidences
    // are per DEMAND EVENT (a step-side with positive potential demand).
    pub incidence_any_fund_step: Option<f64>,
    pub incidence_buy_event: Option<f64>,
    pub incidence_sell_event: Option<f64>,
    pub incidence_pooled_event: Option<f64>,
    // Conditional AMM fill size (mean |delta| over fills > 0).
    pub cond_fill_size_buy: Option<f64>,
    pub cond_fill_size_sell: Option<f64>,
    pub cond_fill_size_pooled: Option<f64>,
    // Per-served-unit adverse ratios, reported SEPARATELY per m1_theory
    // (a single ratio would let the arb-leg numerator mask the source).
    // All None under shutdown (served = 0), never 0.
    pub a_arb_per_served_unit: Option<f64>,
    pub a_fund_per_served_unit: Option<f64>,
    pub a_total_per_served_unit: Option<f64>,
    // Quote accuracy: time-average |log P - log p_amm| (end-of-step).
    pub mean_abs_log_gap: Option<f64>,
    // Cross-check: hedging - pool_value from the final record.
    pub tracking_error: f64,
}

pub fn summarize(records: &[StepRecord]) -> LvrSummary {
    let (mut a_fill, mut b_fill) = (0.0, 0.0);
    let (mut a_event, mut b_event) = (0.0, 0.0);
    let (mut a_type, mut b_type) = (0.0, 0.0);
    let (mut a_arb, mut a_fund, mut b_fund) = (0.0, 0.0, 0.0);
    let (mut fees_total, mut fees_arb, mut fees_fund) = (0.0, 0.0, 0.0);
    let (mut served, mut potential, mut cex_total, mut unserved_total) = (0.0, 0.0, 0.0, 0.0);
    let mut any_fund_steps = 0usize;
    let (mut buy_events, mut buy_fills, mut sell_events, mut sell_fills) = (0u64, 0u64, 0u64, 0u64);
    let (mut buy_fill_vol, mut sell_fill_vol) = (0.0, 0.0);
    let mut gap_sum = 0.0;

    for r in records {
        for ell in [r.ell_arb, r.ell_buy, r.ell_sell] {
            a_fill += ell.max(0.0);
            b_fill += (-ell).max(0.0);
        }
        let ell_all = r.ell_arb + r.ell_buy + r.ell_sell;
        a_event += ell_all.max(0.0);
        b_event += (-ell_all).max(0.0);
        let ell_fund = r.ell_buy + r.ell_sell;
        a_type += r.ell_arb.max(0.0) + ell_fund.max(0.0);
        b_type += (-r.ell_arb).max(0.0) + (-ell_fund).max(0.0);
        a_arb += r.ell_arb; // per-fill nonnegative (Lemma 2 iii)
        a_fund += r.ell_buy.max(0.0) + r.ell_sell.max(0.0);
        b_fund += (-r.ell_buy).max(0.0) + (-r.ell_sell).max(0.0);

        fees_total += r.step_fee;
        fees_arb += r.step_fee_arb;
        fees_fund += r.step_fee_fund;

        let step_served = r.buy_delta.abs() + r.sell_delta.abs();
        served += step_served;
        potential += r.pot_buy + r.pot_sell;
        cex_total += r.cex_buy + r.cex_sell;
        unserved_total += r.unserved_buy + r.unserved_sell;
        if step_served > 0.0 {
            any_fund_steps += 1;
        }
        if r.pot_buy > 0.0 {
            buy_events += 1;
            if r.buy_delta.abs() > 0.0 {
                buy_fills += 1;
                buy_fill_vol += r.buy_delta.abs();
            }
        }
        if r.pot_sell > 0.0 {
            sell_events += 1;
            if r.sell_delta.abs() > 0.0 {
                sell_fills += 1;
                sell_fill_vol += r.sell_delta.abs();
            }
        }
        gap_sum += r.log_gap_abs;
    }

    let l_total = a_fill - b_fill;
    LvrSummary {
        l_total,
        a_fill,
        b_fill,
        a_event,
        b_event,
        a_type,
        b_type,
        a_arb,
        a_fund,
        b_fund,
        fees_total,
        fees_arb,
        fees_fund,
        u_lp_rel: fees_total - l_total,
        served_fund_volume: served,
        potential_volume: potential,
        alloc_amm_share: ratio(served, potential),
        alloc_cex_share: ratio(cex_total, potential),
        alloc_unserved_share: ratio(unserved_total, potential),
        incidence_any_fund_step: ratio(any_fund_steps as f64, records.len() as f64),
        incidence_buy_event: ratio(buy_fills as f64, buy_events as f64),
        incidence_sell_event: ratio(sell_fills as f64, sell_events as f64),
        incidence_pooled_event: ratio(
            (buy_fills + sell_fills) as f64,
            (buy_events + sell_events) as f64,
        ),
        cond_fill_size_buy: ratio(buy_fill_vol, buy_fills as f64),
        cond_fill_size_sell: ratio(sell_fill_vol, sell_fills as f64),
        cond_fill_size_pooled: ratio(
            buy_fill_vol + sell_fill_vol,
            (buy_fills + sell_fills) as f64,
        ),
        a_arb_per_served_unit: ratio(a_arb, served),
        a_fund_per_served_unit: ratio(a_fund, served),
        a_total_per_served_unit: ratio(a_fill, served),
        mean_abs_log_gap: ratio(gap_sum, records.len() as f64),
        tracking_error: records
            .last()
            .map(|r| r.hedging_portfolio - r.pool_value)
            .unwrap_or(0.0),
    }
}

#[derive(Debug, Serialize)]
pub struct MarkoutSummary {
    pub horizon: usize,
    /// Size-weighted per-unit LP-side markout, by leg and total.
    /// None = no included fills for that leg (never reported as 0).
    pub per_unit_total: Option<f64>,
    pub per_unit_arb: Option<f64>,
    pub per_unit_buy: Option<f64>,
    pub per_unit_sell: Option<f64>,
    /// Raw sum (sum_k MO_{k,h}); at h = 0 the total equals L_T exactly.
    pub sum_total: f64,
    /// Fill-count coverage: included fills / all fills.
    pub coverage_fills: Option<f64>,
    /// Volume coverage: included |delta| / total |delta|, total and by
    /// leg — guards against a few large tail fills being dropped while
    /// fill-count coverage still looks high.
    pub coverage_volume_total: Option<f64>,
    pub coverage_volume_arb: Option<f64>,
    pub coverage_volume_buy: Option<f64>,
    pub coverage_volume_sell: Option<f64>,
}

/// LP-side markout at a PHYSICAL horizon (round 13: horizons are frozen
/// in hours, {0, 1, 5, 20} h primary, and converted per clock so results
/// are clock-invariant).
pub fn markout_hours(records: &[StepRecord], horizon_hours: f64, dt_hours: f64) -> MarkoutSummary {
    markout(records, (horizon_hours / dt_hours).round() as usize)
}

/// LP-side markout at `horizon` event steps:
/// MO_{k,h} = delta_k (P_{t_k+h} - pbar_k) against the CEX mid;
/// fills past the episode end are dropped and reported via coverage.
/// Prefer [`markout_hours`] in experiments (physical horizons).
pub fn markout(records: &[StepRecord], horizon: usize) -> MarkoutSummary {
    let mut sums = [0.0f64; 3]; // arb, buy, sell (included fills)
    let mut vols = [0.0f64; 3]; // included volume
    let mut all_vols = [0.0f64; 3]; // all-fill volume
    let mut included = 0usize;
    let mut fills_total = 0usize;

    for (i, r) in records.iter().enumerate() {
        let legs = [
            (r.arb_delta, r.pbar_arb),
            (r.buy_delta, r.pbar_buy),
            (r.sell_delta, r.pbar_sell),
        ];
        for (leg, (delta, pbar)) in legs.iter().enumerate() {
            if *delta == 0.0 {
                continue;
            }
            fills_total += 1;
            all_vols[leg] += delta.abs();
            let Some(future) = records.get(i + horizon) else {
                continue;
            };
            included += 1;
            sums[leg] += delta * (future.cex_price - pbar);
            vols[leg] += delta.abs();
        }
    }

    let vol_total: f64 = vols.iter().sum();
    let all_vol_total: f64 = all_vols.iter().sum();
    MarkoutSummary {
        horizon,
        per_unit_total: ratio(sums.iter().sum::<f64>(), vol_total),
        per_unit_arb: ratio(sums[0], vols[0]),
        per_unit_buy: ratio(sums[1], vols[1]),
        per_unit_sell: ratio(sums[2], vols[2]),
        sum_total: sums.iter().sum(),
        coverage_fills: ratio(included as f64, fills_total as f64),
        coverage_volume_total: ratio(vol_total, all_vol_total),
        coverage_volume_arb: ratio(vols[0], all_vols[0]),
        coverage_volume_buy: ratio(vols[1], all_vols[1]),
        coverage_volume_sell: ratio(vols[2], all_vols[2]),
    }
}

// ── M2.6/M3 primitive-event metrics (round 22) ───────────────────────────
//
// With multiple primitive events per clock bin, aggregation granularities
// are renamed (the old step-based names were ambiguous):
//   primitive-fill (PRIMARY): pos/neg parts per primitive event;
//   clock-bin-netted: all ell in a physical clock bin net first;
//   clock-bin x fill-type netted: within a bin, arb net separately and
//     fundamental buy/sell net together (ell_fund per bin).
// Orderings: bin <= type <= fill for both A and B; A - B = L at all three.

use crate::campbell::simulation::{EventKind, EventRecord};

#[derive(Debug, Serialize)]
pub struct EventSummary {
    pub l_total: f64,
    pub a_fill: f64,
    pub b_fill: f64,
    pub a_bin: f64,
    pub b_bin: f64,
    pub a_type: f64,
    pub b_type: f64,
    pub a_arb: f64,
    pub a_fund: f64,
    pub b_fund: f64,
    pub fees_total: f64,
    pub u_lp_rel: f64,
    pub served_fund_volume: f64,
    pub potential_volume: f64,
    pub alloc_amm_share: Option<f64>,
    pub alloc_cex_share: Option<f64>,
    pub alloc_unserved_share: Option<f64>,
    pub incidence_event: Option<f64>,
    pub incidence_buy_event: Option<f64>,
    pub incidence_sell_event: Option<f64>,
    pub cond_fill_size: Option<f64>,
    pub cond_fill_size_buy: Option<f64>,
    pub cond_fill_size_sell: Option<f64>,
    // side-level allocation (risky units)
    pub served_buy: f64,
    pub served_sell: f64,
    pub pot_buy: f64,
    pub pot_sell: f64,
    pub cex_buy: f64,
    pub cex_sell: f64,
    pub unserved_buy: f64,
    pub unserved_sell: f64,
    pub fees_arb: f64,
    pub fees_fund: f64,
    pub a_arb_per_served_unit: Option<f64>,
    pub a_fund_per_served_unit: Option<f64>,
    pub a_total_per_served_unit: Option<f64>,
    pub n_fund_events: u64,
    pub tracking_error: f64,
}

/// Fill-level and netted A/B accounting on the PRIMITIVE event ledger.
/// `records` supplies fees and the tracking-error cross-check.
pub fn summarize_events(events: &[EventRecord], records: &[StepRecord]) -> EventSummary {
    let (mut a_fill, mut b_fill) = (0.0, 0.0);
    let (mut a_arb, mut a_fund, mut b_fund) = (0.0, 0.0, 0.0);
    let (mut served, mut potential, mut cex_total, mut uns_total) = (0.0, 0.0, 0.0, 0.0);
    let (mut n_fund, mut n_fill) = (0u64, 0u64);
    let mut fill_vol = 0.0;
    let (mut served_b, mut served_s, mut pot_b, mut pot_s) = (0.0, 0.0, 0.0, 0.0);
    let (mut cex_b, mut cex_s, mut uns_b, mut uns_s) = (0.0, 0.0, 0.0, 0.0);
    let (mut nb, mut ns, mut nb_fill, mut ns_fill) = (0u64, 0u64, 0u64, 0u64);
    let (mut vol_b, mut vol_s) = (0.0, 0.0);
    // per-bin accumulators: (all, arb, fund)
    use std::collections::HashMap;
    let mut bins: HashMap<usize, (f64, f64, f64)> = HashMap::new();
    for e in events {
        a_fill += e.ell.max(0.0);
        b_fill += (-e.ell).max(0.0);
        let b = bins.entry(e.step).or_insert((0.0, 0.0, 0.0));
        b.0 += e.ell;
        if e.kind == EventKind::Arb {
            // Lemma 2(iii): arb fills are per-fill nonnegative; a silent
            // engine regression must not leak negative gaps into A_arb.
            assert!(
                e.ell >= -1e-9 * e.delta.abs().max(1.0),
                "negative arb ell {} at step {}",
                e.ell,
                e.step
            );
            a_arb += e.ell;
            b.1 += e.ell;
        } else {
            a_fund += e.ell.max(0.0);
            b_fund += (-e.ell).max(0.0);
            b.2 += e.ell;
            n_fund += 1;
            served += e.delta.abs();
            potential += e.pot;
            cex_total += e.cex;
            uns_total += e.unserved;
            if e.delta.abs() > 0.0 {
                n_fill += 1;
                fill_vol += e.delta.abs();
            }
            if e.kind == EventKind::FundBuy {
                nb += 1;
                served_b += e.delta.abs();
                pot_b += e.pot;
                cex_b += e.cex;
                uns_b += e.unserved;
                if e.delta.abs() > 0.0 {
                    nb_fill += 1;
                    vol_b += e.delta.abs();
                }
            } else {
                ns += 1;
                served_s += e.delta.abs();
                pot_s += e.pot;
                cex_s += e.cex;
                uns_s += e.unserved;
                if e.delta.abs() > 0.0 {
                    ns_fill += 1;
                    vol_s += e.delta.abs();
                }
            }
        }
    }
    let (mut a_bin, mut b_bin, mut a_type, mut b_type) = (0.0, 0.0, 0.0, 0.0);
    for (_, (all, arb, fund)) in bins {
        a_bin += all.max(0.0);
        b_bin += (-all).max(0.0);
        a_type += arb.max(0.0) + fund.max(0.0);
        b_type += (-arb).max(0.0) + (-fund).max(0.0);
    }
    let fees_total: f64 = records.iter().map(|r| r.step_fee).sum();
    let fees_arb: f64 = records.iter().map(|r| r.step_fee_arb).sum();
    let fees_fund: f64 = records.iter().map(|r| r.step_fee_fund).sum();
    let l_total = a_fill - b_fill;
    EventSummary {
        l_total,
        a_fill,
        b_fill,
        a_bin,
        b_bin,
        a_type,
        b_type,
        a_arb,
        a_fund,
        b_fund,
        fees_total,
        u_lp_rel: fees_total - l_total,
        served_fund_volume: served,
        potential_volume: potential,
        alloc_amm_share: ratio(served, potential),
        alloc_cex_share: ratio(cex_total, potential),
        alloc_unserved_share: ratio(uns_total, potential),
        incidence_event: ratio(n_fill as f64, n_fund as f64),
        incidence_buy_event: ratio(nb_fill as f64, nb as f64),
        incidence_sell_event: ratio(ns_fill as f64, ns as f64),
        cond_fill_size: ratio(fill_vol, n_fill as f64),
        cond_fill_size_buy: ratio(vol_b, nb_fill as f64),
        cond_fill_size_sell: ratio(vol_s, ns_fill as f64),
        served_buy: served_b,
        served_sell: served_s,
        pot_buy: pot_b,
        pot_sell: pot_s,
        cex_buy: cex_b,
        cex_sell: cex_s,
        unserved_buy: uns_b,
        unserved_sell: uns_s,
        fees_arb,
        fees_fund,
        a_arb_per_served_unit: ratio(a_arb, served),
        a_fund_per_served_unit: ratio(a_fund, served),
        a_total_per_served_unit: ratio(a_fill, served),
        n_fund_events: n_fund,
        tracking_error: records
            .last()
            .map(|r| r.hedging_portfolio - r.pool_value)
            .unwrap_or(0.0),
    }
}

/// LP-side markout on the PRIMITIVE event ledger at a PHYSICAL horizon.
/// External prices exist once per clock bin, so an event in bin s maps to
/// the first external observation at or after its bin, offset by
/// ceil(h/dt) bins; h = 0 uses the event's own bin and recovers ell_k
/// exactly. Coverage reported by event count AND volume.
pub fn markout_events(
    events: &[EventRecord],
    records: &[StepRecord],
    horizon_hours: f64,
    dt_hours: f64,
) -> MarkoutSummary {
    let mut sums = [0.0f64; 3]; // arb, buy, sell
    let mut vols = [0.0f64; 3];
    let mut all_vols = [0.0f64; 3];
    let mut included = 0usize;
    let mut total = 0usize;
    for e in events {
        if e.delta == 0.0 {
            continue;
        }
        let leg = match e.kind {
            EventKind::Arb => 0,
            EventKind::FundBuy => 1,
            EventKind::FundSell => 2,
        };
        total += 1;
        all_vols[leg] += e.delta.abs();
        // Round-23 fill-time mapping: the target is the first external
        // observation at or after event_time + h, using the event's
        // intra-bin fraction u: j = step + ceil(u + h/dt); h = 0 is
        // special-cased to the contemporaneous bin so MO_{k,0} = ell_k
        // exactly.
        let target = if horizon_hours <= 0.0 {
            e.step
        } else {
            e.step + (e.time_frac + horizon_hours / dt_hours).ceil() as usize
        };
        let Some(fut) = records.get(target) else {
            continue;
        };
        included += 1;
        sums[leg] += e.delta * (fut.cex_price - e.pbar);
        vols[leg] += e.delta.abs();
    }
    let vol_total: f64 = vols.iter().sum();
    let all_total: f64 = all_vols.iter().sum();
    MarkoutSummary {
        horizon: (horizon_hours / dt_hours).ceil() as usize,
        per_unit_total: ratio(sums.iter().sum::<f64>(), vol_total),
        per_unit_arb: ratio(sums[0], vols[0]),
        per_unit_buy: ratio(sums[1], vols[1]),
        per_unit_sell: ratio(sums[2], vols[2]),
        sum_total: sums.iter().sum(),
        coverage_fills: ratio(included as f64, total as f64),
        coverage_volume_total: ratio(vol_total, all_total),
        coverage_volume_arb: ratio(vols[0], all_vols[0]),
        coverage_volume_buy: ratio(vols[1], all_vols[1]),
        coverage_volume_sell: ratio(vols[2], all_vols[2]),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::campbell::fee_policy::{FixedFeePolicy, OracleGapFeePolicy};
    use crate::campbell::gbm::generate_gbm;
    use crate::campbell::simulation::{ArrivalModel, FlowRegime, SimConfig, run_simulation};

    fn config(amm_fee: f64, policy_lag: usize) -> SimConfig {
        SimConfig {
            name: "summary".into(),
            description: "m2 summary tests".into(),
            amm_fee,
            cex_fee: 0.003,
            buy_demand: 5.0,
            sell_demand: 5.0,
            reserve_x: 2.0e7,
            reserve_y: 1.0e4,
            sigma: 0.4,
            mu: 0.0,
            n_steps: 800,
            seed: 42,
            flow_regime: FlowRegime::Normal,
            toxic_burst_prob: 0.0,
            toxic_burst_arb_scale: 1.0,
            toxic_burst_fund_scale: 1.0,
            regime_switch_period: 0,
            e1_lambda: 0.3,
            e1_fee_ref: 0.0006,
            e5_arb_prob: 1.0,
            policy_lag,
            dt_hours: 1.0,
            pooled_fund_arrival_rate_per_hour: None,
            buy_arrival_share: 0.5,
            arb_arrival_rate_per_hour: None,
            lookback_hours: 20.0,
            arrival_model: ArrivalModel::Bernoulli,
            log_inactive_arb: false,
        }
    }

    fn run(cfg: &SimConfig) -> Vec<StepRecord> {
        let prices = generate_gbm(
            cfg.n_steps,
            2000.0,
            0.0,
            cfg.sigma,
            1.0 / (365.0 * 24.0),
            cfg.seed,
        );
        let mut policy = FixedFeePolicy::new(cfg.amm_fee);
        run_simulation(cfg, &prices, &mut policy)
    }

    /// Round-22 gate: primitive-event granularities satisfy
    /// bin <= type <= fill for A and B, A - B = L at every granularity,
    /// and markout_events at h = 0 recovers the event-ledger L exactly.
    #[test]
    fn event_summary_granularities_and_markout() {
        use crate::campbell::simulation::{ArrivalModel, run_simulation_with_events};
        let mut cfg = config(0.0005, 0);
        cfg.arrival_model = ArrivalModel::Poisson;
        cfg.dt_hours = 1.0 / 12.0;
        cfg.pooled_fund_arrival_rate_per_hour = Some(60.0);
        let prices = generate_gbm(
            cfg.n_steps,
            2000.0,
            0.0,
            cfg.sigma,
            cfg.dt_years(),
            cfg.seed,
        );
        let mut policy = FixedFeePolicy::new(cfg.amm_fee);
        let (records, events) = run_simulation_with_events(&cfg, &prices, &mut policy);
        let s = summarize_events(&events, &records);
        for (a, b) in [
            (s.a_fill, s.b_fill),
            (s.a_bin, s.b_bin),
            (s.a_type, s.b_type),
        ] {
            assert!(((a - b) - s.l_total).abs() < 1e-9 * a.max(1.0));
        }
        assert!(s.a_bin <= s.a_type + 1e-12 && s.a_type <= s.a_fill + 1e-12);
        assert!(s.b_bin <= s.b_type + 1e-12 && s.b_type <= s.b_fill + 1e-12);
        assert!((s.l_total - s.tracking_error).abs() < 1e-6 * s.tracking_error.abs().max(1.0));
        let m0 = markout_events(&events, &records, 0.0, cfg.dt_hours);
        assert!((m0.sum_total - s.l_total).abs() < 1e-9 * s.l_total.abs().max(1.0));
        assert_eq!(m0.coverage_fills, Some(1.0));
        let m20 = markout_events(&events, &records, 20.0, cfg.dt_hours);
        assert!(m20.coverage_fills.unwrap() < 1.0);
        assert!(m20.coverage_volume_total.unwrap() < 1.0);
    }

    /// Summary identities: L = A - B = tracking error at ALL THREE
    /// granularities; netting ordering event <= type <= fill for A and B;
    /// arb leg nonnegative; allocation shares sum to 1; U_LP = fees - L.
    #[test]
    fn summary_identities_hold() {
        let records = run(&config(0.0005, 0));
        let s = summarize(&records);
        assert!((s.l_total - s.tracking_error).abs() < 1e-6 * s.tracking_error.abs().max(1.0));
        for (a, b) in [
            (s.a_fill, s.b_fill),
            (s.a_event, s.b_event),
            (s.a_type, s.b_type),
        ] {
            assert!(
                ((a - b) - s.l_total).abs() < 1e-9 * a.max(1.0),
                "A - B = L must hold at every granularity: {a} - {b} vs {}",
                s.l_total
            );
        }
        assert!(s.a_event <= s.a_type + 1e-12 && s.a_type <= s.a_fill + 1e-12);
        assert!(s.b_event <= s.b_type + 1e-12 && s.b_type <= s.b_fill + 1e-12);
        assert!(s.a_arb >= 0.0);
        let shares = s.alloc_amm_share.unwrap()
            + s.alloc_cex_share.unwrap()
            + s.alloc_unserved_share.unwrap();
        assert!(
            (shares - 1.0).abs() < 1e-9,
            "allocation shares sum {shares}"
        );
        assert!((s.u_lp_rel - (s.fees_total - s.l_total)).abs() < 1e-9 * s.fees_total.max(1.0));
        assert!(s.b_fill > 0.0, "f < c config must show favorable fills");
        assert!(s.incidence_buy_event.is_some() && s.cond_fill_size_pooled.is_some());
    }

    /// Zero-denominator discipline: under shutdown every ratio must be
    /// None, never 0 — a shutdown policy must not appear to have zero
    /// adverse loss per served unit.
    #[test]
    fn shutdown_ratios_are_none_not_zero() {
        let records = run(&config(0.5, 0)); // 50% fee shuts everything
        let s = summarize(&records);
        assert_eq!(s.served_fund_volume, 0.0);
        assert!(s.a_arb_per_served_unit.is_none());
        assert!(s.a_fund_per_served_unit.is_none());
        assert!(s.a_total_per_served_unit.is_none());
        assert!(s.cond_fill_size_pooled.is_none());
        assert_eq!(s.incidence_buy_event, Some(0.0)); // events exist, no fills
        let m = markout(&records, 0);
        assert!(m.per_unit_total.is_none() && m.coverage_fills.is_none());
    }

    /// Frozen markout definition: at h = 0 the raw sum equals L_T exactly
    /// and both coverages are 1; at long horizons both drop below 1.
    #[test]
    fn markout_h0_recovers_tracking_error() {
        let records = run(&config(0.0005, 0));
        let s = summarize(&records);
        let m0 = markout(&records, 0);
        assert!((m0.sum_total - s.l_total).abs() < 1e-9 * s.l_total.abs().max(1.0));
        assert_eq!(m0.coverage_fills, Some(1.0));
        assert_eq!(m0.coverage_volume_total, Some(1.0));
        let m_long = markout(&records, 790);
        assert!(m_long.coverage_fills.unwrap() < 1.0);
        assert!(m_long.coverage_volume_total.unwrap() < 1.0);
    }

    /// Defensive-family lag parity: a trigger-heavy gap policy run under
    /// zero-lag and one-step-lag must (a) actually differ in decisions and
    /// (b) satisfy every accounting identity in both modes.
    #[test]
    fn defensive_family_lag_parity() {
        let prices = generate_gbm(800, 2000.0, 0.0, 0.4, 1.0 / (365.0 * 24.0), 42);
        let mut out = Vec::new();
        for lag in [0usize, 1] {
            let cfg = config(0.0005, lag);
            let mut policy = OracleGapFeePolicy {
                base_fee: 0.0005,
                gap_multiplier: 5.0, // defensive: escalates hard on gap
                min_fee: 0.0005,
                max_fee: 0.05,
            };
            let records = run_simulation(&cfg, &prices, &mut policy);
            let s = summarize(&records);
            assert!(
                (s.l_total - s.tracking_error).abs() < 1e-6 * s.tracking_error.abs().max(1.0),
                "identity broken under lag={lag}"
            );
            let shares = s.alloc_amm_share.unwrap()
                + s.alloc_cex_share.unwrap()
                + s.alloc_unserved_share.unwrap();
            assert!((shares - 1.0).abs() < 1e-9, "ledger broken under lag={lag}");
            out.push((records, s));
        }
        let fees_differ = out[0]
            .0
            .iter()
            .zip(out[1].0.iter())
            .any(|(a, b)| (a.fee_used - b.fee_used).abs() > 1e-15);
        assert!(fees_differ, "defensive policy must react to the lag change");
    }
}

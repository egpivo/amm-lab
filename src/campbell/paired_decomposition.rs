//! Paired-ledger policy contrast decomposition on shared primitive keys.

use crate::campbell::simulation::{EventKind, EventRecord};
use crate::campbell::summary::{EventSummary, summarize_events};
use std::collections::{HashMap, HashSet};

pub const DEFAULT_N_STEPS: usize = 604_800;

#[derive(Hash, Eq, PartialEq, Clone, Copy, Debug)]
pub struct OppKey {
    pub step: usize,
    pub time_frac_bits: u64,
    pub kind_tag: u8,
}

pub fn kind_tag(kind: EventKind) -> u8 {
    match kind {
        EventKind::Arb => 0,
        EventKind::FundBuy => 1,
        EventKind::FundSell => 2,
    }
}

pub fn tag_kind(tag: u8) -> EventKind {
    match tag {
        0 => EventKind::Arb,
        1 => EventKind::FundBuy,
        2 => EventKind::FundSell,
        _ => unreachable!("invalid kind tag"),
    }
}

impl OppKey {
    pub fn from_event(e: &EventRecord) -> Self {
        Self {
            step: e.step,
            time_frac_bits: e.time_frac.to_bits(),
            kind_tag: kind_tag(e.kind),
        }
    }
}

#[derive(Clone, Copy, Default, Debug)]
pub struct OppFields {
    pub e: u8,
    pub q: f64,
    pub a: f64,
    pub c: f64,
}

impl OppFields {
    pub fn from_event(e: &EventRecord) -> Self {
        let q = e.delta.abs();
        let inc = q > 0.0;
        let a = e.ell.max(0.0);
        let c = if inc { a / q } else { 0.0 };
        Self {
            e: u8::from(inc),
            q,
            a,
            c,
        }
    }

    pub fn zero() -> Self {
        Self::default()
    }
}

#[derive(Clone, Copy, Default, Debug)]
pub struct LegDecomp {
    pub delta_a: f64,
    pub delta_qty_c: f64,
    pub delta_sev_c: f64,
    pub delta_entry: f64,
    pub delta_exit: f64,
    pub n_opportunities: u64,
    pub n_common: u64,
    pub n_entry: u64,
    pub n_exit: u64,
}

impl LegDecomp {
    pub fn delta_common(self) -> f64 {
        self.delta_qty_c + self.delta_sev_c
    }

    pub fn delta_selection(self) -> f64 {
        self.delta_entry + self.delta_exit
    }

    pub fn reconstruct_err(self) -> f64 {
        (self.delta_a - self.delta_common() - self.delta_selection()).abs()
    }
}

#[derive(Clone, Copy, Default, Debug)]
pub struct SeedDecomp {
    pub total: LegDecomp,
    pub fund: LegDecomp,
    pub arb: LegDecomp,
}

#[derive(Clone, Copy, Default, Debug)]
pub struct AggregateDeltas {
    pub delta_a: f64,
    pub delta_s: f64,
    pub delta_fees: f64,
    pub delta_u: f64,
}

struct IndexedLedger {
    fund: HashMap<OppKey, OppFields>,
    arb: Vec<OppFields>,
}

fn index_events(events: &[EventRecord], n_steps: usize) -> IndexedLedger {
    let mut fund = HashMap::new();
    let mut arb = vec![OppFields::zero(); n_steps + 1];
    for e in events {
        if e.kind == EventKind::Arb {
            if e.step <= n_steps {
                arb[e.step] = OppFields::from_event(e);
            }
        } else {
            fund.insert(OppKey::from_event(e), OppFields::from_event(e));
        }
    }
    IndexedLedger { fund, arb }
}

fn accumulate_pair(f1: OppFields, f0: OppFields, out: &mut LegDecomp) {
    out.n_opportunities += 1;
    out.delta_a += f1.a - f0.a;
    match (f1.e, f0.e) {
        (1, 1) => {
            out.n_common += 1;
            out.delta_qty_c += (f1.q - f0.q) * f0.c;
            out.delta_sev_c += f1.q * (f1.c - f0.c);
        }
        (1, 0) => {
            out.n_entry += 1;
            out.delta_entry += f1.a;
        }
        (0, 1) => {
            out.n_exit += 1;
            out.delta_exit -= f0.a;
        }
        (0, 0) => {}
        _ => unreachable!("incidence is binary"),
    }
}

fn decompose_fund(p1: &HashMap<OppKey, OppFields>, p0: &HashMap<OppKey, OppFields>) -> LegDecomp {
    let keys: HashSet<OppKey> = p1
        .keys()
        .chain(p0.keys())
        .copied()
        .filter(|k| {
            matches!(
                tag_kind(k.kind_tag),
                EventKind::FundBuy | EventKind::FundSell
            )
        })
        .collect();
    let mut out = LegDecomp::default();
    for key in keys {
        let f1 = p1.get(&key).copied().unwrap_or_else(OppFields::zero);
        let f0 = p0.get(&key).copied().unwrap_or_else(OppFields::zero);
        accumulate_pair(f1, f0, &mut out);
    }
    out
}

fn decompose_arb(a1: &[OppFields], a0: &[OppFields]) -> LegDecomp {
    let n = a1.len().min(a0.len());
    let mut out = LegDecomp::default();
    for step in 0..n {
        accumulate_pair(a1[step], a0[step], &mut out);
    }
    out
}

pub fn decompose_pair(events1: &[EventRecord], events0: &[EventRecord]) -> SeedDecomp {
    decompose_pair_with_steps(events1, events0, DEFAULT_N_STEPS)
}

pub fn decompose_pair_with_steps(
    events1: &[EventRecord],
    events0: &[EventRecord],
    n_steps: usize,
) -> SeedDecomp {
    let p1 = index_events(events1, n_steps);
    let p0 = index_events(events0, n_steps);
    let fund = decompose_fund(&p1.fund, &p0.fund);
    let arb = decompose_arb(&p1.arb, &p0.arb);
    let mut total = LegDecomp::default();
    for leg in [fund, arb] {
        total.delta_a += leg.delta_a;
        total.delta_qty_c += leg.delta_qty_c;
        total.delta_sev_c += leg.delta_sev_c;
        total.delta_entry += leg.delta_entry;
        total.delta_exit += leg.delta_exit;
        total.n_opportunities += leg.n_opportunities;
        total.n_common += leg.n_common;
        total.n_entry += leg.n_entry;
        total.n_exit += leg.n_exit;
    }
    SeedDecomp { total, fund, arb }
}

pub fn aggregate_deltas(es1: &EventSummary, es0: &EventSummary) -> AggregateDeltas {
    AggregateDeltas {
        delta_a: es1.a_fill - es0.a_fill,
        delta_s: es1.served_fund_volume - es0.served_fund_volume,
        delta_fees: es1.fees_total - es0.fees_total,
        delta_u: es1.u_lp_rel - es0.u_lp_rel,
    }
}

pub fn summarize_run(
    events: &[EventRecord],
    records: &[crate::campbell::simulation::StepRecord],
) -> EventSummary {
    summarize_events(events, records)
}

pub fn assert_exact_reconstruction(decomp: &SeedDecomp, scale: f64) {
    let tol = 1e-8_f64.max(1e-8 * scale.abs());
    for leg in [decomp.total, decomp.fund, decomp.arb] {
        assert!(
            leg.reconstruct_err() <= tol,
            "reconstruction failed: err={} scale={}",
            leg.reconstruct_err(),
            scale
        );
    }
}

//! ISO-week primitive aggregation with a caller-supplied frozen calendar.

use crate::pstt::error::{PsttError, Result};
use crate::pstt::schema::{ReferenceKind, WeeklyPrimitives, WeeklyRow};
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct WeeklyKey {
    pub pool: String,
    pub reference: String,
    pub week: String,
}

/// Initialize every `(pool, reference, week)` slot to zeros, preserving zero-service weeks.
pub fn empty_calendar(
    pools: &[String],
    references: &[ReferenceKind],
    weeks: &[String],
) -> BTreeMap<WeeklyKey, WeeklyPrimitives> {
    let mut out = BTreeMap::new();
    for pool in pools {
        for reference in references {
            for week in weeks {
                out.insert(
                    WeeklyKey {
                        pool: pool.clone(),
                        reference: reference.as_str().to_string(),
                        week: week.clone(),
                    },
                    WeeklyPrimitives::default(),
                );
            }
        }
    }
    out
}

pub fn bump_fill_count(
    table: &mut BTreeMap<WeeklyKey, WeeklyPrimitives>,
    pool: &str,
    week: &str,
    references: &[ReferenceKind],
) -> Result<()> {
    for reference in references {
        let key = WeeklyKey {
            pool: pool.to_string(),
            reference: reference.as_str().to_string(),
            week: week.to_string(),
        };
        let row = table
            .get_mut(&key)
            .ok_or_else(|| PsttError::invariant(format!("fill outside frozen calendar: {week}")))?;
        row.fill_count += 1;
    }
    Ok(())
}

pub fn accumulate(
    table: &mut BTreeMap<WeeklyKey, WeeklyPrimitives>,
    pool: &str,
    reference: ReferenceKind,
    week: &str,
    ell: f64,
    q: f64,
) -> Result<()> {
    let key = WeeklyKey {
        pool: pool.to_string(),
        reference: reference.as_str().to_string(),
        week: week.to_string(),
    };
    let row = table
        .get_mut(&key)
        .ok_or_else(|| PsttError::invariant(format!("fill outside frozen calendar: {week}")))?;
    row.accumulate_mark(ell, q);
    Ok(())
}

pub fn to_rows(
    table: &BTreeMap<WeeklyKey, WeeklyPrimitives>,
    meta: &BTreeMap<String, (String, u32)>,
) -> Result<Vec<WeeklyRow>> {
    let mut rows = Vec::with_capacity(table.len());
    for (key, prim) in table {
        let (pair, fee) = meta
            .get(&key.pool)
            .ok_or_else(|| PsttError::schema(format!("missing pool metadata for {}", key.pool)))?;
        if prim.identity_residual() > 1e-8 {
            return Err(PsttError::invariant(format!(
                "weekly L!=A-B for {} {} {}",
                key.pool, key.reference, key.week
            )));
        }
        rows.push(WeeklyRow {
            pool: key.pool.clone(),
            pair: pair.clone(),
            fee: *fee,
            reference: key.reference.clone(),
            week: key.week.clone(),
            l: prim.l,
            a: prim.a,
            b: prim.b,
            s: prim.s,
            observed_mass: prim.observed_mass,
            service_q2: prim.service_q2,
            fill_count: prim.fill_count,
            matched_count: prim.matched_count,
        });
    }
    Ok(rows)
}

/// Effective sample size from total service and sum of squares.
pub fn n_eff(service: f64, service_q2: f64) -> Option<f64> {
    if service_q2 > 0.0 && service.is_finite() && service_q2.is_finite() {
        Some((service * service) / service_q2)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pstt::block_time::panel_calendar_weeks;

    #[test]
    fn materializes_zero_service_weeks() {
        let weeks = panel_calendar_weeks();
        let pools = vec!["0xabc".into()];
        let refs = [ReferenceKind::LastTrade, ReferenceKind::Vwap1s];
        let table = empty_calendar(&pools, &refs, &weeks);
        assert_eq!(table.len(), 104 * 2);
        let first = table
            .get(&WeeklyKey {
                pool: "0xabc".into(),
                reference: "last_trade".into(),
                week: "2024-01".into(),
            })
            .unwrap();
        assert_eq!(first.fill_count, 0);
        assert_eq!(first.observed_mass, 0);
    }

    #[test]
    fn accumulation_preserves_identity() {
        let weeks = vec!["2024-01".into()];
        let pools = vec!["p".into()];
        let refs = [ReferenceKind::LastTrade];
        let mut table = empty_calendar(&pools, &refs, &weeks);
        bump_fill_count(&mut table, "p", "2024-01", &refs).unwrap();
        accumulate(
            &mut table,
            "p",
            ReferenceKind::LastTrade,
            "2024-01",
            3.0,
            1.5,
        )
        .unwrap();
        accumulate(
            &mut table,
            "p",
            ReferenceKind::LastTrade,
            "2024-01",
            -1.0,
            0.5,
        )
        .unwrap();
        let row = table.values().next().unwrap();
        assert!((row.l - 2.0).abs() < 1e-12);
        assert!((row.a - 3.0).abs() < 1e-12);
        assert!((row.b - 1.0).abs() < 1e-12);
        assert!((row.l - (row.a - row.b)).abs() < 1e-12);
        assert_eq!(row.fill_count, 1);
        assert_eq!(row.matched_count, 2);
    }
}

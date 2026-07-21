//! Block headers, timestamp-unit normalization, and ISO-week labels.

use crate::pstt::error::{PsttError, Result};
use chrono::{DateTime, Datelike, NaiveDate, TimeZone, Utc};

/// Normalize a Binance integer timestamp using the frozen digit-length rule.
///
/// - 13-digit values are milliseconds (`/ 1e3`)
/// - 16-digit values are microseconds (`/ 1e6`)
/// - all other digit lengths are rejected rather than guessed
pub fn normalize_aggtrade_timestamp(raw: i64) -> Result<f64> {
    if raw < 0 {
        return Err(PsttError::parse(format!(
            "negative aggTrade timestamp: {raw}"
        )));
    }
    let digits = digit_count(raw);
    match digits {
        13 => Ok((raw as f64) / 1e3),
        16 => Ok((raw as f64) / 1e6),
        n => Err(PsttError::parse(format!(
            "ambiguous aggTrade timestamp digit length {n}: {raw}"
        ))),
    }
}

fn digit_count(raw: i64) -> u32 {
    if raw == 0 {
        1
    } else {
        ((raw as f64).log10().floor() as u32) + 1
    }
}

pub fn utc_day(timestamp_unix: f64) -> Result<NaiveDate> {
    let secs = timestamp_unix.floor() as i64;
    let nsecs = ((timestamp_unix - secs as f64) * 1e9).round() as u32;
    DateTime::from_timestamp(secs, nsecs)
        .map(|dt| dt.date_naive())
        .ok_or_else(|| PsttError::parse(format!("invalid unix timestamp: {timestamp_unix}")))
}

pub fn iso_week_label(timestamp_unix: f64) -> Result<String> {
    let dt = DateTime::from_timestamp(timestamp_unix.floor() as i64, 0).ok_or_else(|| {
        PsttError::parse(format!(
            "invalid unix timestamp for ISO week: {timestamp_unix}"
        ))
    })?;
    let iso = dt.iso_week();
    Ok(format!("{:04}-{:02}", iso.year(), iso.week()))
}

pub fn iso_week_label_date(date: NaiveDate) -> String {
    let iso = date.iso_week();
    format!("{:04}-{:02}", iso.year(), iso.week())
}

/// Build the closed ISO-week calendar covering `[start_date, end_date]` inclusive,
/// using Monday-based ISO week labels `%G-%V`.
pub fn iso_weeks_inclusive(start: NaiveDate, end: NaiveDate) -> Result<Vec<String>> {
    if end < start {
        return Err(PsttError::invariant("ISO week calendar end precedes start"));
    }
    let mut out = Vec::new();
    let mut cur = start;
    while cur <= end {
        let label = iso_week_label_date(cur);
        if out.last() != Some(&label) {
            out.push(label);
        }
        cur = cur
            .succ_opt()
            .ok_or_else(|| PsttError::invariant("date overflow while building ISO weeks"))?;
    }
    Ok(out)
}

/// Paper panel window matching `panel_audit_v2.iso_weeks_window`:
/// step Monday-aligned dates from 2024-01-01 through 2025-12-27 by 7 days,
/// then keep labels with string order in `2024-01..=2025-52` (104 slots).
pub fn panel_calendar_weeks() -> Vec<String> {
    let mut out = Vec::new();
    let mut d = NaiveDate::from_ymd_opt(2024, 1, 1).expect("valid");
    let end = NaiveDate::from_ymd_opt(2025, 12, 27).expect("valid");
    while d <= end {
        let w = iso_week_label_date(d);
        if out.last() != Some(&w) {
            out.push(w);
        }
        d += chrono::Duration::days(7);
    }
    out.into_iter()
        .filter(|w| w.as_str() >= "2024-01" && w.as_str() <= "2025-52")
        .collect()
}

pub fn timestamps_nondecreasing(sorted_by_block: &[(u64, i64)]) -> bool {
    sorted_by_block.windows(2).all(|w| w[0].1 <= w[1].1)
}

pub fn sparse_parent_chain_ok(headers: &[(u64, &str, &str)]) -> bool {
    // Only check numerically consecutive blocks.
    for w in headers.windows(2) {
        let (n0, h0, _) = w[0];
        let (n1, _, p1) = w[1];
        if n1 == n0 + 1 && p1 != h0 {
            return false;
        }
    }
    true
}

pub fn datetime_utc(timestamp_unix: i64) -> Result<DateTime<Utc>> {
    Utc.timestamp_opt(timestamp_unix, 0)
        .single()
        .ok_or_else(|| PsttError::parse(format!("invalid unix timestamp: {timestamp_unix}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ms_and_us_timestamps() {
        let ms = normalize_aggtrade_timestamp(1_720_051_200_000).unwrap();
        let us = normalize_aggtrade_timestamp(1_735_689_600_000_000).unwrap();
        assert!((ms - 1_720_051_200.0).abs() < 1e-9);
        assert!((us - 1_735_689_600.0).abs() < 1e-9);
    }

    #[test]
    fn rejects_ambiguous_digit_lengths() {
        assert!(normalize_aggtrade_timestamp(1_720_051_200).is_err()); // 10-digit seconds
        assert!(normalize_aggtrade_timestamp(172_005_120_000_000).is_err()); // 15 digits
    }

    #[test]
    fn iso_week_year_boundary() {
        // 2024-12-30 is ISO week 2025-01.
        let ts = NaiveDate::from_ymd_opt(2024, 12, 30)
            .unwrap()
            .and_hms_opt(12, 0, 0)
            .unwrap()
            .and_utc()
            .timestamp() as f64;
        assert_eq!(iso_week_label(ts).unwrap(), "2025-01");
    }

    #[test]
    fn panel_calendar_has_104_weeks() {
        assert_eq!(panel_calendar_weeks().len(), 104);
        let weeks = panel_calendar_weeks();
        assert_eq!(weeks.first().unwrap(), "2024-01");
        assert_eq!(weeks.last().unwrap(), "2025-52");
    }
}

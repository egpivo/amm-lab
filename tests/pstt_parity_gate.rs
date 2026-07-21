//! Weekly primitive + parity CLI gate on synthetic fixtures.

use amm_lab::pstt::parity::FloatTol;
use amm_lab::pstt::parity::float_close;
use amm_lab::pstt::schema::WeeklyRow;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

fn fixture_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/pstt/e1_weekly_pipeline_v1")
}

#[test]
fn weekly_marks_and_identity() {
    let root = fixture_root();
    let out = std::env::temp_dir().join(format!(
        "pstt_parity_gate_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let _ = fs::remove_dir_all(&out);

    let status = Command::new(env!("CARGO_BIN_EXE_pstt_build_weekly_primitives"))
        .args([
            "--fills-csv",
            root.join("inputs/fills.csv").to_str().unwrap(),
            "--trades-dir",
            root.join("inputs").to_str().unwrap(),
            "--output-dir",
            out.to_str().unwrap(),
            "--compact-calendar",
        ])
        .status()
        .expect("spawn weekly");
    assert!(status.success(), "weekly CLI failed");

    let rows: Vec<WeeklyRow> =
        serde_json::from_str(&fs::read_to_string(out.join("pstt_weekly.json")).unwrap()).unwrap();
    assert_eq!(rows.len(), 2); // one pool × two references × one week

    let last = rows
        .iter()
        .find(|r| r.reference == "last_trade")
        .expect("last_trade row");
    // Fill1: dir=+1,q=1,p_ref=2480,p_exec=2490 -> ell = -10
    // Fill2: dir=-1,q=2,p_ref=2480,p_exec=2510 -> ell = (-1)*2*(2480-2510)=60
    // Fill3 at 1704110500: last trade before T is 2508 at 1704110499 -> ell=1*(2508-2500)=8
    // L = -10 + 60 + 8 = 58; A = 0+60+8=68; B=10
    assert!(float_close(last.l, 58.0, FloatTol::WEEKLY));
    assert!(float_close(last.a, 68.0, FloatTol::WEEKLY));
    assert!(float_close(last.b, 10.0, FloatTol::WEEKLY));
    assert!(float_close(last.l, last.a - last.b, FloatTol::WEEKLY));
    assert_eq!(last.fill_count, 3);
    assert_eq!(last.matched_count, 3);

    let vwap = rows
        .iter()
        .find(|r| r.reference == "vwap1s")
        .expect("vwap row");
    // Fills at T=1704110400 have empty [T-1,T) trade window; only fill3 matches.
    assert_eq!(vwap.fill_count, 3);
    assert_eq!(vwap.matched_count, 1);
    assert!(float_close(vwap.l, 8.0, FloatTol::WEEKLY));
    assert!(float_close(vwap.a, 8.0, FloatTol::WEEKLY));
    assert!(float_close(vwap.b, 0.0, FloatTol::WEEKLY));
    assert!(float_close(vwap.l, vwap.a - vwap.b, FloatTol::WEEKLY));

    let golden = root.join("expected/pstt_weekly.json");
    let status = Command::new(env!("CARGO_BIN_EXE_pstt_verify_parity"))
        .args([
            "--expected",
            golden.to_str().unwrap(),
            "--actual",
            out.join("pstt_weekly.json").to_str().unwrap(),
        ])
        .status()
        .expect("spawn parity golden");
    assert!(status.success(), "diverged from committed golden");

    let _ = fs::remove_dir_all(&out);
}

#[test]
fn missing_golden_is_hard_failure_without_smoke() {
    let status = Command::new(env!("CARGO_BIN_EXE_pstt_verify_parity"))
        .args([
            "--expected",
            "/tmp/pstt_does_not_exist_expected.json",
            "--actual",
            "/tmp/pstt_does_not_exist_actual.json",
        ])
        .status()
        .expect("spawn");
    assert!(!status.success());
}

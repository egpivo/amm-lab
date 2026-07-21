//! Integration gate for `pstt_audit_staleness` against synthetic fixtures.

use serde_json::Value;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

fn fixture_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/pstt/e1_staleness_v1")
}

#[test]
fn staleness_gate_survives_with_full_coverage() {
    let root = fixture_root();
    let out = std::env::temp_dir().join(format!(
        "pstt_staleness_gate_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let _ = fs::remove_dir_all(&out);

    let status = Command::new(env!("CARGO_BIN_EXE_pstt_audit_staleness"))
        .args([
            "--inventory-json",
            root.join("inputs/inventory.json").to_str().unwrap(),
            "--block-timestamps-csv",
            root.join("inputs/block_timestamps.csv").to_str().unwrap(),
            "--pool-blocks-csv",
            root.join("inputs/pool_blocks.csv").to_str().unwrap(),
            "--trades-dir",
            root.join("inputs").to_str().unwrap(),
            "--output-dir",
            out.to_str().unwrap(),
            "--start-day",
            "2024-01-01",
            "--end-day",
            "2024-01-01",
        ])
        .status()
        .expect("spawn staleness");
    assert!(status.success(), "staleness CLI failed");

    let audit: Value =
        serde_json::from_str(&fs::read_to_string(out.join("pstt_staleness_audit.json")).unwrap())
            .unwrap();
    assert_eq!(audit["surviving"][0], "USDC/WETH:500-3000");
    assert!(audit["reference_limited"].as_array().unwrap().is_empty());

    let p1 = &audit["pool_stats"]["0x0000000000000000000000000000000000000001"];
    assert_eq!(p1["blocks"], 2);
    assert_eq!(p1["joined"], 2);
    assert!((p1["coverage"].as_f64().unwrap() - 1.0).abs() < 1e-12);
    // staleness values: 5s (vs trade @0395) and 15s -> nearest-rank q50 = 5
    assert!((p1["q50_seconds"].as_f64().unwrap() - 5.0).abs() < 1e-12);

    let _ = fs::remove_dir_all(&out);
}

//! Integration gate for `pstt_extract_target_blocks` against synthetic fixtures.

use std::fs;
use std::path::PathBuf;
use std::process::Command;

fn fixture_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/pstt/e1_blocks_v1")
}

#[test]
fn extracts_eligible_blocks_exactly() {
    let root = fixture_root();
    let out = std::env::temp_dir().join(format!(
        "pstt_prepare_gate_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let _ = fs::remove_dir_all(&out);

    let status = Command::new(env!("CARGO_BIN_EXE_pstt_extract_target_blocks"))
        .args([
            "--events",
            root.join("inputs/events.csv").to_str().unwrap(),
            "--pools-json",
            root.join("inputs/pools.json").to_str().unwrap(),
            "--start-unix",
            "1704067200",
            "--end-unix",
            "1704153600",
            "--output-dir",
            out.to_str().unwrap(),
        ])
        .status()
        .expect("spawn extract");
    assert!(status.success(), "extract CLI failed");

    let got_targets = fs::read_to_string(out.join("pstt_target_blocks.txt")).unwrap();
    let exp_targets = fs::read_to_string(root.join("expected/pstt_target_blocks.txt")).unwrap();
    assert_eq!(got_targets, exp_targets);

    let got_pools = fs::read_to_string(out.join("pstt_pool_blocks.csv")).unwrap();
    let exp_pools = fs::read_to_string(root.join("expected/pstt_pool_blocks.csv")).unwrap();
    assert_eq!(got_pools, exp_pools);

    let _ = fs::remove_dir_all(&out);
}

#[test]
fn refuses_nonempty_output_directory() {
    let root = fixture_root();
    let out = std::env::temp_dir().join(format!("pstt_prepare_nonempty_{}", std::process::id()));
    let _ = fs::remove_dir_all(&out);
    fs::create_dir_all(&out).unwrap();
    fs::write(out.join("existing.txt"), b"x").unwrap();

    let status = Command::new(env!("CARGO_BIN_EXE_pstt_extract_target_blocks"))
        .args([
            "--events",
            root.join("inputs/events.csv").to_str().unwrap(),
            "--pools-json",
            root.join("inputs/pools.json").to_str().unwrap(),
            "--start-unix",
            "1704067200",
            "--end-unix",
            "1704153600",
            "--output-dir",
            out.to_str().unwrap(),
        ])
        .status()
        .expect("spawn extract");
    assert!(!status.success(), "should refuse nonempty output");
    let _ = fs::remove_dir_all(&out);
}

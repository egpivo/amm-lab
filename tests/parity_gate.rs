//! Integration tests for the `reconstruct_parity` binary's fail-safe behaviour.
//!
//! The gate once had a high-risk bug where a missing/mis-pathed golden could exit 0 and be
//! read as "parity passed". These tests run the actual compiled binary and pin:
//!   - golden missing, no flag        -> nonzero exit, "PARITY NOT CHECKED"
//!   - golden missing, --smoke        -> zero exit, reconstruction only
//!   - default golden path is DATA_DIR/panel_weekly_frozen.csv, and a matching golden PASSES

use flate2::{Compression, write::GzEncoder};
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Command;

const HEADER: &str = "pool,unit_role,tx_hash,block,tx_index,log_index,ts,type,owner,tickLower,tickUpper,liquidity_delta,swap_liquidity,amount0,amount1,sqrtP,tick,token0,token1,removed";

/// A minimal build_outcomes-style data dir: one pool, one mint + one swap in one week.
fn setup_data_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("paritygate_{}_{name}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(dir.join("events")).unwrap();

    let rows = [
        "0xaaa,matched_treated,0xt1,100,0,0,1700000000,mint,0xlp,-300,300,1000,,0,0,0,,0xt0,0xt1,0",
        "0xaaa,matched_treated,0xt2,101,0,1,1700003600,swap,,,,,2000,10,20,0,50,0xt0,0xt1,0",
    ];
    let f = fs::File::create(dir.join("events").join("events.csv.gz")).unwrap();
    let mut enc = GzEncoder::new(f, Compression::default());
    writeln!(enc, "{HEADER}").unwrap();
    for r in rows {
        writeln!(enc, "{r}").unwrap();
    }
    enc.finish().unwrap();

    fs::write(
        dir.join("panel_units.json"),
        r#"{"treated_matched":["0xaaa"]}"#,
    )
    .unwrap();
    fs::write(dir.join("feerev_panelvars.csv"), "pool,tier\n0xaaa,3000\n").unwrap();
    fs::write(dir.join("ckpt_tickbook.json"), "{}").unwrap();
    // token_decimals.json intentionally absent -> reader warns and defaults to 18 (smoke ok).
    dir
}

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_reconstruct_parity")
}

#[test]
fn golden_missing_without_flag_fails() {
    let dir = setup_data_dir("missing");
    let out = Command::new(bin()).arg(&dir).output().unwrap();
    assert!(
        !out.status.success(),
        "missing golden must be a nonzero exit; got {:?}",
        out.status.code()
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("PARITY NOT CHECKED"),
        "expected 'PARITY NOT CHECKED' in stdout, got:\n{stdout}"
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn smoke_flag_succeeds_without_golden() {
    let dir = setup_data_dir("smoke");
    let out = Command::new(bin())
        .arg(&dir)
        .arg("--smoke")
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "--smoke must exit 0 even without golden; got {:?}",
        out.status.code()
    );
    // reconstruction still ran and wrote its panel
    assert!(
        dir.join("panel_weekly_rust.csv").exists(),
        "rust panel not written"
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn default_golden_path_is_data_dir_root_and_matching_golden_passes() {
    let dir = setup_data_dir("pass");
    // 1) produce the Rust panel via a smoke run
    let smoke = Command::new(bin())
        .arg(&dir)
        .arg("--smoke")
        .output()
        .unwrap();
    assert!(smoke.status.success());
    let rust_panel = dir.join("panel_weekly_rust.csv");
    assert!(rust_panel.exists());

    // 2) install it as the golden at the DEFAULT path the binary should look at
    let golden = dir.join("panel_weekly_frozen.csv");
    fs::copy(&rust_panel, &golden).unwrap();

    // 3) default run must find that golden and report a pass
    let out = Command::new(bin()).arg(&dir).output().unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("PARITY PASS"),
        "expected default golden path {} to be found and pass; stdout:\n{stdout}",
        golden.display()
    );
    assert!(out.status.success(), "matching golden must exit 0");
    let _ = fs::remove_dir_all(&dir);
}

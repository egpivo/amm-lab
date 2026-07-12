//! Integration test for the `event_study` harness: end-to-end wiring from real file formats
//! (panel CSV + feerev + tokens + matched_pairs) through design assembly to a coefficient
//! path. Uses a small internally-consistent synthetic dataset that crosses t0 so the design
//! has both pre-period leads and post-period lags. Pins:
//!   - dry run (no --estimate) -> exit 0, "DESIGN OK", no results written
//!   - --estimate -> exit 0, writes event_study_coefficients.csv, prints PRE-TREND + PROVISIONAL

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

const PANEL_HEADER: &str = "pool,unit_role,week,swaps,vol0,vol1,twl_active_liquidity,depth_1pct,depth_2pct,depth_5pct,lp_entry_count,lp_exit_count,unique_lp_count,jit_share_same_block,lp_fee_income_native1,lp_fee_income_per_active_liquidity,collect_amount1_native,position_duration_days,net_liq";

// 7 contiguous weeks around t0 = 2025-49 (index 3) -> event-times -3..3 (-1 omitted as ref).
const WEEKS: &[&str] = &[
    "2025-46", "2025-47", "2025-48", "2025-49", "2025-50", "2025-51", "2025-52",
];
const T0: &str = "2025-49";

fn setup_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("esgate_{}_{name}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();

    // 3 treated + 3 control; each treated matched to one control sharing a token pair.
    let treated = ["t0", "t1", "t2"];
    let control = ["c0", "c1", "c2"];
    // 3 distinct token pairs -> 3 clusters (CR1 needs >= 2)
    let pairs = [("0xaaa", "0xbbb"), ("0xaaa", "0xccc"), ("0xaaa", "0xddd")];

    // panel
    let mut panel = String::from(PANEL_HEADER);
    panel.push('\n');
    let mut push_rows = |pool: &str, role: &str, pool_ix: usize, treated_unit: bool| {
        for (wk_ix, wk) in WEEKS.iter().enumerate() {
            let post = treated_unit && wk_ix >= 3; // t0 at index 3
            // deterministic, well-conditioned outcome with a post-treatment bump
            let y =
                100.0 + pool_ix as f64 * 10.0 + wk_ix as f64 * 5.0 + if post { 20.0 } else { 0.0 };
            panel.push_str(&format!(
                "{pool},{role},{wk},0,0.0,0.0,{y},0.0,0.0,0.0,0,0,0,0.0,0.0,0.0,0.0,0.0,0\n"
            ));
        }
    };
    for (i, t) in treated.iter().enumerate() {
        push_rows(t, "matched_treated", i, true);
    }
    for (i, c) in control.iter().enumerate() {
        push_rows(c, "matched_control", i + 3, false);
    }
    fs::write(dir.join("panel_weekly_frozen.csv"), panel).unwrap();

    // feerev_panelvars.csv
    let mut fr = fs::File::create(dir.join("feerev_panelvars.csv")).unwrap();
    writeln!(
        fr,
        "pool,treated,tier,class,fr12_usd,fr4_usd,sw12,old,covered"
    )
    .unwrap();
    for t in treated {
        writeln!(fr, "{t},1,3000,weth-pair,1000,1,1,1,1").unwrap();
    }
    for c in control {
        writeln!(fr, "{c},0,3000,weth-pair,900,1,1,1,1").unwrap();
    }

    // ckpt_tokens.json: matched t/c share a pair
    let mut tok = String::from("{");
    let mut entries = Vec::new();
    for i in 0..3 {
        let (a, b) = pairs[i];
        entries.push(format!("\"{}\":[\"{a}\",\"{b}\"]", treated[i]));
        entries.push(format!("\"{}\":[\"{a}\",\"{b}\"]", control[i]));
    }
    tok.push_str(&entries.join(","));
    tok.push('}');
    fs::write(dir.join("ckpt_tokens.json"), tok).unwrap();

    // matched_pairs.json
    let mp: Vec<String> = (0..3)
        .map(|i| {
            format!(
                "{{\"treated\":\"{}\",\"controls\":[\"{}\"]}}",
                treated[i], control[i]
            )
        })
        .collect();
    fs::write(
        dir.join("matched_pairs.json"),
        format!("[{}]", mp.join(",")),
    )
    .unwrap();

    dir
}

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_event_study")
}

fn panel_csv(dir: &Path) -> PathBuf {
    dir.join("panel_weekly_frozen.csv")
}

#[test]
fn dry_run_reports_design_and_writes_nothing() {
    let dir = setup_dir("dry");
    let out = Command::new(bin())
        .arg(panel_csv(&dir))
        .arg(&dir)
        .arg("--t0")
        .arg(T0)
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "dry run should exit 0; stderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        stdout.contains("DESIGN OK"),
        "expected 'DESIGN OK'; got:\n{stdout}"
    );
    assert!(
        !dir.join("event_study_out").exists(),
        "dry run must not write results"
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn estimate_writes_coefficient_path_with_pretrend_and_provisional_labels() {
    let dir = setup_dir("est");
    let out = Command::new(bin())
        .arg(panel_csv(&dir))
        .arg(&dir)
        .arg("--t0")
        .arg(T0)
        .arg("--nboot")
        .arg("199")
        .arg("--estimate")
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.success(),
        "estimate should exit 0; stderr:\n{stderr}\nstdout:\n{stdout}"
    );
    assert!(
        stdout.contains("PRE-TREND"),
        "expected PRE-TREND section; got:\n{stdout}"
    );
    assert!(
        stdout.contains("PROVISIONAL"),
        "expected PROVISIONAL label; got:\n{stdout}"
    );
    let coefs = dir
        .join("event_study_out")
        .join("event_study_coefficients.csv");
    assert!(coefs.exists(), "coefficient path not written");
    let body = fs::read_to_string(&coefs).unwrap();
    // header + at least the lead/lag bins (-3,-2,0,1,2,3 => 6 rows)
    assert!(
        body.lines().count() >= 7,
        "expected coefficient rows; got:\n{body}"
    );
    let _ = fs::remove_dir_all(&dir);
}

//! Integration gate for the standalone WETH-USDC data-path CLIs:
//! fill spooling, aggTrades archive normalization, offline block-header
//! verification, and the thin application orchestrator. All offline.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

fn temp_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("pstt_standalone_{tag}_{}", std::process::id()));
    if dir.exists() {
        fs::remove_dir_all(&dir).unwrap();
    }
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn run_ok(bin: &str, args: &[&str]) -> String {
    let out = Command::new(bin).args(args).output().expect("spawn");
    assert!(
        out.status.success(),
        "{bin} failed: {}\n{}",
        String::from_utf8_lossy(&out.stderr),
        String::from_utf8_lossy(&out.stdout)
    );
    String::from_utf8_lossy(&out.stdout).to_string()
}

fn run_fail(bin: &str, args: &[&str]) {
    let out = Command::new(bin).args(args).output().expect("spawn");
    assert!(!out.status.success(), "{bin} unexpectedly succeeded");
}

const USDC: &str = "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48";
const WETH: &str = "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2";
const POOL5: &str = "0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640";
const POOL30: &str = "0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8";

#[test]
fn spool_fills_orients_and_filters() {
    let dir = temp_dir("spool");
    let swaps = dir.join("swaps.csv");
    // 2024-01-03 00:00:00 UTC = 1704240000
    fs::write(
        &swaps,
        format!(
            "type,block,pool,token0,token1,amount0,amount1\n\
             swap,100,{POOL5},{USDC},{WETH},-2500000000,1000000000000000000\n\
             swap,101,{POOL5},{USDC},{WETH},2600000000,-1000000000000000000\n\
             swap,102,{POOL30},{USDC},{WETH},-5100000000,2000000000000000000\n\
             mint,100,{POOL5},{USDC},{WETH},-1,1\n\
             swap,100,{POOL5},{USDC},{WETH},-1,0\n\
             swap,999,{POOL5},{USDC},{WETH},-1,1000000000000000000\n\
             swap,100,0x1111111111111111111111111111111111111111,{USDC},{WETH},-1,1000000000000000000\n"
        ),
    )
    .unwrap();
    let blkts = dir.join("blocks.csv");
    fs::write(
        &blkts,
        "block,x,y,timestamp\n100,_,_,1704240000\n101,_,_,1704240012\n102,_,_,1704240024\n",
    )
    .unwrap();
    let pools = dir.join("pools.json");
    fs::write(
        &pools,
        format!(
            r#"[
  {{"pool":"{POOL5}","label":"5bp","pair":"WETH-USDC","fee":500,"cex_symbol":"ETHUSDC","invert":false,"token0":"{USDC}","token1":"{WETH}"}},
  {{"pool":"{POOL30}","label":"30bp","pair":"WETH-USDC","fee":3000,"cex_symbol":"ETHUSDC","invert":false,"token0":"{USDC}","token1":"{WETH}"}}
]"#
        ),
    )
    .unwrap();
    let out = dir.join("out");
    run_ok(
        env!("CARGO_BIN_EXE_pstt_spool_fills"),
        &[
            "--swaps-csv",
            swaps.to_str().unwrap(),
            "--block-ts-csv",
            blkts.to_str().unwrap(),
            "--pools-json",
            pools.to_str().unwrap(),
            "--window-start-unix",
            "1704067200",
            "--window-end-unix",
            "1767139200",
            "--output-dir",
            out.to_str().unwrap(),
        ],
    );
    let fills = fs::read_to_string(out.join("pstt_oriented_fills.csv")).unwrap();
    let lines: Vec<&str> = fills.trim().lines().collect();
    // header + 3 kept fills (mint, a1==0, out-of-window block, foreign pool dropped)
    assert_eq!(lines.len(), 4, "fills:\n{fills}");
    // First fill: a1 > 0 (pool sells WETH to trader? frozen: direction=-1),
    // q=1.0, p_exec=|a0/a1|*1e12=2500.
    let f0: Vec<&str> = lines[1].split(',').collect();
    assert_eq!(f0[0], "1704240000");
    assert_eq!(f0[1], POOL5);
    assert_eq!(f0[2], "1");
    assert_eq!(f0[3], "2500");
    assert_eq!(f0[4], "-1");
    assert_eq!(f0[5], "2024-01");
    // Second fill: a1 < 0 -> direction +1, p_exec 2600.
    let f1: Vec<&str> = lines[2].split(',').collect();
    assert_eq!(f1[3], "2600");
    assert_eq!(f1[4], "1");
    // Third: 30bp pool, q=2, p_exec=5100/2=2550.
    let f2: Vec<&str> = lines[3].split(',').collect();
    assert_eq!(f2[1], POOL30);
    assert_eq!(f2[2], "2");
    assert_eq!(f2[3], "2550");
    // Refuses nonempty output dir by default.
    run_fail(
        env!("CARGO_BIN_EXE_pstt_spool_fills"),
        &[
            "--swaps-csv",
            swaps.to_str().unwrap(),
            "--block-ts-csv",
            blkts.to_str().unwrap(),
            "--pools-json",
            pools.to_str().unwrap(),
            "--window-start-unix",
            "1704067200",
            "--window-end-unix",
            "1767139200",
            "--output-dir",
            out.to_str().unwrap(),
        ],
    );
    fs::remove_dir_all(&dir).unwrap();
}

fn write_zip(path: &Path, inner_name: &str, contents: &str) {
    let file = fs::File::create(path).unwrap();
    let mut zw = zip::ZipWriter::new(file);
    zw.start_file::<_, ()>(inner_name, Default::default())
        .unwrap();
    zw.write_all(contents.as_bytes()).unwrap();
    zw.finish().unwrap();
}

#[test]
fn normalize_aggtrades_zip_csv_and_skip() {
    let dir = temp_dir("agg");
    let input = dir.join("in");
    fs::create_dir_all(&input).unwrap();
    // day 1: zip archive, out-of-order rows (ms stamps)
    write_zip(
        &input.join("ETHUSDC-aggTrades-2024-01-02.zip"),
        "ETHUSDC-aggTrades-2024-01-02.csv",
        "2,2501.0,0.2,2,2,1704153601000,true,true\n1,2500.0,0.1,1,1,1704153600000,true,true\n",
    );
    // day 2: plain csv with us stamps (2025-style)
    fs::write(
        input.join("ETHUSDC-aggTrades-2024-01-03.csv"),
        "3,2502.0,0.3,3,3,1704240000000000,true,true\n",
    )
    .unwrap();
    // day 3: empty file -> skipped
    fs::write(input.join("ETHUSDC-aggTrades-2024-01-04.csv"), "").unwrap();
    let out = dir.join("out");
    let stdout = run_ok(
        env!("CARGO_BIN_EXE_pstt_normalize_aggtrades"),
        &[
            "--input-dir",
            input.to_str().unwrap(),
            "--symbol",
            "ETHUSDC",
            "--output-dir",
            out.to_str().unwrap(),
        ],
    );
    assert!(stdout.contains("\"rows\": 3"), "{stdout}");
    assert!(stdout.contains("2024-01-04"), "skip list: {stdout}");
    let day1 = fs::read_to_string(out.join("ETHUSDC/ETHUSDC-aggTrades-2024-01-02.csv")).unwrap();
    let rows: Vec<&str> = day1.trim().lines().collect();
    assert_eq!(rows.len(), 2);
    // sorted by timestamp, re-emitted as 16-digit us stamps
    assert!(rows[0].contains(",2500,") && rows[0].contains("1704153600000000"));
    assert!(rows[1].contains(",2501,") && rows[1].contains("1704153601000000"));
    // Output is loadable by the weekly builder's aggTrades loader
    // (headerless frozen column layout).
    let reloaded = amm_lab::pstt::cex::load_aggtrades_csv(
        &out.join("ETHUSDC/ETHUSDC-aggTrades-2024-01-02.csv"),
        false,
    )
    .unwrap();
    assert_eq!(reloaded.len(), 2);
    assert!((reloaded[0].timestamp_secs - 1_704_153_600.0).abs() < 1e-9);
    fs::remove_dir_all(&dir).unwrap();
}

const H1: &str = "0x1111111111111111111111111111111111111111111111111111111111111111";
const H2: &str = "0x2222222222222222222222222222222222222222222222222222222222222222";
const H3: &str = "0x3333333333333333333333333333333333333333333333333333333333333333";
const H0: &str = "0x0000000000000000000000000000000000000000000000000000000000000000";

#[test]
fn verify_block_headers_gates() {
    let dir = temp_dir("hdr");
    let good = dir.join("good.csv");
    // Blocks 10, 11 consecutive (parent chain checked), 20 sparse.
    fs::write(
        &good,
        format!(
            "block,block_hash,parent_hash,timestamp_unix\n\
             10,{H1},{H0},1704240000\n\
             11,{H2},{H1},1704240012\n\
             20,{H3},{H0},1704240120\n"
        ),
    )
    .unwrap();
    let blocks = dir.join("blocks.txt");
    fs::write(&blocks, "10\n11\n20\n").unwrap();
    let report = dir.join("report.json");
    let stdout = run_ok(
        env!("CARGO_BIN_EXE_pstt_verify_block_headers"),
        &[
            "--headers-csv",
            good.to_str().unwrap(),
            "--blocks-file",
            blocks.to_str().unwrap(),
            "--report-json",
            report.to_str().unwrap(),
        ],
    );
    assert!(stdout.contains("\"pass\": true"), "{stdout}");
    // Broken parent chain on consecutive blocks -> fail.
    let bad = dir.join("bad.csv");
    fs::write(
        &bad,
        format!(
            "block,block_hash,parent_hash,timestamp_unix\n\
             10,{H1},{H0},1704240000\n\
             11,{H2},{H3},1704240012\n"
        ),
    )
    .unwrap();
    run_fail(
        env!("CARGO_BIN_EXE_pstt_verify_block_headers"),
        &["--headers-csv", bad.to_str().unwrap()],
    );
    // Missing coverage -> fail.
    fs::write(&blocks, "10\n11\n20\n30\n").unwrap();
    run_fail(
        env!("CARGO_BIN_EXE_pstt_verify_block_headers"),
        &[
            "--headers-csv",
            good.to_str().unwrap(),
            "--blocks-file",
            blocks.to_str().unwrap(),
        ],
    );
    // Non-monotone timestamps -> fail.
    let unordered = dir.join("unordered.csv");
    fs::write(
        &unordered,
        format!(
            "block,block_hash,parent_hash,timestamp_unix\n\
             10,{H1},{H0},1704240012\n\
             20,{H3},{H0},1704240000\n"
        ),
    )
    .unwrap();
    run_fail(
        env!("CARGO_BIN_EXE_pstt_verify_block_headers"),
        &["--headers-csv", unordered.to_str().unwrap()],
    );
    fs::remove_dir_all(&dir).unwrap();
}

fn synthetic_weekly_json() -> String {
    // 12 common weeks; 5bp strongly negative L, 30bp mildly negative.
    let mut lab5 = Vec::new();
    let mut lab30 = Vec::new();
    for i in 0..12 {
        let week = format!("2024-{:02}", i + 1);
        let l5 = -50.0 + (i as f64 % 5.0);
        let l30 = -5.0 + (i as f64 % 3.0);
        let s5 = 20.0 + (i as f64 % 3.0);
        let s30 = 18.0 + (i as f64 % 4.0);
        lab5.push(serde_json::json!({
            "week": week, "L": l5, "A": 0.0, "B": -l5, "S": s5,
            "Om": 10.0, "q2": s5 * s5 / 10.0, "n": 10
        }));
        lab30.push(serde_json::json!({
            "week": week, "L": l30, "A": 0.0, "B": -l30, "S": s30,
            "Om": 9.0, "q2": s30 * s30 / 9.0, "n": 9
        }));
    }
    serde_json::json!({
        "5bp": {"last_trade": lab5, "vwap1s": lab5},
        "30bp": {"last_trade": lab30, "vwap1s": lab30},
    })
    .to_string()
}

#[test]
fn application_orchestrator_is_deterministic_and_replayable() {
    let dir = temp_dir("app");
    let weekly = dir.join("weekly.json");
    fs::write(&weekly, synthetic_weekly_json()).unwrap();

    let run_app = |out: &Path, extra: &[&str]| -> serde_json::Value {
        let mut args = vec![
            "--weekly-json",
            weekly.to_str().unwrap(),
            "--lower-label",
            "5bp",
            "--higher-label",
            "30bp",
            "--draws",
            "300",
            "--output-dir",
            out.to_str().unwrap(),
        ];
        args.extend_from_slice(extra);
        run_ok(env!("CARGO_BIN_EXE_pstt_build_application"), &args);
        serde_json::from_str(
            &fs::read_to_string(out.join("pstt_application_manifest.json")).unwrap(),
        )
        .unwrap()
    };

    // Same seed twice -> byte-identical manifests (determinism).
    let m1 = run_app(&dir.join("run1"), &["--seed", "990000000"]);
    let m2 = run_app(&dir.join("run2"), &["--seed", "990000000"]);
    assert_eq!(m1, m2, "seeded runs must be deterministic");
    // Different seed -> same structure, generally different numbers.
    let m3 = run_app(&dir.join("run3"), &["--seed", "12345"]);
    assert_eq!(
        m1["claim_table"][0]["contrast"],
        m3["claim_table"][0]["contrast"]
    );

    // Claim table shape and grid completeness.
    let claim = &m1["claim_table"][0];
    assert!(claim["primary_last_trade"].is_string());
    assert!(claim["robustness_vwap1s"].is_string());
    assert!(
        claim["reference_robustness"] == "STABLE"
            || claim["reference_robustness"] == "REFERENCE-SENSITIVE"
    );
    let regions = &m1["pools"]["5bp"]["last_trade"]["outer_regions"];
    for key in ["0.0", "0.05", "0.1", "0.5", "1.0"] {
        assert!(!regions[key].is_null() || regions.get(key).is_some());
    }
    // identical references in fixture -> statuses equal -> STABLE
    assert_eq!(claim["reference_robustness"], "STABLE");

    // Index-file replay: extract nothing from the run; build a trivial
    // schedule file and confirm exact replay determinism.
    let n_weeks = 12;
    let draws = 40;
    let mut sched = serde_json::Map::new();
    let mut mat = Vec::new();
    for d in 0..draws {
        // deterministic rotation schedule
        let row: Vec<usize> = (0..n_weeks).map(|i| (i + d) % n_weeks).collect();
        mat.push(row);
    }
    for key in [
        "5bp:last_trade",
        "5bp:vwap1s",
        "30bp:last_trade",
        "30bp:vwap1s",
        "rank:last_trade",
        "rank:vwap1s",
    ] {
        sched.insert(key.to_string(), serde_json::to_value(&mat).unwrap());
    }
    let index_file = dir.join("schedule.json");
    fs::write(&index_file, serde_json::to_string(&sched).unwrap()).unwrap();
    let r1 = run_app(
        &dir.join("replay1"),
        &["--index-file", index_file.to_str().unwrap()],
    );
    let r2 = run_app(
        &dir.join("replay2"),
        &["--index-file", index_file.to_str().unwrap()],
    );
    assert_eq!(r1, r2, "index-file replay must be exact");
    assert_eq!(
        r1["provenance"]["bootstrap"],
        "externally serialized index schedule (exact replay)"
    );
    fs::remove_dir_all(&dir).unwrap();
}

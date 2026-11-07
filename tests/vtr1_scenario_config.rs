//! Scenario JSON parsing parity with `scenario_config.py` (shared fixture). Parses JSON only:
//! no environment is constructed and nothing is simulated.

use amm_lab::sim::env::ScenarioConfig;

#[test]
fn rust_accepts_and_rejects_the_shared_fixture_cases() {
    let cases: Vec<serde_json::Value> =
        serde_json::from_str(include_str!("fixtures/vtr1_scenario_cases.json")).unwrap();
    assert!(cases.len() >= 20);
    for c in cases {
        let (name, valid) = (c["name"].as_str().unwrap(), c["valid"].as_bool().unwrap());
        let parsed = ScenarioConfig::from_json(&c["json"].to_string());
        assert_eq!(parsed.is_ok(), valid, "{name}: {parsed:?}");
    }
}

#[test]
fn non_finite_tokens_are_rejected() {
    for text in [
        r#"{"sigma": NaN}"#,
        r#"{"sigma": Infinity}"#,
        r#"{"sigma": 1e999}"#,
    ] {
        assert!(ScenarioConfig::from_json(text).is_err(), "{text}");
    }
}

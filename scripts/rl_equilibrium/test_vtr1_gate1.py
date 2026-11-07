"""Static / unit tests for the VTR-1 Gate 1 instrumentation. None runs the simulator or a DQN training."""

import argparse
import json
import random
import re
from pathlib import Path

import pytest
import torch
import torch.nn as nn

import seed_blocks
import dqn_train
import scenario_config
import seed_collision_audit
import vtr1_eval
import vtr1_reference
from training_validity import TrainingInvalid, ValidityMonitor

HERE = Path(__file__).parent


def test_registry_is_disjoint_and_finite():
    seed_blocks.assert_registry_disjoint()
    assert all(b.end > b.start for b in seed_blocks.BLOCKS)
    # the historical training block is finite and no longer swallows the VTR-1 blocks
    hist = next(b for b in seed_blocks.BLOCKS if b.name == "hist_train")
    assert (hist.start, hist.end) == (1_000_000, 1_012_000)


def test_default_dqn_train_ranges_are_accepted():
    seed_blocks.require("train", 1_000_000, 12_000)
    seed_blocks.require("val", 20_000, 50)
    seed_blocks.require("eval", 30_000, 500)
    seed_blocks.require("eval", 40_000, 500)


def test_defaults_in_dqn_train_source_are_the_historical_literals():
    src = (HERE / "dqn_train.py").read_text()
    for flag, value in [
        ("--train-seed-base", "1_000_000"),
        ("--val-seed-base", "20_000"),
        ("--test-seed-base", "30_000"),
        ("--fresh-seed-base", "40_000"),
    ]:
        assert re.search(rf'"{flag}", type=int, default={value}\)', src)


@pytest.mark.parametrize(
    "use,base,n",
    [
        ("train", 1_000_000, 12_001),  # leaves the registered training block
        ("train", 190_000, 10),  # final block
        ("val", 20_150, 100),  # runs past dev_val into unregistered seeds
        ("eval", 170_000, 10),  # pilot block
        ("train", 10_000_000 - 1, 5),  # straddles into W6 R0 from outside
        ("eval", 250_000, 10),  # W0 block
    ],
)
def test_overlaps_and_unregistered_ranges_abort(use, base, n):
    with pytest.raises(SystemExit):
        seed_blocks.require(use, base, n)


def _auth(tmp_path, scope, blocks, name="auth.json"):
    f = tmp_path / name
    f.write_text(json.dumps({"scope": scope, "blocks": blocks}))
    return str(f)


def test_protected_uses_need_a_structured_authorization_file(tmp_path):
    with pytest.raises(SystemExit):
        seed_blocks.require("final", 190_000, 1_000)
    with pytest.raises(SystemExit):
        seed_blocks.require("pilot", 170_000, 200, authorization=str(tmp_path / "missing"))
    plain = tmp_path / "plain.txt"
    plain.write_text("x")
    with pytest.raises(SystemExit):  # not JSON
        seed_blocks.require("pilot", 170_000, 200, authorization=str(plain))
    with pytest.raises(SystemExit):  # wrong scope
        seed_blocks.require("pilot", 170_000, 200, authorization=_auth(tmp_path, "final", ["vtr1_pilot"]))
    with pytest.raises(SystemExit):  # block not named
        seed_blocks.require("pilot", 170_000, 200, authorization=_auth(tmp_path, "pilot", ["vtr1_final"]))
    ok = _auth(tmp_path, "pilot", ["vtr1_pilot"])
    assert seed_blocks.require("pilot", 170_000, 200, authorization=ok).name == "vtr1_pilot"


def test_vtr1_training_blocks_are_protected_roles(tmp_path):
    with pytest.raises(SystemExit):
        seed_blocks.require("train", 30_000_000, 12_000)  # wrong role
    with pytest.raises(SystemExit):
        seed_blocks.require("train", 10_000_000, 12_000)  # W6 is no longer an ordinary train block
    with pytest.raises(SystemExit):
        seed_blocks.require("pilot_train", 30_000_000, 12_000)  # no authorization
    with pytest.raises(SystemExit):
        seed_blocks.require("vtr1_train", 10_000_000, 12_000)  # no authorization
    ok = _auth(tmp_path, "pilot_train", ["pilot_train"])
    assert seed_blocks.require("pilot_train", 30_000_000, 12_000, ok).name == "pilot_train"
    for r in range(3):
        name = f"w6_train_R{r}"
        auth = _auth(tmp_path, "vtr1_train", [name], name + ".json")
        assert seed_blocks.require("vtr1_train", 10_000_000 + r * 1_000_000, 12_000, auth).name == name
    with pytest.raises(SystemExit):  # R0 authorization does not cover R1
        seed_blocks.require("vtr1_train", 11_000_000, 12_000, _auth(tmp_path, "vtr1_train", ["w6_train_R0"], "r0.json"))


def test_new_outputs_must_not_exist_or_be_archived(tmp_path):
    sentinel = tmp_path / "keep.csv"
    sentinel.write_text("original")
    with pytest.raises(SystemExit):
        seed_blocks.check_new_outputs([tmp_path / "new.csv", sentinel])
    assert sentinel.read_text() == "original"
    archived = Path(seed_blocks.__file__).resolve().parents[2] / "data/rl_equilibrium/never_written.csv"
    with pytest.raises(SystemExit):
        seed_blocks.check_new_outputs([archived])
    seed_blocks.check_new_outputs([tmp_path / "new.csv"])


def test_no_derived_seed_collision_over_all_blocks():
    result = seed_collision_audit.audit()
    assert result["collisions"] == []
    assert list(result["streams"]) == ["oracle", "noise", "arb", "order", "jit", "xi", "eta"]
    assert result["streams"]["xi"] == (1_597_334_677, 5)
    assert result["streams"]["eta"] == (374_761_393, 6)
    assert result["streams"]["noise"] == (2_654_435_761, 1)  # existing streams untouched


def _tiny():
    torch.manual_seed(3)
    net = nn.Linear(4, 2)
    target = nn.Linear(4, 2)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    return net, target, opt


def _update(net, target, opt, monitor, step=1):
    monitor.begin_update()
    states, nxt, rew = torch.randn(8, 4), torch.randn(8, 4), torch.randn(8)
    dones = torch.zeros(8)
    monitor.check_inputs(states, nxt, rew, dones)
    online_q = net(states)
    values = online_q.max(1).values
    with torch.inference_mode():
        target_q = target(nxt)
        targets = rew + (1 - dones) * target_q.max(1).values
    loss = nn.functional.smooth_l1_loss(values, targets.clone())
    monitor.check_q_outputs(online_q, target_q, targets)
    monitor.check_loss(loss)
    opt.zero_grad()
    loss.backward()
    monitor.check_grads(net)
    opt.step()
    monitor.check_after_step(net, opt)
    target.load_state_dict(net.state_dict())
    monitor.check_target_sync(target)


def test_checks_do_not_touch_rng_or_parameters():
    results = []
    for enabled in (False, True):
        net, target, opt = _tiny()
        random.seed(1)
        mon = ValidityMonitor(enabled)
        for step in range(5):
            _update(net, target, opt, mon, step)
        results.append(([p.clone() for p in net.parameters()], torch.get_rng_state(), random.getstate()))
    for a, b in zip(results[0][0], results[1][0]):
        assert torch.equal(a, b)
    assert torch.equal(results[0][1], results[1][1])
    assert results[0][2] == results[1][2]


def test_every_contract_check_is_exercised_and_counted():
    net, target, opt = _tiny()
    mon = ValidityMonitor(True, {"replicate_id": "R0"})
    _update(net, target, opt, mon)
    for name in ["minibatch_input", "online_q", "target_q", "loss", "gradient", "parameters",
                 "adam_exp_avg", "adam_exp_avg_sq", "target_network"]:
        assert mon.counts[name]["checked"] > 0 and mon.counts[name]["nonfinite"] == 0
    rec = mon.record("t")
    assert rec["status"] == "VALID" and rec["replicate_id"] == "R0" and rec["first_failure"] is None


def test_validation_q_values_are_checked_only_inside_the_eval_scope():
    net, _, _ = _tiny()
    mon = ValidityMonitor(True)
    net.register_forward_hook(mon.eval_forward_hook)
    net(torch.randn(2, 4))
    assert "validation_q" not in mon.counts  # training-phase forward passes are not validation
    with mon.eval_scope():
        net(torch.randn(2, 4))
        assert mon.counts["validation_q"]["checked"] == 1
        with torch.no_grad():
            net.weight.fill_(float("nan"))
        with pytest.raises(TrainingInvalid) as err:
            net(torch.randn(2, 4))
    assert err.value.quantity == "validation_q" and mon.phase == "train"


@pytest.mark.parametrize(
    "quantity,label",
    [
        ("minibatch_input", "dones"),
        ("online_q", "online_q"),
        ("target_q", "target_q"),
        ("loss", "loss"),
        ("gradient", "bias"),
        ("parameters", "weight"),
        ("adam_exp_avg", "weight"),
        ("adam_exp_avg_sq", "bias"),
        ("target_network", "weight"),
    ],
)
def test_each_nonfinite_quantity_is_detected_with_location(quantity, label):
    net, target, opt = _tiny()
    mon = ValidityMonitor(True)
    mon.begin_episode(3)
    mon.begin_env_step(42)
    for _ in range(7):
        _update(net, target, opt, mon, 0)
    mon.begin_update()  # the 8th update is the one that fails
    bad = torch.tensor([1.0, float("nan")])
    ok = torch.ones(2)
    with pytest.raises(TrainingInvalid) as err:
        if quantity == "minibatch_input":
            mon.check_inputs(ok, ok, ok, bad)
        elif quantity == "online_q":
            mon.check_q_outputs(bad, ok, ok)
        elif quantity == "target_q":
            mon.check_q_outputs(ok, bad, ok)
        elif quantity == "loss":
            mon.check_loss(bad)
        elif quantity == "gradient":
            for p in net.parameters():
                p.grad = torch.ones_like(p)
            net.bias.grad[1] = float("inf")
            mon.check_grads(net)
        elif quantity == "parameters":
            with torch.no_grad():
                net.weight[0, 0] = float("nan")
            mon.check_after_step(net, opt)
        elif quantity in ("adam_exp_avg", "adam_exp_avg_sq"):
            key = "exp_avg" if quantity == "adam_exp_avg" else "exp_avg_sq"
            state = opt.state[dict(net.named_parameters())[label]]
            state[key].view(-1)[0] = float("inf")
            mon.check_after_step(net, opt)
        else:
            with torch.no_grad():
                target.weight[1, 1] = float("nan")
            mon.check_target_sync(target)
    assert err.value.quantity == quantity
    ff = mon.record("t")["first_failure"]
    assert mon.record("t")["status"] == "TRAINING_INVALID"
    assert ff["update_index"] == 8 and ff["quantity"] == quantity and ff["tensor"] == label
    assert ff["phase"] == "train" and ff["env_step"] == 42 and ff["training_episode"] == 3 and "value" in ff
    assert isinstance(ff["flat_index"], int)


def test_scenario_file_is_validated_and_forwarded(tmp_path):
    f = tmp_path / "s1.json"
    f.write_text(json.dumps({"name": "S1-0.65", "sigma": 0.65}))
    assert scenario_config.overrides(str(f)) == {"sigma": 0.65}
    assert scenario_config.overrides(None) == {}
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"sigma": 0.65, "volatility": 1}))
    with pytest.raises(SystemExit):
        scenario_config.load(str(bad))
    neg = tmp_path / "neg.json"
    neg.write_text(json.dumps({"sigma": -1}))
    with pytest.raises(SystemExit):
        scenario_config.load(str(neg))
    with pytest.raises(SystemExit):
        scenario_config.check_cell("dynamic_duopoly/sideways")


def _ref_args(**kw):
    base = dict(cell="dynamic_duopoly/before", scenario=None, tune_kappa=False, role="eval",
                seed_base=30_000, n=100, authorization=None, kappa=16.0, seed_label="vtr1",
                out="nonexistent_ref.csv", binary="bin/ref")
    base.update(kw)
    return argparse.Namespace(**base)


def test_reference_wrapper_applies_registry_guard_and_scenario(tmp_path):
    sc = tmp_path / "sc.json"
    sc.write_text(json.dumps({"sigma": 0.8}))
    cmd = vtr1_reference.build_command(_ref_args(scenario=str(sc)))
    assert cmd[:3] == ["bin/ref", "--cell", "dynamic_duopoly/before"]
    assert "--scenario" in cmd and "--seed-base" in cmd
    with pytest.raises(SystemExit):  # final block without authorization
        vtr1_reference.build_command(_ref_args(role="final", seed_base=190_000, n=1000))
    auth = _auth(tmp_path, "final", ["vtr1_final"])
    vtr1_reference.build_command(_ref_args(role="final", seed_base=190_000, n=1000, authorization=auth))
    with pytest.raises(SystemExit):  # training seeds are not evaluation seeds
        vtr1_reference.build_command(_ref_args(seed_base=1_000_000))
    with pytest.raises(SystemExit):  # tuning outside validation block B
        vtr1_reference.build_command(_ref_args(tune_kappa=True, seed_base=170_000, n=50))


def test_vtr1_kappa_tuning_needs_scoped_authorization_and_never_spawns(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("subprocess must not run in a rejected or dry invocation")

    monkeypatch.setattr(vtr1_reference.subprocess, "run", boom)
    base = dict(tune_kappa=True, seed_base=150_000, n=50)
    for auth in (None, str(tmp_path / "missing.json")):
        with pytest.raises(SystemExit):
            vtr1_reference.build_command(_ref_args(authorization=auth, **base))
    wrong_scope = _auth(tmp_path, "vtr1_train", ["vtr1_val_B"], "ws.json")
    wrong_block = _auth(tmp_path, "kappa_tuning", ["vtr1_pilot"], "wb.json")
    for auth in (wrong_scope, wrong_block):
        with pytest.raises(SystemExit):
            vtr1_reference.build_command(_ref_args(authorization=auth, **base))
    good = _auth(tmp_path, "kappa_tuning", ["vtr1_val_B"], "ok.json")
    tune = vtr1_reference.build_command(_ref_args(authorization=good, **base))
    assert "--tune-kappa" in tune and tune[tune.index("--tune-seed-base") + 1] == "150000"


def test_reference_output_must_be_new(tmp_path):
    existing = tmp_path / "ref.csv"
    existing.write_text("keep")
    with pytest.raises(SystemExit):
        vtr1_reference.build_command(_ref_args(out=str(existing)))
    assert existing.read_text() == "keep"


def test_eval_entry_point_plans_without_the_simulator(tmp_path):
    ckpt = tmp_path / "net.pt"
    ckpt.write_bytes(b"x")
    ns = vtr1_eval.parser().parse_args(
        ["--checkpoint", str(ckpt), "--cell", "dynamic_duopoly/random", "--role", "eval",
         "--seed-base", "30000", "--n", "10", "--out", str(tmp_path / "o.csv")]
    )
    block, overrides, (mode, order) = vtr1_eval.plan(ns)
    assert (block.name, mode, order, overrides["agent_order"]) == ("dev_test", "dynamic_duopoly", "random", "random")
    assert overrides["completion_rule"] == "forced_terminal"  # mandatory, not configurable
    assert "--completion-rule" not in vtr1_eval.parser().format_help()
    ns.role, ns.seed_base, ns.n = "final", 190_000, 1000
    with pytest.raises(SystemExit):
        vtr1_eval.plan(ns)  # no authorization


def test_w0_harness_blocks_match_registry():
    src = (HERE.parents[1] / "src/bin/vtr1_w0.rs").read_text()
    db1 = re.search(r"const DB1: \(u64, u64\) = \(([\d_]+), ([\d_]+)\);", src)
    db2 = re.search(r"const DB2: \(u64, u64\) = \(([\d_]+), ([\d_]+)\);", src)
    by_name = {b.name: b for b in seed_blocks.BLOCKS}
    for m, name in ((db1, "w0_paths_DB1"), (db2, "w0_days_DB2")):
        assert (int(m[1].replace("_", "")), int(m[2].replace("_", ""))) == (by_name[name].start, by_name[name].end)
    assert "check_authorization" in src  # refuses to run without a structured W0 authorization


def test_w0_harness_blocks_match_registry():
    src = (HERE.parents[1] / "src/bin/vtr1_w0.rs").read_text()
    db1 = re.search(r"const DB1: \(u64, u64\) = \(([\d_]+), ([\d_]+)\);", src)
    db2 = re.search(r"const DB2: \(u64, u64\) = \(([\d_]+), ([\d_]+)\);", src)
    by_name = {b.name: b for b in seed_blocks.BLOCKS}
    for m, name in ((db1, "w0_paths_DB1"), (db2, "w0_days_DB2")):
        assert (int(m[1].replace("_", "")), int(m[2].replace("_", ""))) == (by_name[name].start, by_name[name].end)
    assert "check_authorization" in src  # refuses to run without a structured W0 authorization


def test_dqn_train_is_wired_to_monitor_hooks_and_resolve():
    src = (HERE / "dqn_train.py").read_text()
    for hook in ["check_q_outputs", "check_loss", "check_grads", "check_after_step", "check_target_sync",
                 "eval_scope", "register_forward_hook", "begin_episode", "begin_env_step", "begin_update"]:
        assert hook in src
    assert "plan = resolve(args, parser.error)" in src and "run(args, monitor, plan)" in src
    assert src.count("**scenario") >= 2  # training and validation overrides carry the scenario

def _train_args(**kw):
    base = dict(episodes=12_000, eval_every=500, n_val=50, n_test=500, mode="dynamic_duopoly",
                agent_order="before", train_penalty=0.02, tag="t", seed=7, train_seed_base=1_000_000,
                val_seed_base=20_000, test_seed_base=30_000, fresh_seed_base=40_000,
                check_finite=False, validity_out=None, scenario=None, train_role="train",
                authorization=None, no_test_eval=False, replicate_id=None, out_dir=None)
    base.update(kw)
    return argparse.Namespace(**base)


def _fail(msg):
    raise SystemExit(msg)


def _w6(tmp_path, **kw):
    sc = tmp_path / "sc.json"
    sc.write_text(json.dumps({"name": "S1", "sigma": 0.65}))
    auth = _auth(tmp_path, "vtr1_train", ["w6_train_R0", "vtr1_val_B"])
    return _train_args(train_role="vtr1_train", train_seed_base=10_000_000, val_seed_base=150_000,
                       scenario=str(sc), authorization=auth, out_dir=str(tmp_path / "out"), **kw)


def test_legacy_training_plan_is_unchanged():
    plan = dqn_train.resolve(_train_args(), _fail)
    assert plan.out_dir == dqn_train.OUT and plan.train_block.name == "hist_train"
    assert plan.train_ov == {"agent_order": "before", "unfinished_penalty": 0.02}
    assert plan.eval_ov == {"agent_order": "before"} and plan.val_ov == plan.eval_ov  # Standard completion kept
    with pytest.raises(SystemExit):  # a scenario needs a protected role
        dqn_train.resolve(_train_args(scenario="x.json"), _fail)
    with pytest.raises(SystemExit):  # legacy role cannot use W6 seeds
        dqn_train.resolve(_train_args(train_seed_base=10_000_000), _fail)


def test_vtr1_training_forces_instrumentation_and_forced_terminal_validation(tmp_path):
    args = _w6(tmp_path)
    assert args.check_finite is False and args.no_test_eval is False
    plan = dqn_train.resolve(args, _fail)
    assert args.check_finite is True and args.no_test_eval is True  # cannot be bypassed
    assert plan.val_block.name == "vtr1_val_B" and plan.train_block.name == "w6_train_R0"
    assert plan.val_ov["completion_rule"] == "forced_terminal" and plan.val_ov["sigma"] == 0.65
    assert "completion_rule" not in plan.train_ov  # training episodes keep their completion rule
    assert plan.train_ov["unfinished_penalty"] == 0.02
    assert not any(p.name.startswith("m3_dqn_results") for p in plan.outputs)
    assert not (tmp_path / "out").exists()  # planning creates nothing


def test_vtr1_training_rejections_happen_before_any_file_is_created(tmp_path):
    for kw in (dict(authorization=None), dict(scenario=None), dict(out_dir=None)):
        args = _w6(tmp_path)
        for k, v in kw.items():
            setattr(args, k, v)
        with pytest.raises(SystemExit):
            dqn_train.resolve(args, _fail)
    args = _w6(tmp_path)
    args.authorization = _auth(tmp_path, "vtr1_train", ["w6_train_R1", "vtr1_val_B"], "other.json")
    with pytest.raises(SystemExit):  # authorization names another W6 block
        dqn_train.resolve(args, _fail)
    args = _w6(tmp_path)
    args.authorization = _auth(tmp_path, "vtr1_train", ["w6_train_R0"], "noval.json")
    with pytest.raises(SystemExit):  # validation block B not covered
        dqn_train.resolve(args, _fail)
    assert not (tmp_path / "out").exists()


def test_vtr1_training_never_overwrites_any_output(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    for name in ("dqn_t.pt", "m3_dqn_curve_t.csv", "validity_t.json", "custom_validity.json"):
        (out / name).write_text("sentinel")
    for validity_out in (None, str(out / "custom_validity.json")):
        with pytest.raises(SystemExit):
            dqn_train.resolve(_w6(tmp_path, validity_out=validity_out), _fail)
    assert {p.read_text() for p in out.iterdir()} == {"sentinel"}
    # a fresh directory with an explicit validity path is accepted
    plan = dqn_train.resolve(_w6(tmp_path, validity_out=str(tmp_path / "v.json"), tag="u"), _fail)
    assert plan.validity_path == tmp_path / "v.json"


def test_legacy_runs_with_finite_checks_also_never_overwrite(tmp_path):
    out = tmp_path / "o"
    out.mkdir()
    (out / "m3_dqn_results_t.csv").write_text("sentinel")
    with pytest.raises(SystemExit):
        dqn_train.resolve(_train_args(check_finite=True, out_dir=str(out)), _fail)
    assert (out / "m3_dqn_results_t.csv").read_text() == "sentinel"


def test_validation_helper_matches_evaluate_and_tracks_validation_episodes(monkeypatch):
    seen = []

    def fake_episode(env, net, seed, mode, **ov):
        seen.append((seed, mon.validation_seed, ov["completion_rule"]))
        return {"shortfall_bps": float(seed % 7)}

    mon = ValidityMonitor(True)
    monkeypatch.setattr(dqn_train, "greedy_episode", fake_episode)
    value = dqn_train.validate(None, None, range(150_000, 150_004), "m", mon, completion_rule="forced_terminal")
    assert value == sum(s % 7 for s in range(150_000, 150_004)) / 4
    assert seen == [(s, s, "forced_terminal") for s in range(150_000, 150_004)]


def test_both_validation_calls_use_the_validation_overrides():
    src = (HERE / "dqn_train.py").read_text()
    assert src.count("validate(env, net, val_seeds, args.mode, monitor, **val_ov)") == 2  # periodic + fallback
    assert "evaluate(" not in src.replace("validate(", "")  # no un-overridden validation path remains
    assert "greedy_episode(env, net, seed, args.mode, **eval_ov)" in src  # legacy test eval unchanged


def test_validation_failure_reports_validation_location_not_training_counters():
    net, target, opt = _tiny()
    mon = ValidityMonitor(True)
    net.register_forward_hook(mon.eval_forward_hook)
    mon.begin_episode(499)
    mon.begin_env_step(123_456)
    for _ in range(3):
        mon.begin_update()
    with mon.eval_scope():
        mon.begin_validation_episode(150_007)
        net(torch.randn(1, 4))
        net(torch.randn(1, 4))
        with torch.no_grad():
            net.weight.fill_(float("nan"))
        with pytest.raises(TrainingInvalid):
            net(torch.randn(1, 4))
    ff = mon.record("t")["first_failure"]
    assert (ff["phase"], ff["validation_seed"], ff["validation_step"]) == ("eval", 150_007, 2)
    assert (ff["training_episode"], ff["env_step"], ff["update_index"]) == (499, 123_456, 3)
    assert ff["quantity"] == "validation_q" and ff["tensor"] == "validation_q" and ff["flat_index"] == 0


def test_completion_rule_is_interpreted_consistently_by_python_bridge_and_reference():
    bridge = (HERE.parents[1] / "src/bin/rl_equilibrium_bridge.rs").read_text()
    ref = (HERE.parents[1] / "src/bin/rl_equilibrium_reference.rs").read_text()
    assert vtr1_eval.FORCED_TERMINAL == "forced_terminal"
    assert 'Some("forced_terminal") =>' in bridge and "CompletionRule::ForcedTerminal" in bridge
    assert "unsupported completion_rule" in bridge  # unknown values are errors, not Standard
    assert ref.count("cfg.completion_rule = CompletionRule::ForcedTerminal;") >= 2  # tuning and evaluation


def test_scenario_fixture_parity_with_rust_cases():
    cases = json.loads((HERE.parents[1] / "tests/fixtures/vtr1_scenario_cases.json").read_text())
    assert len(cases) >= 20
    for c in cases:
        try:
            scenario_config.validate(c["json"])
            ok = True
        except SystemExit:
            ok = False
        assert ok == c["valid"], c["name"]


def test_scenario_loader_rejects_booleans_and_non_finite_tokens(tmp_path):
    for text in ('{"sigma": true}', '{"sigma": NaN}', '{"sigma": Infinity}', '{"sigma": -Infinity}'):
        f = tmp_path / "s.json"
        f.write_text(text)
        with pytest.raises(SystemExit):
            scenario_config.load(str(f))

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


def test_pilot_training_block_is_reachable_only_with_its_role_and_authorization(tmp_path):
    with pytest.raises(SystemExit):
        seed_blocks.require("train", 30_000_000, 12_000)  # wrong role
    with pytest.raises(SystemExit):
        seed_blocks.require("pilot_train", 30_000_000, 12_000)  # no authorization
    ok = _auth(tmp_path, "pilot_train", ["pilot_train"])
    assert seed_blocks.require("pilot_train", 30_000_000, 12_000, ok).name == "pilot_train"
    for r in range(3):  # W6 blocks are plain training blocks
        seed_blocks.require("train", 10_000_000 + r * 1_000_000, 12_000)


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
    states, nxt, rew = torch.randn(8, 4), torch.randn(8, 4), torch.randn(8)
    dones = torch.zeros(8)
    monitor.check_inputs(step, states, nxt, rew, dones)
    online_q = net(states)
    values = online_q.max(1).values
    with torch.inference_mode():
        target_q = target(nxt)
        targets = rew + (1 - dones) * target_q.max(1).values
    loss = nn.functional.smooth_l1_loss(values, targets.clone())
    monitor.check_q_outputs(step, online_q, target_q, targets)
    monitor.check_loss(step, loss)
    opt.zero_grad()
    loss.backward()
    monitor.check_grads(step, net)
    opt.step()
    monitor.check_after_step(step, net, opt)
    target.load_state_dict(net.state_dict())
    monitor.check_target_sync(step, target)


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
    _update(net, target, opt, mon, 0)
    bad = torch.tensor([1.0, float("nan")])
    ok = torch.ones(2)
    with pytest.raises(TrainingInvalid) as err:
        if quantity == "minibatch_input":
            mon.check_inputs(7, ok, ok, ok, bad)
        elif quantity == "online_q":
            mon.check_q_outputs(7, bad, ok, ok)
        elif quantity == "target_q":
            mon.check_q_outputs(7, ok, bad, ok)
        elif quantity == "loss":
            mon.check_loss(7, bad)
        elif quantity == "gradient":
            for p in net.parameters():
                p.grad = torch.ones_like(p)
            net.bias.grad[1] = float("inf")
            mon.check_grads(7, net)
        elif quantity == "parameters":
            with torch.no_grad():
                net.weight[0, 0] = float("nan")
            mon.check_after_step(7, net, opt)
        elif quantity in ("adam_exp_avg", "adam_exp_avg_sq"):
            key = "exp_avg" if quantity == "adam_exp_avg" else "exp_avg_sq"
            state = opt.state[dict(net.named_parameters())[label]]
            state[key].view(-1)[0] = float("inf")
            mon.check_after_step(7, net, opt)
        else:
            with torch.no_grad():
                target.weight[1, 1] = float("nan")
            mon.check_target_sync(7, target)
    assert err.value.quantity == quantity
    ff = mon.record("t")["first_failure"]
    assert mon.record("t")["status"] == "TRAINING_INVALID"
    assert ff["update"] == 7 and ff["quantity"] == quantity and ff["tensor"] == label
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
    tune = vtr1_reference.build_command(_ref_args(tune_kappa=True, seed_base=150_000, n=50))
    assert "--tune-kappa" in tune and "--tune-seed-base" in tune
    with pytest.raises(SystemExit):  # tuning outside validation block B / dev val
        vtr1_reference.build_command(_ref_args(tune_kappa=True, seed_base=170_000, n=50))


def test_eval_entry_point_plans_without_the_simulator(tmp_path):
    ckpt = tmp_path / "net.pt"
    ckpt.write_bytes(b"x")
    ns = vtr1_eval.parser().parse_args(
        ["--checkpoint", str(ckpt), "--cell", "dynamic_duopoly/random", "--role", "eval",
         "--seed-base", "30000", "--n", "10", "--out", str(tmp_path / "o.csv")]
    )
    block, overrides, (mode, order) = vtr1_eval.plan(ns)
    assert (block.name, mode, order, overrides["agent_order"]) == ("dev_test", "dynamic_duopoly", "random", "random")
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


def test_dqn_train_is_wired_to_scenario_role_and_monitor_hooks():
    src = (HERE / "dqn_train.py").read_text()
    assert "**scenario," in src and src.count("**scenario") >= 2  # training and validation overrides
    assert "seed_blocks.require(\n        args.train_role, args.train_seed_base, args.episodes, args.authorization" in src
    for hook in ["check_q_outputs", "check_loss", "check_grads", "check_after_step", "check_target_sync",
                 "eval_scope", "register_forward_hook"]:
        assert hook in src
    assert src.count("monitor.eval_scope()") == 2  # periodic and fallback validation
    assert "VTR-1 runs must pass --out-dir" in src

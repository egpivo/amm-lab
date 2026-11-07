"""Static / unit tests for the VTR-1 Gate 1 instrumentation. None runs the simulator or a DQN training."""

import random
import re
from pathlib import Path

import pytest
import torch
import torch.nn as nn

import seed_blocks
import seed_collision_audit
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


def test_protected_uses_need_an_authorization_file(tmp_path):
    with pytest.raises(SystemExit):
        seed_blocks.require("final", 190_000, 1_000)
    with pytest.raises(SystemExit):
        seed_blocks.require("pilot", 170_000, 200, authorization=str(tmp_path / "missing"))
    auth = tmp_path / "auth.txt"
    auth.write_text("x")
    assert seed_blocks.require("pilot", 170_000, 200, authorization=str(auth)).name == "vtr1_pilot"


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
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    return net, opt


def _update(net, opt, monitor, step=1):
    states, nxt, rew = torch.randn(8, 4), torch.randn(8, 4), torch.randn(8)
    monitor.check_inputs(step, states, nxt, rew)
    values = net(states).max(1).values
    targets = rew + net(nxt).max(1).values.detach()
    loss = nn.functional.smooth_l1_loss(values, targets)
    monitor.check_forward(step, values, targets, loss)
    opt.zero_grad()
    loss.backward()
    monitor.check_grads(step, net)
    opt.step()
    monitor.check_after_step(step, net, opt)


def test_checks_do_not_touch_rng_or_parameters():
    results = []
    for enabled in (False, True):
        net, opt = _tiny()
        random.seed(1)
        mon = ValidityMonitor(enabled)
        for step in range(5):
            _update(net, opt, mon, step)
        results.append(([p.clone() for p in net.parameters()], torch.get_rng_state(), random.getstate()))
    for a, b in zip(results[0][0], results[1][0]):
        assert torch.equal(a, b)
    assert torch.equal(results[0][1], results[1][1])
    assert results[0][2] == results[1][2]


@pytest.mark.parametrize("quantity", ["minibatch_input", "online_q", "target_q", "loss", "gradient", "parameters", "adam_exp_avg"])
def test_each_nonfinite_quantity_is_detected(quantity):
    net, opt = _tiny()
    mon = ValidityMonitor(True)
    _update(net, opt, mon, 0)
    bad = torch.tensor([float("nan")])
    ok = torch.ones(1)
    with pytest.raises(TrainingInvalid) as err:
        if quantity == "minibatch_input":
            mon.check_inputs(7, bad, ok, ok)
        elif quantity == "online_q":
            mon.check_forward(7, bad, ok, ok)
        elif quantity == "target_q":
            mon.check_forward(7, ok, bad, ok)
        elif quantity == "loss":
            mon.check_forward(7, ok, ok, bad)
        elif quantity == "gradient":
            for p in net.parameters():
                p.grad = torch.full_like(p, float("inf"))
            mon.check_grads(7, net)
        elif quantity == "parameters":
            with torch.no_grad():
                next(net.parameters()).fill_(float("nan"))
            mon.check_after_step(7, net, opt)
        else:
            for state in opt.state.values():
                state["exp_avg"].fill_(float("inf"))
            mon.check_after_step(7, net, opt)
    assert err.value.quantity == quantity
    assert mon.record("t")["status"] == "TRAINING_INVALID"
    assert mon.record("t")["first_failure"] == {"update": 7, "quantity": quantity}


def test_w0_harness_blocks_match_registry():
    src = (HERE.parents[1] / "src/bin/vtr1_w0.rs").read_text()
    db1 = re.search(r"const DB1: \(u64, u64\) = \(([\d_]+), ([\d_]+)\);", src)
    db2 = re.search(r"const DB2: \(u64, u64\) = \(([\d_]+), ([\d_]+)\);", src)
    by_name = {b.name: b for b in seed_blocks.BLOCKS}
    for m, name in ((db1, "w0_paths_DB1"), (db2, "w0_days_DB2")):
        assert (int(m[1].replace("_", "")), int(m[2].replace("_", ""))) == (by_name[name].start, by_name[name].end)
    assert "VTR1_W0_AUTHORIZED" in src  # refuses to run without the authorization file

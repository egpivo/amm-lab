"""Technical training-validity checks (VTR-1 contract §5.5). Pure tensor reads.

No RNG draw, no extra minibatch and no change to the update: every check is
`torch.isfinite` on tensors the update already holds. A failure marks the run
TRAINING_INVALID; the caller stops that run and downstream cells are NOT_ESTIMABLE.

Checked: minibatch states / next states / rewards / dones; the complete online and
target network outputs of the minibatch; the loss; gradients; online parameters and
Adam `exp_avg` / `exp_avg_sq` after every optimizer step; the target network after
every hard synchronization; every Q-value produced by the online network during
validation. (The optional fixed reference probe is deliberately not implemented.)
"""

import contextlib
import json
from pathlib import Path
from typing import Iterable

import torch

NamedTensors = Iterable[tuple[str, torch.Tensor | None]]


class TrainingInvalid(RuntimeError):
    def __init__(self, step: int, quantity: str, location: dict | None = None) -> None:
        super().__init__(f"non-finite {quantity} at update {step}: {location}")
        self.step, self.quantity, self.location = step, quantity, location or {}


class ValidityMonitor:
    def __init__(self, enabled: bool, context: dict | None = None) -> None:
        self.enabled = enabled
        self.context: dict = dict(context or {})
        self.updates_checked = 0
        self.counts: dict[str, dict[str, int]] = {}
        self.failure: dict | None = None
        self.phase = "train"

    # -- core -------------------------------------------------------------
    def _check(self, step: int, name: str, items: NamedTensors) -> None:
        entry = self.counts.setdefault(name, {"checked": 0, "nonfinite": 0})
        for label, t in items:
            if t is None:
                continue
            entry["checked"] += 1
            finite = torch.isfinite(t)
            if not bool(finite.all()):
                entry["nonfinite"] += 1
                flat = (~finite).flatten().nonzero()
                index = int(flat[0])
                self.failure = {
                    "update": step,
                    "quantity": name,
                    "tensor": label,
                    "flat_index": index,
                    "value": repr(t.flatten()[index].item()),
                    "n_nonfinite": int(flat.numel()),
                }
                raise TrainingInvalid(step, name, self.failure)

    # -- update-step checks -----------------------------------------------
    def check_inputs(self, step, states, next_states, rewards, dones) -> None:
        if self.enabled:
            self._check(
                step,
                "minibatch_input",
                (("states", states), ("next_states", next_states), ("rewards", rewards), ("dones", dones)),
            )

    def check_q_outputs(self, step, online_q, target_q, targets) -> None:
        """Complete network outputs on the minibatch (not only the selected actions)."""
        if self.enabled:
            self._check(step, "online_q", (("online_q", online_q),))
            self._check(step, "target_q", (("target_q", target_q), ("targets", targets)))

    def check_loss(self, step, loss) -> None:
        if self.enabled:
            self._check(step, "loss", (("loss", loss.detach()),))

    def check_grads(self, step, net) -> None:
        if self.enabled:
            self._check(step, "gradient", ((n, p.grad) for n, p in net.named_parameters()))

    def check_after_step(self, step, net, optimizer) -> None:
        if not self.enabled:
            return
        self._check(step, "parameters", ((n, p.detach()) for n, p in net.named_parameters()))
        for n, p in net.named_parameters():
            state = optimizer.state.get(p, {})
            self._check(step, "adam_exp_avg", ((n, state.get("exp_avg")),))
            self._check(step, "adam_exp_avg_sq", ((n, state.get("exp_avg_sq")),))
        self.updates_checked += 1

    def check_target_sync(self, step, target) -> None:
        if self.enabled:
            self._check(step, "target_network", ((n, p.detach()) for n, p in target.named_parameters()))

    # -- validation Q-values ----------------------------------------------
    def eval_forward_hook(self, module, inputs, output) -> None:
        if self.enabled and self.phase == "eval":
            self._check(int(self.context.get("episodes_completed", 0)), "validation_q", (("validation_q", output),))

    @contextlib.contextmanager
    def eval_scope(self):
        self.phase = "eval"
        try:
            yield
        finally:
            self.phase = "train"

    # -- record -----------------------------------------------------------
    def record(self, tag: str) -> dict:
        return {
            "tag": tag,
            "status": "TRAINING_INVALID" if self.failure else "VALID",
            "updates_checked": self.updates_checked,
            "check_counts": self.counts,
            "first_failure": self.failure,
            **self.context,
        }

    def write(self, path: Path, tag: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.record(tag), indent=2) + "\n")

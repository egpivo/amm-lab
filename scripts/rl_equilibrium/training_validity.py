"""Technical training-validity checks (VTR-1 contract §5.5). Pure tensor reads.

No RNG draw, no extra minibatch and no change to the update: every check is
`torch.isfinite` on tensors the update already holds. A failure marks the run
TRAINING_INVALID; the caller stops that run and downstream cells are NOT_ESTIMABLE.
"""

import json
from pathlib import Path
from typing import Iterable

import torch


class TrainingInvalid(RuntimeError):
    def __init__(self, step: int, quantity: str) -> None:
        super().__init__(f"non-finite {quantity} at update {step}")
        self.step, self.quantity = step, quantity


class ValidityMonitor:
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self.updates_checked = 0
        self.failure: tuple[int, str] | None = None

    @staticmethod
    def _finite(x: torch.Tensor) -> bool:
        return bool(torch.isfinite(x).all())

    def _require(self, step: int, name: str, tensors: Iterable[torch.Tensor]) -> None:
        for t in tensors:
            if t is not None and not self._finite(t):
                self.failure = (step, name)
                raise TrainingInvalid(step, name)

    def check_inputs(self, step, states, next_states, rewards) -> None:
        if self.enabled:
            self._require(step, "minibatch_input", (states, next_states, rewards))

    def check_forward(self, step, values, targets, loss) -> None:
        if self.enabled:
            self._require(step, "online_q", (values,))
            self._require(step, "target_q", (targets,))
            self._require(step, "loss", (loss.detach(),))

    def check_grads(self, step, net) -> None:
        if self.enabled:
            self._require(step, "gradient", (p.grad for p in net.parameters()))

    def check_after_step(self, step, net, optimizer) -> None:
        if not self.enabled:
            return
        self._require(step, "parameters", (p.detach() for p in net.parameters()))
        for p in net.parameters():
            state = optimizer.state.get(p, {})
            self._require(step, "adam_exp_avg", (state.get("exp_avg"),))
            self._require(step, "adam_exp_avg_sq", (state.get("exp_avg_sq"),))
        self.updates_checked += 1

    def record(self, tag: str) -> dict:
        return {
            "tag": tag,
            "status": "TRAINING_INVALID" if self.failure else "VALID",
            "updates_checked": self.updates_checked,
            "first_failure": None
            if self.failure is None
            else {"update": self.failure[0], "quantity": self.failure[1]},
        }

    def write(self, path: Path, tag: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.record(tag), indent=2) + "\n")

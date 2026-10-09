"""AdamW proposal and projection of its actual parameter decrement."""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import torch
from torch import nn


TensorList = list[torch.Tensor]


def zeros_for(parameters: list[nn.Parameter]) -> TensorList:
    return [torch.zeros_like(parameter) for parameter in parameters]


def gradients(
    loss: torch.Tensor, parameters: list[nn.Parameter], retain_graph: bool = False
) -> TensorList:
    raw = torch.autograd.grad(
        loss, parameters, retain_graph=retain_graph, allow_unused=True)
    return [torch.zeros_like(p) if g is None else g for p, g in zip(parameters, raw, strict=True)]


def list_dot(left: TensorList, right: TensorList) -> torch.Tensor:
    return sum((a.double() * b.double()).sum() for a, b in zip(left, right, strict=True))


def list_norm(values: TensorList) -> torch.Tensor:
    return list_dot(values, values).sqrt()


def clip_global_norm(values: TensorList, maximum: float) -> TensorList:
    norm = list_norm(values)
    factor = min(1.0, float(maximum / (norm + 1e-12)))
    return [value * factor for value in values]


def _combine(base: TensorList, rows: list[TensorList], coefficients: torch.Tensor) -> TensorList:
    result = [value.clone() for value in base]
    for row, coefficient in zip(rows, coefficients, strict=True):
        result = [a + coefficient.to(a) * b for a, b in zip(result, row, strict=True)]
    return result


@dataclass(frozen=True)
class ProjectionStats:
    active_constraints: int
    candidate_norm: float
    projected_norm: float
    changed_fraction: float
    fallback_zero: bool


def project_two_halfspaces(
    candidate: TensorList,
    gradient_a: TensorList,
    gradient_f: TensorList,
    tolerance: float = 1e-10,
) -> tuple[TensorList, ProjectionStats]:
    """Euclidean projection in FP64 onto g_A.d>=0 and g_F.d>=0."""
    # The constraints are intentionally solved in FP64.  Keeping only the dot
    # products in FP64 while casting normalized rows and Lagrange multipliers
    # back to FP32 can reject a tangent solution and spuriously select the
    # always-feasible zero fallback.
    original_dtypes = [value.dtype for value in candidate]
    candidate64 = [value.double() for value in candidate]
    rows = []
    for gradient in (gradient_a, gradient_f):
        gradient64 = [value.double() for value in gradient]
        norm = list_norm(gradient64)
        if float(norm) > 1e-14:
            rows.append([value / norm for value in gradient64])
    candidate_norm = float(list_norm(candidate64))
    if not rows:
        return candidate, ProjectionStats(0, candidate_norm, candidate_norm, 0, False)
    feasible: list[tuple[TensorList, int, bool]] = []
    for active_count in range(len(rows) + 1):
        for active_indices in itertools.combinations(range(len(rows)), active_count):
            if not active_indices:
                proposal = [x.clone() for x in candidate64]
            else:
                active = [rows[index] for index in active_indices]
                gram = torch.stack([
                    torch.stack([list_dot(a, b) for b in active]) for a in active])
                rhs = -torch.stack([list_dot(row, candidate64) for row in active])
                multipliers = torch.linalg.pinv(gram) @ rhs
                if (multipliers < -tolerance).any():
                    continue
                proposal = _combine(candidate64, active, multipliers)
            if all(float(list_dot(row, proposal)) >= -tolerance for row in rows):
                feasible.append((proposal, active_count, False))
    zero = [torch.zeros_like(value) for value in candidate64]
    feasible.append((zero, len(rows), True))
    selected, active, fallback = min(
        feasible, key=lambda item: float(list_dot(
            [a - b for a, b in zip(item[0], candidate64, strict=True)],
            [a - b for a, b in zip(item[0], candidate64, strict=True)])))
    projected_norm = float(list_norm(selected))
    change = list_norm([a - b for a, b in zip(selected, candidate64, strict=True)])
    cast_selected = [
        value.to(dtype=dtype)
        for value, dtype in zip(selected, original_dtypes, strict=True)]
    return cast_selected, ProjectionStats(
        active, candidate_norm, projected_norm,
        float(change) / max(candidate_norm, 1e-12), fallback)


class ActualStepAdamW:
    """AdamW whose proposal/state commit/parameter application are separate."""

    def __init__(
        self,
        named_parameters: list[tuple[str, nn.Parameter]],
        lr: float = 1e-4,
        unet_lr: float = 1e-5,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-4,
    ) -> None:
        self.named_parameters = [(n, p) for n, p in named_parameters if p.requires_grad]
        self.parameters = [p for _, p in self.named_parameters]
        self.lr, self.unet_lr = float(lr), float(unet_lr)
        self.betas, self.eps, self.weight_decay = betas, float(eps), float(weight_decay)
        self.state: dict[int, dict[str, torch.Tensor | int]] = {}

    def _group(self, name: str, parameter: nn.Parameter) -> tuple[float, float]:
        learning_rate = self.unet_lr if "encoder.map_unet" in name else self.lr
        decay = 0.0 if parameter.ndim <= 1 or name.endswith("bias") or "norm" in name.lower() else self.weight_decay
        return learning_rate, decay

    def propose(self, gradient: TensorList) -> tuple[TensorList, list[dict[str, torch.Tensor | int]]]:
        if len(gradient) != len(self.parameters):
            raise ValueError("gradient list does not match optimizer parameters")
        beta1, beta2 = self.betas
        decrement, pending = [], []
        for (name, parameter), grad in zip(self.named_parameters, gradient, strict=True):
            old = self.state.get(id(parameter))
            moment = torch.zeros_like(parameter, dtype=torch.float32) if old is None else old["moment"]
            variance = torch.zeros_like(parameter, dtype=torch.float32) if old is None else old["variance"]
            step = 1 if old is None else int(old["step"]) + 1
            moment_new = beta1 * moment + (1 - beta1) * grad.float()
            variance_new = beta2 * variance + (1 - beta2) * grad.float().square()
            mhat = moment_new / (1 - beta1 ** step)
            vhat = variance_new / (1 - beta2 ** step)
            learning_rate, decay = self._group(name, parameter)
            update = learning_rate * (
                mhat / (vhat.sqrt() + self.eps) + decay * parameter.detach().float())
            decrement.append(update.to(parameter.dtype))
            pending.append({"moment": moment_new, "variance": variance_new, "step": step})
        return decrement, pending

    def apply(
        self, decrement: TensorList, pending: list[dict[str, torch.Tensor | int]]
    ) -> None:
        with torch.no_grad():
            for parameter, update, state in zip(
                self.parameters, decrement, pending, strict=True):
                parameter.sub_(update)
                self.state[id(parameter)] = state

    def state_dict(self) -> dict:
        states = []
        for parameter in self.parameters:
            state = self.state.get(id(parameter), {})
            states.append({
                key: value.detach().cpu() if torch.is_tensor(value) else value
                for key, value in state.items()})
        return {
            "states": states,
            "parameter_names": [name for name, _ in self.named_parameters],
            "lr": self.lr,
            "unet_lr": self.unet_lr,
        }

    def load_state_dict(self, payload: dict) -> None:
        if "lr" not in payload or "unet_lr" not in payload:
            raise ValueError("optimizer checkpoint is missing learning rates")
        learning_rate = float(payload["lr"])
        unet_learning_rate = float(payload["unet_lr"])
        if not torch.isfinite(torch.tensor([learning_rate, unet_learning_rate])).all():
            raise ValueError("optimizer checkpoint learning rates must be finite")
        expected_names = [name for name, _ in self.named_parameters]
        saved_names = payload.get("parameter_names")
        if saved_names is not None and list(saved_names) != expected_names:
            raise ValueError("optimizer parameter names/order differ from checkpoint")
        states = payload.get("states")
        if not isinstance(states, list) or len(states) != len(self.parameters):
            raise ValueError("optimizer state count differs from parameter count")
        self.state.clear()
        for parameter, state in zip(self.parameters, states, strict=True):
            self.state[id(parameter)] = {
                key: value.to(parameter.device) if torch.is_tensor(value) else value
                for key, value in state.items()}
        self.lr = learning_rate
        self.unet_lr = unet_learning_rate

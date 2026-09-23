"""Exact one-solve refinement with persistent exchangeable tie semantics.

This production helper lifts canonical FP32 score and world-goal snapshots to
Python arbitrary-precision integers and solves the strict objective
``(J, C_stay, C_geom, R)`` in one rectangular assignment solve.  ``R`` is a
persistent, exchangeable fourth-level symmetry priority; it is never added to
the learned conditional score.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
import random
from typing import Optional, Sequence

import numpy as np
import torch


TIE_NAMESPACE = 0x524553494455414C


@dataclass(frozen=True)
class ExactPersistentAssignment:
    assignment: tuple[int, ...]
    objective: tuple[int, int, int, int]
    encoded_objective: int
    priorities: tuple[tuple[int, ...], ...]
    valid_edge_count: int


@dataclass(frozen=True)
class IntegerLift:
    values: tuple
    unit_numerator: int
    unit_denominator: int
    shape: tuple[int, ...]

    @property
    def unit(self) -> Fraction:
        return Fraction(self.unit_numerator, self.unit_denominator)


def stable_tie_seed(evaluation_seed: int, window_index: int,
                    agent_index: int) -> int:
    """Match approved audit replica-zero seed mixing without ``hash()``."""
    mask = (1 << 64) - 1
    value = TIE_NAMESPACE & mask
    for item in (evaluation_seed, window_index, 0, agent_index):
        value ^= (int(item) + 0x9E3779B97F4A7C15) & mask
        value = (value + 0x9E3779B97F4A7C15) & mask
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
        value ^= value >> 31
    return value & mask


def _freeze_nested(value):
    if isinstance(value, list):
        return tuple(_freeze_nested(item) for item in value)
    return int(value)


def _power_two_fraction(exponent: int) -> Fraction:
    if exponent >= 0:
        return Fraction(1 << exponent, 1)
    return Fraction(1, 1 << (-exponent))


def lift_fp32_to_integers(tensor: torch.Tensor) -> IntegerLift:
    """Losslessly lift the stored finite IEEE-754 FP32 values."""
    value = tensor.detach().to(
        device="cpu", dtype=torch.float32).contiguous()
    array = value.numpy()
    if not np.isfinite(array).all():
        raise ValueError("exact lift requires finite FP32 values")
    bits = array.view(np.uint32).reshape(-1)
    terms = []
    minimum_exponent = None
    for raw_value in bits.tolist():
        raw = int(raw_value)
        sign = -1 if raw >> 31 else 1
        exponent_bits = (raw >> 23) & 0xFF
        fraction_bits = raw & 0x7FFFFF
        if exponent_bits == 0:
            if fraction_bits == 0:
                terms.append((0, 0))
                continue
            mantissa = fraction_bits
            exponent = -149
        else:
            mantissa = (1 << 23) | fraction_bits
            exponent = exponent_bits - 150
        terms.append((sign * mantissa, exponent))
        minimum_exponent = exponent if minimum_exponent is None else min(
            minimum_exponent, exponent)
    if minimum_exponent is None:
        integers = [0] * len(terms)
        unit = Fraction(1, 1)
    else:
        integers = [
            0 if mantissa == 0 else mantissa << (
                exponent - minimum_exponent)
            for mantissa, exponent in terms
        ]
        divisor = 0
        for item in integers:
            divisor = math.gcd(divisor, abs(item))
        divisor = max(divisor, 1)
        integers = [item // divisor for item in integers]
        unit = Fraction(divisor, 1) * _power_two_fraction(minimum_exponent)
    nested = np.asarray(
        integers, dtype=object).reshape(array.shape).tolist()
    return IntegerLift(
        values=_freeze_nested(nested),
        unit_numerator=unit.numerator,
        unit_denominator=unit.denominator,
        shape=tuple(array.shape),
    )


def _normalize_integer_matrix(matrix):
    divisor = 0
    for row in matrix:
        for item in row:
            divisor = math.gcd(divisor, abs(int(item)))
    divisor = max(divisor, 1)
    return tuple(tuple(int(item) // divisor for item in row)
                 for row in matrix), divisor


def exact_negative_squared_geometry(goal_candidates: torch.Tensor,
                                    previous_ids: torch.Tensor):
    """Compute exact negative squared endpoint movement in world metres."""
    goals = lift_fp32_to_integers(goal_candidates)
    if len(goals.shape) != 2 or goals.shape[1] != 2:
        raise ValueError("goal_candidates must have shape [K,2]")
    previous = previous_ids.detach().to(device="cpu", dtype=torch.long)
    if previous.ndim != 1:
        raise ValueError("previous_ids must have shape [P]")
    coordinates = goals.values
    if any(item < 0 or item >= len(coordinates)
           for item in previous.tolist()):
        raise ValueError("previous candidate ID is invalid")
    matrix = []
    for old in previous.tolist():
        old_x, old_y = coordinates[old]
        matrix.append(tuple(
            -((x - old_x) * (x - old_x) +
              (y - old_y) * (y - old_y))
            for x, y in coordinates))
    normalized, divisor = _normalize_integer_matrix(matrix)
    return normalized, goals.unit * goals.unit * divisor


def _validate_problem(weight, mask):
    rows = len(weight)
    columns = len(weight[0]) if rows else 0
    if rows <= 0 or columns <= 0:
        raise ValueError("assignment matrix must be non-empty")
    if rows > columns:
        raise ValueError("injective assignment requires P<=K")
    if len(mask) != rows or any(len(row) != columns for row in mask):
        raise ValueError("mask shape differs from assignment matrix")
    if any(len(row) != columns for row in weight):
        raise ValueError("ragged assignment matrix")
    if any(sum(bool(item) for item in row) < 1 for row in mask):
        raise ValueError("every slot requires a valid candidate")
    valid_columns = {
        column for column in range(columns)
        if any(bool(mask[row][column]) for row in range(rows))
    }
    if len(valid_columns) < rows:
        raise ValueError("valid_K<P for injective assignment")
    return rows, columns


def exact_max_weight_assignment(weight, mask):
    """Rectangular Hungarian solver using exact Python integer arithmetic."""
    rows, columns = _validate_problem(weight, mask)
    u = [0] * (rows + 1)
    v = [0] * (columns + 1)
    p = [0] * (columns + 1)
    way = [0] * (columns + 1)
    for incoming in range(1, rows + 1):
        p[0] = incoming
        minimum = [None] * (columns + 1)
        used = [False] * (columns + 1)
        current_column = 0
        while True:
            used[current_column] = True
            current_row = p[current_column]
            delta = None
            next_column = -1
            source_row = current_row - 1
            for column in range(1, columns + 1):
                if used[column]:
                    continue
                source_column = column - 1
                if not mask[source_row][source_column]:
                    continue
                cost = -int(weight[source_row][source_column])
                reduced = cost - u[current_row] - v[column]
                if minimum[column] is None or reduced < minimum[column]:
                    minimum[column] = reduced
                    way[column] = current_column
                if minimum[column] is not None and (
                        delta is None or minimum[column] < delta):
                    delta = minimum[column]
                    next_column = column
            if delta is None:
                raise ValueError("assignment is infeasible")
            for column in range(columns + 1):
                if used[column]:
                    u[p[column]] += delta
                    v[column] -= delta
                elif minimum[column] is not None:
                    minimum[column] -= delta
            current_column = next_column
            if p[current_column] == 0:
                break
        while True:
            previous_column = way[current_column]
            p[current_column] = p[previous_column]
            current_column = previous_column
            if current_column == 0:
                break
    assignment = [-1] * rows
    for column in range(1, columns + 1):
        if p[column]:
            assignment[p[column] - 1] = column - 1
    if any(column < 0 for column in assignment):
        raise ValueError("assignment solver returned an incomplete matching")
    objective = sum(weight[row][column]
                    for row, column in enumerate(assignment))
    return tuple(assignment), int(objective)


def exchangeable_edge_priorities(mask: torch.Tensor, seed: int):
    """Assign distinct powers of two to valid edges using a local RNG."""
    mask = mask.detach().to(device="cpu", dtype=torch.bool)
    if mask.ndim != 2:
        raise ValueError("mask must have shape [P,K]")
    edges = [(row, column) for row in range(mask.shape[0])
             for column in range(mask.shape[1]) if bool(mask[row, column])]
    shuffled = list(edges)
    random.Random(int(seed)).shuffle(shuffled)
    priorities = [[0 for _ in range(mask.shape[1])]
                  for _ in range(mask.shape[0])]
    for rank, (row, column) in enumerate(shuffled):
        priorities[row][column] = 1 << rank
    return tuple(tuple(row) for row in priorities), len(edges)


def _validate_priorities(priorities, mask, rows, columns):
    priorities = tuple(tuple(int(item) for item in row)
                       for row in priorities)
    if len(priorities) != rows or any(
            len(row) != columns for row in priorities):
        raise ValueError("priority shape differs from score")
    valid = [priorities[row][column]
             for row in range(rows) for column in range(columns)
             if mask[row][column]]
    if len(set(valid)) != len(valid) or any(
            item <= 0 or item & (item - 1) for item in valid):
        raise ValueError("valid priorities must be distinct powers of two")
    return priorities, len(valid)


def _exact_objective(assignment, primary, stay, geometry, priorities):
    return tuple(sum(matrix[row][column]
                     for row, column in enumerate(assignment))
                 for matrix in (primary, stay, geometry, priorities))


def solve_exact_persistent_tie(
    score: torch.Tensor,
    mask: torch.Tensor,
    previous_ids: torch.Tensor,
    goal_candidates_world: torch.Tensor,
    *,
    evaluation_seed: Optional[int] = None,
    window_index: Optional[int] = None,
    agent_index: Optional[int] = None,
    priorities: Optional[Sequence[Sequence[int]]] = None,
) -> ExactPersistentAssignment:
    """Solve strict ``(J,C_stay,C_geom,R)`` with exactly one assignment call."""
    score = score.detach().to(device="cpu", dtype=torch.float32).contiguous()
    mask_tensor = mask.detach().to(device="cpu", dtype=torch.bool).contiguous()
    previous = previous_ids.detach().to(device="cpu", dtype=torch.long)
    if score.ndim != 2 or mask_tensor.shape != score.shape:
        raise ValueError("score/mask must share shape [P,K]")
    rows, columns = score.shape
    if previous.shape != (rows,):
        raise ValueError("previous_ids must have shape [P]")
    if goal_candidates_world.shape != (columns, 2):
        raise ValueError("goal_candidates_world must have shape [K,2]")
    if not torch.equal(mask_tensor, mask_tensor[:1].expand_as(mask_tensor)):
        raise ValueError("exact policy requires a slot-invariant mask")
    if int(mask_tensor[0].sum()) < rows:
        raise ValueError("exact policy requires valid_K>=P")
    if torch.unique(previous).numel() != rows:
        raise ValueError("previous_ids must be injective")
    if any(candidate < 0 or candidate >= columns
           for candidate in previous.tolist()):
        raise ValueError("previous candidate ID is invalid")
    if any(not bool(mask_tensor[row, int(candidate)])
           for row, candidate in enumerate(previous.tolist())):
        raise ValueError("previous assignment contains an invalid candidate")

    primary = lift_fp32_to_integers(score).values
    boolean_mask = tuple(tuple(bool(item) for item in row)
                         for row in mask_tensor.tolist())
    stay = tuple(tuple(int(column == int(previous[row]))
                       for column in range(columns)) for row in range(rows))
    geometry, _ = exact_negative_squared_geometry(
        goal_candidates_world, previous)
    if priorities is None:
        if evaluation_seed is None or window_index is None or \
                agent_index is None:
            raise ValueError("production tie key is incomplete")
        priorities, valid_edge_count = exchangeable_edge_priorities(
            mask_tensor, stable_tie_seed(
                evaluation_seed, window_index, agent_index))
    else:
        priorities, valid_edge_count = _validate_priorities(
            priorities, boolean_mask, rows, columns)

    geometry_valid = [geometry[row][column]
                      for row in range(rows) for column in range(columns)
                      if boolean_mask[row][column]]
    geometry_min = min(geometry_valid)
    geometry_max = max(geometry_valid)
    geometry_span = rows * (geometry_max - geometry_min)
    r_base = 1 << valid_edge_count
    geometry_base = (geometry_span + 1) * r_base
    primary_base = (rows + 1) * geometry_base
    encoded = tuple(tuple(
        int(primary[row][column]) * primary_base +
        int(stay[row][column]) * geometry_base +
        (int(geometry[row][column]) - geometry_min) * r_base +
        int(priorities[row][column])
        for column in range(columns)) for row in range(rows))
    assignment, encoded_objective = exact_max_weight_assignment(
        encoded, boolean_mask)
    return ExactPersistentAssignment(
        assignment=assignment,
        objective=_exact_objective(
            assignment, primary, stay, geometry, priorities),
        encoded_objective=encoded_objective,
        priorities=priorities,
        valid_edge_count=valid_edge_count,
    )


def exact_persistent_refinement(
    score: torch.Tensor,
    mask: torch.Tensor,
    previous_ids: torch.Tensor,
    goal_candidates_world: torch.Tensor,
    degree: torch.Tensor,
    *,
    evaluation_seed: int,
    window_index: int,
) -> torch.Tensor:
    """Apply structural identity or the one-solve exact policy per agent."""
    if score.ndim != 3 or mask.shape != score.shape:
        raise ValueError("score/mask must have shape [N,P,K]")
    agents, slots, candidates = score.shape
    if previous_ids.shape != (agents, slots):
        raise ValueError("previous_ids must have shape [N,P]")
    if goal_candidates_world.shape != (agents, candidates, 2):
        raise ValueError("goal_candidates_world must have shape [N,K,2]")
    if degree.shape != (agents,):
        raise ValueError("degree must have shape [N]")
    selected = previous_ids.clone()
    for agent in range(agents):
        if int(degree[agent]) == 0:
            continue
        solved = solve_exact_persistent_tie(
            score[agent], mask[agent], previous_ids[agent],
            goal_candidates_world[agent], evaluation_seed=evaluation_seed,
            window_index=window_index, agent_index=agent)
        selected[agent] = torch.tensor(
            solved.assignment, dtype=torch.long, device=selected.device)
    return selected


__all__ = [
    "ExactPersistentAssignment",
    "TIE_NAMESPACE",
    "exact_max_weight_assignment",
    "exact_negative_squared_geometry",
    "exact_persistent_refinement",
    "exchangeable_edge_priorities",
    "lift_fp32_to_integers",
    "solve_exact_persistent_tie",
    "stable_tie_seed",
]

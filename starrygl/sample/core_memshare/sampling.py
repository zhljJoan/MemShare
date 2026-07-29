"""Prediction-corrected, communication-aware sampling primitives."""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass(frozen=True)
class ResidualSamplingConfig:
    """Configuration shared by the Python and native residual samplers."""

    min_probability: float = 0.05
    max_probability: float = 1.0
    target_probability: float = 0.1
    uncertainty_weight: float = 1.0
    age_weight: float = 0.25
    edge_error_weight: float = 0.5
    age_scale: float = 1.0
    memory_cost: float = 1.0
    edge_cost: float = 1.0
    request_cost: float = 0.05
    cached_edge_cost: float = 0.0
    group_size: int = 8
    seed: int = 6773

    def __post_init__(self) -> None:
        if not 0.0 < self.min_probability <= self.max_probability <= 1.0:
            raise ValueError("sampling probabilities must satisfy 0 < min <= max <= 1")
        if not 0.0 < self.target_probability <= 1.0:
            raise ValueError("target_probability must be in (0, 1]")
        if self.age_scale <= 0:
            raise ValueError("age_scale must be positive")
        if self.group_size <= 0:
            raise ValueError("group_size must be positive")


def _as_float_tensor(value: torch.Tensor) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError("expected a torch.Tensor")
    return value if value.is_floating_point() else value.float()


def communication_aware_probabilities(
    residual: torch.Tensor,
    cost: torch.Tensor,
    *,
    budget: Optional[float] = None,
    target_probability: Optional[float] = None,
    min_probability: float = 0.0,
    max_probability: float = 1.0,
    iterations: int = 48,
) -> torch.Tensor:
    r"""Return the clipped solution :math:`p_i=\lambda |r_i|/\sqrt{c_i}`.

    ``budget`` is an expected communication budget, i.e.
    :math:`\sum_i p_i c_i`.  If only ``target_probability`` is supplied, the
    equivalent budget is that fraction of the cost of fetching every object.
    A monotone binary search handles both lower and upper probability clipping.
    """

    residual = _as_float_tensor(residual)
    cost = _as_float_tensor(cost).to(device=residual.device)
    if residual.shape != cost.shape:
        raise ValueError("residual and cost must have the same shape")
    if residual.numel() == 0:
        return residual.clone()
    if torch.any(cost <= 0):
        raise ValueError("communication costs must be positive")
    if not 0.0 <= min_probability <= max_probability <= 1.0:
        raise ValueError("probability bounds must satisfy 0 <= min <= max <= 1")
    if budget is None:
        if target_probability is None:
            raise ValueError("either budget or target_probability is required")
        if not 0.0 <= target_probability <= 1.0:
            raise ValueError("target_probability must be in [0, 1]")
        budget = float(target_probability) * float(cost.sum().item())
    if budget < 0:
        raise ValueError("budget must be non-negative")

    lower = torch.full_like(residual, min_probability)
    upper = torch.full_like(residual, max_probability)
    min_budget = float((lower * cost).sum().item())
    max_budget = float((upper * cost).sum().item())
    if budget <= min_budget:
        return lower
    if budget >= max_budget:
        return upper

    utility = residual.abs() / torch.sqrt(cost)
    # Zero residuals retain exploration probability and do not affect scaling.
    positive = utility > 0
    if not bool(positive.any()):
        return lower

    lo = 0.0
    hi = 1.0

    def expected_cost(scale: float) -> float:
        probability = torch.clamp(utility * scale, min_probability, max_probability)
        return float((probability * cost).sum().item())

    while expected_cost(hi) < budget:
        hi *= 2.0
        if hi > 1e30:
            break
    for _ in range(iterations):
        mid = (lo + hi) * 0.5
        if expected_cost(mid) < budget:
            lo = mid
        else:
            hi = mid
    return torch.clamp(utility * ((lo + hi) * 0.5), min_probability, max_probability)


def _splitmix64(value: int) -> int:
    mask = (1 << 64) - 1
    value = (value + 0x9E3779B97F4A7C15) & mask
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
    return (value ^ (value >> 31)) & mask


def _keyed_uniform(object_id: int, group_id: int, seed: int, batch_id: int) -> float:
    mask = (1 << 64) - 1
    key = int(seed) & mask
    key ^= _splitmix64(int(object_id) & mask)
    key ^= _splitmix64((int(group_id) + 0xD1B54A32D192ED03) & mask)
    key ^= _splitmix64((int(batch_id) + 0x94D049BB133111EB) & mask)
    # Use the high 53 bits so conversion to double is exact.
    return ((_splitmix64(key) >> 11) & ((1 << 53) - 1)) / float(1 << 53)


def coordinated_bernoulli(
    object_ids: torch.Tensor,
    probabilities: torch.Tensor,
    root_groups: torch.Tensor,
    *,
    seed: int,
    batch_id: int = 0,
) -> torch.Tensor:
    """Sample Bernoulli decisions with one keyed uniform per object/group.

    Occurrences of the same object in the same group share a random number.
    Each occurrence still has marginal selection probability ``probabilities``.
    """

    if object_ids.shape != probabilities.shape or object_ids.shape != root_groups.shape:
        raise ValueError("object_ids, probabilities, and root_groups must align")
    if torch.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("probabilities must be in [0, 1]")
    original_device = probabilities.device
    ids = object_ids.detach().reshape(-1).cpu().tolist()
    groups = root_groups.detach().reshape(-1).cpu().tolist()
    probs = probabilities.detach().reshape(-1).cpu().tolist()
    uniforms = {}
    decisions = []
    for object_id, group_id, probability in zip(ids, groups, probs):
        key = (int(object_id), int(group_id))
        uniform = uniforms.get(key)
        if uniform is None:
            uniform = _keyed_uniform(key[0], key[1], seed, batch_id)
            uniforms[key] = uniform
        decisions.append(uniform < float(probability))
    return torch.tensor(decisions, dtype=torch.bool, device=original_device).reshape(
        probabilities.shape
    )


def _expand_mask(mask: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    while mask.ndim < target.ndim:
        mask = mask.unsqueeze(-1)
    return mask


def residual_correct(
    prediction: torch.Tensor,
    actual: torch.Tensor,
    selected: torch.Tensor,
    probability: torch.Tensor,
) -> torch.Tensor:
    r"""Apply :math:`\bar y + z(y-\bar y)/p` elementwise."""

    if prediction.shape != actual.shape:
        raise ValueError("prediction and actual must have the same shape")
    if selected.shape != probability.shape:
        raise ValueError("selected and probability must have the same shape")
    if prediction.shape[: selected.ndim] != selected.shape:
        raise ValueError("selection shape must prefix the contribution shape")
    if torch.any(probability <= 0):
        raise ValueError("probability must be strictly positive")
    selected_f = _expand_mask(selected.to(actual.dtype), actual)
    probability_f = _expand_mask(probability.to(actual.dtype), actual)
    return prediction + selected_f * (actual - prediction) / probability_f


def residual_ratio_aggregate(
    actual_numerator: torch.Tensor,
    actual_denominator: torch.Tensor,
    predicted_numerator: torch.Tensor,
    predicted_denominator: torch.Tensor,
    selected: torch.Tensor,
    probability: torch.Tensor,
    destination: torch.Tensor,
    num_destinations: int,
    *,
    denominator_floor: float = 1e-12,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Correct attention numerator and denominator before taking their ratio."""

    corrected_numerator = residual_correct(
        predicted_numerator, actual_numerator, selected, probability
    )
    corrected_denominator = residual_correct(
        predicted_denominator, actual_denominator, selected, probability
    )
    numerator = torch.zeros(
        (num_destinations,) + corrected_numerator.shape[1:],
        dtype=corrected_numerator.dtype,
        device=corrected_numerator.device,
    )
    denominator = torch.zeros(
        (num_destinations,) + corrected_denominator.shape[1:],
        dtype=corrected_denominator.dtype,
        device=corrected_denominator.device,
    )
    numerator.index_add_(0, destination, corrected_numerator)
    denominator.index_add_(0, destination, corrected_denominator)
    ratio = numerator / denominator.clamp_min(denominator_floor)
    return ratio, numerator, denominator


def effective_sample_size(weight: torch.Tensor) -> float:
    """Return the usual importance-sampling effective sample size."""

    weight = _as_float_tensor(weight).reshape(-1)
    if weight.numel() == 0:
        return 0.0
    denominator = torch.square(weight).sum()
    if float(denominator.item()) == 0.0:
        return 0.0
    return float((weight.sum().square() / denominator).item())

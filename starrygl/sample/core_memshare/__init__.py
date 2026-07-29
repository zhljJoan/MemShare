"""CoRe-MemShare algorithm and runtime building blocks.

The package is intentionally independent from graph partitioning.  It contains
the state and estimators that change during training while node/edge ownership
remains fixed.
"""

from .cache import CacheLookup, ProbabilityAwareEdgeCache
from .negative import (
    CommunicationAwareNegativeSampling,
    importance_weighted_bce_with_logits,
)
from .planner import RequestPlan, UnifiedRequestPlanner
from .sampling import (
    ResidualSamplingConfig,
    communication_aware_probabilities,
    coordinated_bernoulli,
    effective_sample_size,
    residual_correct,
    residual_ratio_aggregate,
)
from .state import LastFetchedMemoryPredictor, PredictionBatch

__all__ = [
    "CacheLookup",
    "CommunicationAwareNegativeSampling",
    "LastFetchedMemoryPredictor",
    "PredictionBatch",
    "ProbabilityAwareEdgeCache",
    "RequestPlan",
    "ResidualSamplingConfig",
    "UnifiedRequestPlanner",
    "communication_aware_probabilities",
    "coordinated_bernoulli",
    "effective_sample_size",
    "importance_weighted_bce_with_logits",
    "residual_correct",
    "residual_ratio_aggregate",
]

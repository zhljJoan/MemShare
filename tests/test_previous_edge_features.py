import pytest
import torch

from starrygl.sample.previous_edge_features import PreviousEdgeFeatures


@pytest.mark.parametrize("sorted_ids", [True, False])
def test_previous_batch_preserves_feature_order(sorted_ids):
    cache = PreviousEdgeFeatures(sorted_ids=sorted_ids)
    table = torch.arange(27).view(9, 3)
    for sequence in ([4, 1, 7], [7, 4, 1], [2, 7, 2], [], [1], [5]):
        ids = torch.tensor(sequence, dtype=torch.long)
        if sorted_ids:
            ids = ids.sort().values
        missing, plan = cache.lookup(ids)
        actual = cache.complete(ids, table[missing], plan)
        assert torch.equal(actual, table[ids])

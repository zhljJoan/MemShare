"""One completed batch of immutable edge features, keyed by sorted distributed EID."""
import torch


class PreviousEdgeFeatures:
    def __init__(self, sorted_ids=True):
        self.sorted_ids = sorted_ids
        self.ids = None
        self.features = None

    def lookup(self, ids):
        if self.ids is None or not self.ids.numel():
            return ids, None
        pos = torch.searchsorted(self.ids, ids).clamp_max(self.ids.numel() - 1)
        hit = self.ids[pos] == ids
        missing = (~hit).nonzero().flatten()
        hits = hit.nonzero().flatten()
        return ids[missing], (missing, hits, pos[hits], self.features)

    def complete(self, ids, received, plan):
        if plan is not None:
            missing, hits, old_positions, old = plan
            features = received.new_empty((len(ids), *received.shape[1:]))
            features[missing] = received
            features[hits] = old[old_positions]
        else:
            features = received
        if self.sorted_ids:
            self.ids, self.features = ids, features
        else:
            self.ids, order = ids.sort()
            self.features = features[order]
        return features

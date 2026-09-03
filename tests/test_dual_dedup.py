import torch

from starrygl.sample import dual_dedup_view


class _Block:
    def __init__(self):
        self.srcdata = {}
        self.edata = {}


def test_dual_dedup_view_records_reuse(monkeypatch):
    monkeypatch.setenv("MEMSHARE_DUAL_DEDUP_VIEW", "on")
    dual_dedup_view.reset()
    block = _Block()
    block.srcdata["__ID"] = torch.tensor([0, 1, 0, 2])
    block.edata["__ID"] = torch.tensor([0, 1, 0])
    dual_dedup_view.attach(
        block,
        torch.randn(3, 5),
        torch.randn(2, 4),
        (
            torch.randn(3, 6),
            torch.randn(3),
            torch.randn(3, 1, 7),
            torch.randn(3, 1),
        ),
    )

    stats = dual_dedup_view.stats()
    assert stats["calls"] == 1
    assert stats["state_per_object_node"] == 4 / 3
    assert stats["edge_occurrence_per_object_edge"] == 3 / 2
    assert torch.equal(
        block._memshare_dual_dedup_view["state_view"]["state_to_uid"],
        block.srcdata["__ID"],
    )

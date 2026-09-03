import torch

from starrygl.sample import direct_segment_block


def test_segment_plan_preserves_original_edge_ids():
    row = torch.tensor([4, 2, 3, 5, 1, 0, 4, 2])
    col = torch.tensor([1, 0, 1, 2, 0, 1, 0, 2])
    block = direct_segment_block.create_block(row, col, 6, 4)
    dst_ptr, edge_ids = block._memshare_segment_plan

    assert torch.equal(dst_ptr, torch.tensor([0, 3, 6, 8, 8]))
    assert torch.equal(torch.sort(edge_ids).values, torch.arange(row.numel()))
    assert torch.equal(block._memshare_edge_src, row)
    assert torch.equal(block.edges(order="eid")[0], row)
    assert torch.equal(block.edges(order="eid")[1], col)

"""Reduce initialization footprint identically in both timed arms."""

from pathlib import Path
import sys

p = Path(sys.argv[1]) / "starrygl/sample/part_utils/transformer_from_speed.py"
s = p.read_text()
a = s.index("    df = load_graph(data)")
b = s.index("    dim_feats =", a)
old = s[a:b]
old = "".join(l for l in old.splitlines(True) if "print('tot edge" not in l)
s = (
    s[:a]
    + """    cache = '/mnt/nfs/fzz/TGL-DATA/{}/edges_cache/'.format(data)
    if data in ('GDELT','StackOverflow') and os.path.isfile(cache+'src.npy'):
        src, dst, ts, roll = (torch.from_numpy(np.load(cache+k+'.npy',mmap_mode='c')) for k in ('src','dst','time','ext_roll'))
        num_node=max(src.max().item(),dst.max().item())+1
        train_mask,val_mask,test_mask=roll==0,roll==1,roll==2
    else:
"""
    + "".join("    " + l for l in old.splitlines(True))
    + s[b:]
)
s = s.replace(
    "        _node_feats, _edge_feats = load_feat(data, num_node, src.shape[0], 172, 172)",
    """        if data == 'GDELT':
            node_file = '/mnt/nfs/fzz/TGL-DATA/GDELT/node_features.pt'
            _edge_feats = None
            if os.path.exists(node_file):
                _node_feats = torch.load(node_file)
                if _node_feats.dtype == torch.bool:
                    _node_feats = _node_feats.float()
                if _node_feats.shape[1] < 172:
                    _node_feats = torch.nn.functional.pad(_node_feats,(0,172-_node_feats.shape[1]))
            else:
                _node_feats = torch.randn(num_node,172)
        else:
            _node_feats, _edge_feats = load_feat(data, num_node, src.shape[0], 172, 172)""",
)
s = s.replace(
    "    if reid is not None:",
    """    if data == 'GDELT':
        edge_feats=torch.load('/mnt/nfs/fzz/TGL-DATA/GDELT/edge_features.pt',mmap=True)
        assert edge_feats.dtype==torch.bool and edge_feats.shape[1]>=172 and reid is None
    if reid is not None:""",
    1,
)
s = s.replace(
    "edge_feats = torch.cat((edge_feats[eids],edge_feats[drop_local_eids]),dim = 0)",
    'edge_feats = torch.cat((edge_feats[eids],edge_feats[drop_local_eids]),dim = 0)\n        if data == "GDELT": edge_feats=edge_feats.float()',
)
p.write_text(s)
p = Path(sys.argv[1]) / "examples/train_boundery.py"
s = p.read_text()
start = s.index("    eval_sampler = NeighborSampler(")
end = s.index("\n", start)
s = (
    s[:start]
    + s[start:end].replace(
        "= NeighborSampler(", "= sampler if args.skip_eval else NeighborSampler("
    )
    + s[end:]
)
p.write_text(s)

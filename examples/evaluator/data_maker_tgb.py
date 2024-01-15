import argparse
from tgb.linkproppred.dataset_pyg import PyGLinkPropPredDataset

parser = argparse.ArgumentParser(
    description="RPC Reinforcement Learning Example",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('--data_name', default='tgbl-wiki', type=str, metavar='W',
                    help='name of dataset')
parser.add_argument('--num_neg_sample', default=1, type=int, metavar='W',
                    help='number of negative samples')
args = parser.parse_args()

name = args.data_name
dataset = PyGLinkPropPredDataset(name=name, root="datasets")
print(dataset)
#torchrun --nnodes 1 --node_rank 0 --nproc-per-node 1 --master-addr  192.168.1.105 --master-port 9445 train_boundery.py --dataname WIKI --mode TGN --partition ours --topk 0 > memory/WIKI.out &
#wait
#torchrun --nnodes 1 --node_rank 0 --nproc-per-node 1 --master-addr  192.168.1.105 --master-port 9445 train_boundery.py --dataname GDELT --mode TGN_large --partition ours --topk 0 > memory/GDELT.out &
#wait
torchrun --nnodes 1 --node_rank 0 --nproc-per-node 1 --master-addr  192.168.1.105 --master-port 9445 train_boundery.py --dataname StackOverflow --mode TGN_large --partition ours --topk 0 > memory/StackOverflow.out &
wait
torchrun --nnodes 1 --node_rank 0 --nproc-per-node 1 --master-addr  192.168.1.105 --master-port 9445 train_boundery.py --dataname WikiTalk --mode TGN_large --partition ours --topk 0 > memory/WikiTalk.out &
wait

torchrun --nnodes 1 --node_rank 0 --nproc-per-node 1 --master-addr  192.168.1.105 --master-port 9445 train_boundery.py --dataname LASTFM --mode TGN --partition ours --topk 0 > memory/LASTFM.out &
wait
torchrun --nnodes 1 --node_rank 0 --nproc-per-node 1 --master-addr  192.168.1.105 --master-port 9445 train_boundery.py --dataname REDDIT --mode TGN --partition ours --topk 0 > memory/REDDIT.out &
wait
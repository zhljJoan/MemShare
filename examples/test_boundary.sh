#torchrun --nnodes 1 --node_rank 0 --nproc-per-node 1 --master-addr  192.168.1.105 --master-port 9445 train_boundery.py --dataname WIKI --mode TGN --partition ours --topk 0 > memory/WIKI.out &
#wait
#torchrun --nnodes 1 --node_rank 0 --nproc-per-node 1 --master-addr  192.168.1.105 --master-port 9445 train_boundery.py --dataname GDELT --mode TGN_large --partition ours --topk 0 > memory/GDELT.out &
#wait
torchrun --nnodes 1 --node_rank 0 --nproc-per-node 4 --master-addr  192.168.1.105 --master-port 9445 train_boundery.py --dataname WIKI --mode TGN --partition ours --topk 0 --probability 1 > boundary_sample/WIKI-1.out &
wait
torchrun --nnodes 1 --node_rank 0 --nproc-per-node 4 --master-addr  192.168.1.105 --master-port 9445 train_boundery.py --dataname WIKI --mode TGN --partition ours --topk 0 --probability 0.1 > boundary_sample/WIKI-01.out &
wait
torchrun --nnodes 1 --node_rank 0 --nproc-per-node 4 --master-addr  192.168.1.105 --master-port 9445 train_boundery.py --dataname WIKI --mode TGN --partition ours --topk 0 --probability 0.01 > boundary_sample/WIKI-001.out &
wait
torchrun --nnodes 1 --node_rank 0 --nproc-per-node 4 --master-addr  192.168.1.105 --master-port 9445 train_boundery.py --dataname WIKI --mode TGN --partition ours --topk 0 --probability 0.05 > boundary_sample/WIKI-005.out &
wait
torchrun --nnodes 1 --node_rank 0 --nproc-per-node 4 --master-addr  192.168.1.105 --master-port 9445 train_boundery.py --dataname WIKI --mode TGN --partition ours --topk 0 --probability 0.5 > boundary_sample/WIKI-05.out &
wait
torchrun --nnodes 1 --node_rank 0 --nproc-per-node 4 --master-addr  192.168.1.105 --master-port 9445 train_boundery.py --dataname WIKI --mode TGN --partition ours --topk 0 --probability 0 > boundary_sample/WIKI-0.out &
wait

torchrun --nnodes 1 --node_rank 0 --nproc-per-node 4 --master-addr  192.168.1.105 --master-port 9445 train_boundery.py --dataname REDDIT --mode TGN --partition ours --topk 0 --probability 1 > boundary_sample/REDDIT-1.out &
torchrun --nnodes 1 --node_rank 0 --nproc-per-node 4 --master-addr  192.168.1.105 --master-port 9445 train_boundery.py --dataname REDDIT --mode TGN --partition ours --topk 0 --probability 0.1 > boundary_sample/REDDIT-01.out &
wait
torchrun --nnodes 1 --node_rank 0 --nproc-per-node 4 --master-addr  192.168.1.105 --master-port 9445 train_boundery.py --dataname REDDIT --mode TGN --partition ours --topk 0 --probability 0.01 > boundary_sample/REDDIT-001.out &
wait
torchrun --nnodes 1 --node_rank 0 --nproc-per-node 4 --master-addr  192.168.1.105 --master-port 9445 train_boundery.py --dataname REDDIT --mode TGN --partition ours --topk 0 --probability 0.05 > boundary_sample/REDDIT-005.out &
wait
torchrun --nnodes 1 --node_rank 0 --nproc-per-node 4 --master-addr  192.168.1.105 --master-port 9445 train_boundery.py --dataname REDDIT --mode TGN --partition ours --topk 0 --probability 0.5 > boundary_sample/REDDIT-05.out &
wait
torchrun --nnodes 1 --node_rank 0 --nproc-per-node 4 --master-addr  192.168.1.105 --master-port 9445 train_boundery.py --dataname REDDIT --mode TGN --partition ours --topk 0 --probability 0 > boundary_sample/REDDIT-0.out &
wait

torchrun --nnodes 1 --node_rank 0 --nproc-per-node 4 --master-addr  192.168.1.105 --master-port 9445 train_boundery.py --dataname LASTFM --mode TGN --partition ours --topk 0 --probability 1 > boundary_sample/LASTFM-1.out &
torchrun --nnodes 1 --node_rank 0 --nproc-per-node 4 --master-addr  192.168.1.105 --master-port 9445 train_boundery.py --dataname LASTFM --mode TGN --partition ours --topk 0 --probability 0.1 > boundary_sample/LASTFM-01.out &
wait
torchrun --nnodes 1 --node_rank 0 --nproc-per-node 4 --master-addr  192.168.1.105 --master-port 9445 train_boundery.py --dataname LASTFM --mode TGN --partition ours --topk 0 --probability 0.01 > boundary_sample/LASTFM-001.out &
wait
torchrun --nnodes 1 --node_rank 0 --nproc-per-node 4 --master-addr  192.168.1.105 --master-port 9445 train_boundery.py --dataname LASTFM --mode TGN --partition ours --topk 0 --probability 0.05 > boundary_sample/LASTFM-005.out &
wait
torchrun --nnodes 1 --node_rank 0 --nproc-per-node 4 --master-addr  192.168.1.105 --master-port 9445 train_boundery.py --dataname LASTFM --mode TGN --partition ours --topk 0 --probability 0.5 > boundary_sample/LASTFM-05.out &
wait
torchrun --nnodes 1 --node_rank 0 --nproc-per-node 4 --master-addr  192.168.1.105 --master-port 9445 train_boundery.py --dataname LASTFM --mode TGN --partition ours --topk 0 --probability 0 > boundary_sample/LASTFM-0.out &
wait

torchrun --nnodes 1 --node_rank 0 --nproc-per-node 4 --master-addr  192.168.1.105 --master-port 9445 train_boundery.py --dataname WikiTalk --mode TGN_large --partition ours --topk 0 --probability 1 > boundary_sample/WikiTalk-1.out &
torchrun --nnodes 1 --node_rank 0 --nproc-per-node 4 --master-addr  192.168.1.105 --master-port 9445 train_boundery.py --dataname WikiTalk --mode TGN_large --partition ours --topk 0 --probability 0.1 > boundary_sample/WikiTalk-01.out &
wait
torchrun --nnodes 1 --node_rank 0 --nproc-per-node 4 --master-addr  192.168.1.105 --master-port 9445 train_boundery.py --dataname WikiTalk --mode TGN_large --partition ours --topk 0 --probability 0.01 > boundary_sample/WikiTalk-001.out &
wait
torchrun --nnodes 1 --node_rank 0 --nproc-per-node 4 --master-addr  192.168.1.105 --master-port 9445 train_boundery.py --dataname WikiTalk --mode TGN_large --partition ours --topk 0 --probability 0.05 > boundary_sample/WikiTalk-005.out &
wait
torchrun --nnodes 1 --node_rank 0 --nproc-per-node 4 --master-addr  192.168.1.105 --master-port 9445 train_boundery.py --dataname WikiTalk --mode TGN_large --partition ours --topk 0 --probability 0.5 > boundary_sample/WikiTalk-05.out &
wait
torchrun --nnodes 1 --node_rank 0 --nproc-per-node 4 --master-addr  192.168.1.105 --master-port 9445 train_boundery.py --dataname WikiTalk --mode TGN_large --partition ours --topk 0 --probability 0 > boundary_sample/WikiTalk-0.out &
wait


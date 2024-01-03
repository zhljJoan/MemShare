nohup python train_tgnn.py --dataname TaoBao --world_size 4 --rank 0 > taobao4.out &
nohup python train_tgnn.py --dataname TaoBao --world_size 4 --rank 1 > taobao4.out &
nohup python train_tgnn.py --dataname TaoBao --world_size 4 --rank 2 > taobao4.out &
nohup python train_tgnn.py --dataname TaoBao --world_size 4 --rank 3 > taobao4.out &

nohup python train_tgnn.py --dataname GDELT --world_size 4 --rank 0 > GDELT4.out &
nohup python train_tgnn.py --dataname GDELT --world_size 4 --rank 1 > GDELT4.out &
nohup python train_tgnn.py --dataname GDELT --world_size 4 --rank 2 > GDELT4.out &
nohup python train_tgnn.py --dataname GDELT --world_size 4 --rank 3 > GDELT4.out &

nohup python train_tgnn.py --dataname ML25M --world_size 2 --rank 0 > ML25M2.out &
nohup python train_tgnn.py --dataname ML25M --world_size 2 --rank 1 > ML25M2.out &

nohup python train_tgnn.py --dataname TaoBao --world_size 2 --rank 0 > TaoBao2.out &
nohup python train_tgnn.py --dataname TaoBao --world_size 2 --rank 1 > TaoBao2.out &



nohup python train_tgnn.py --dataname ML25M --world_size 1 --rank 0 > ML25M1.out &
nohup python train_tgnn.py --dataname TaoBao --world_size 1 --rank 0 > TaoBao1.out &
nohup python train_tgnn.py --dataname GDELT --world_size 1 --rank 0 > GDELT1.out &
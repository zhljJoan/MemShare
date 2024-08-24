#!/bin/bash

# 定义数组变量
partition_params=("ours" "metis" "ldg" "random")
partitions="4"
nnodes="1"
node_rank="0"
probability_params=("1" "0.5" "0.1" "0.05" "0.01" "0")
sample_type_params=("recent" "boundery_recent_decay" "boundery_recent_uniform")
memory_type=("all_update" "p2p" "all_reduce" "historical" "local")
shared_memory_ssim=("1" "2" "3" "10" "100")
data_param=("WIKI" "REDDIT" "LASTFM" "WikiTalk" "StackOverflow")

# 创建输出目录
mkdir -p all

# 遍历数组并执行命令
for data in "${data_param[@]}"; do
    torchrun --nnodes "$nnodes" --node_rank 0 --nproc-per-node 1 --master-addr 192.168.1.105 --master-port 9445 train_boundery.py --dataname "$data" --mode TGN_large --partition ours --memory_type local --sample_type recent --topk 0 > all/"$data"-1.out &
    wait
    for partition in "${partition_params[@]}"; do
        for sample in "${sample_type_params[@]}"; do
            if [ "$sample" = "recent" ]; then
                for mem in "${memory_type[@]}"; do
                    if [ "$mem" = "historical" ]; then
                        for ssim in "${shared_memory_ssim[@]}"; do
                            if [ "$partition" = "ours" ]; then
                                torchrun --nnodes "$nnodes" --node_rank "$node_rank" --nproc-per-node "$partitions" --master-addr 192.168.1.105 --master-port 9445 train_boundery.py --dataname "$data" --mode TGN_large --partition "$partition" --topk 0.01 --sample_type "$sample" --memory_type "$mem" --shared_memory_ssim "$ssim" > all/"$data"-"$partitions"-ours_shared-0.01-"$mem"-"$ssim"-"$sample".out &
                                wait
                            fi
                        done
                    else
                        torchrun --nnodes "$nnodes" --node_rank "$node_rank" --nproc-per-node "$partitions" --master-addr 192.168.1.105 --master-port 9445 train_boundery.py --dataname "$data" --mode TGN_large --partition "$partition" --topk 0 --sample_type "$sample" --memory_type "$mem" > all/"$data"-"$partitions"-"$partition"-0-"$mem"-"$sample".out &
                        wait
                        if [ "$partition" = "ours" ]; then
                            torchrun --nnodes "$nnodes" --node_rank "$node_rank" --nproc-per-node "$partitions" --master-addr 192.168.1.105 --master-port 9445 train_boundery.py --dataname "$data" --mode TGN_large --partition "$partition" --topk 0.01 --sample_type "$sample" --memory_type "$mem" > all/"$data"-"$partitions"-ours_shared-0.01-"$mem"-"$sample".out &
                            wait
                        fi
                    fi
                done
            else
                for pro in "${probability_params[@]}"; do
                    for mem in "${memory_type[@]}"; do
                        if [ "$mem" = "historical" ]; then
                            for ssim in "${shared_memory_ssim[@]}"; do
                                if [ "$partition" = "ours" ]; then
                                    torchrun --nnodes "$nnodes" --node_rank "$node_rank" --nproc-per-node "$partitions" --master-addr 192.168.1.105 --master-port 9445 train_boundery.py --dataname "$data" --mode TGN_large --partition "$partition" --topk 0.01 --sample_type "$sample" --probability "$pro" --memory_type "$mem" --shared_memory_ssim "$ssim" > all/"$data"-"$partitions"-ours_shared-0.01"$mem"-"$ssim"-"$sample"-"$pro".out &
                                    wait
                                fi
                            done
                        else
                            torchrun --nnodes "$nnodes" --node_rank "$node_rank" --nproc-per-node "$partitions" --master-addr 192.168.1.105 --master-port 9445 train_boundery.py --dataname "$data" --mode TGN_large --partition "$partition" --topk 0 --sample_type "$sample" --probability "$pro" --memory_type "$mem" > all/"$data"-"$partitions"-"$partition"-0-"$mem"-"$sample"-"$pro".out &
                            wait
                            if [ "$partition" = "ours" ]; then
                                torchrun --nnodes "$nnodes" --node_rank "$node_rank" --nproc-per-node "$partitions" --master-addr 192.168.1.105 --master-port 9445 train_boundery.py --dataname "$data" --mode TGN_large --partition "$partition" --topk 0.01 --sample_type "$sample" --probability "$pro" --memory_type "$mem" > all/"$data"-"$partitions"-ours_shared-0.01-"$mem"-"$sample"-"$pro".out &
                                wait
                            fi
                        fi
                    done
                done
            fi
        done
    done
done

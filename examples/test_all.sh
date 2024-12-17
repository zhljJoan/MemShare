#!/bin/bash
#跑了4卡的TaoBao
# 定义数组变量
seed=$1
addr="192.168.1.107"
partition_params=("ours")
#"metis" "ldg" "random")
#("ours" "metis" "ldg" "random")
partitions="8"
node_per="4"
nnodes="2"
node_rank="1"
probability_params=("0.1")
sample_type_params=("boundery_recent_decay")
#sample_type_params=("recent" "boundery_recent_decay") #"boundery_recent_uniform")
#memory_type=("all_update" "p2p" "all_reduce" "historical" "local")
memory_type=("historical" "all_update")
#"historical")
#memory_type=("local" "all_update" "historical" "all_reduce")
shared_memory_ssim=("0.3")
#data_param=("WIKI" "REDDIT" "LASTFM" "WikiTalk")
data_param=("LASTFM" "WikiTalk")
#"GDELT")
#data_param=("WIKI" "REDDIT" "LASTFM" "DGraphFin" "WikiTalk" "StackOverflow")
#data_param=("WIKI" "REDDIT" "LASTFM" "WikiTalk" "StackOverflow")
#data_param=("REDDIT" "WikiTalk")
# 创建输出目录


# 遍历数组并执行命令

#seed=(( RANDOM % 1000000 + 1 ))
mkdir -p all_"$seed"
for data in "${data_param[@]}"; do
    model="APAN_large"
    if [ "$data" = "WIKI" ] || [ "$data" = "REDDIT" ] || [ "$data" = "LASTFM" ]; then
        model="APAN"
    fi
    #model="APAN"
    mkdir all_"$seed"/"$data"
    mkdir all_"$seed"/"$data"/"$model"
    mkdir all_"$seed"/"$data"/"$model"/comm
    #torchrun --nnodes "$nnodes" --node_rank 0 --nproc-per-node 1 --master-addr "$addr" --master-port 9445 train_boundery.py --dataname "$data" --mode "$model" --partition ours --memory_type local --sample_type recent --topk 0 --seed "$seed" > all_"$seed"/"$data"/"$model"/1.out &
    wait
    for partition in "${partition_params[@]}"; do
        for sample in "${sample_type_params[@]}"; do
            if [ "$sample" = "recent" ]; then
                for mem in "${memory_type[@]}"; do
                    if [ "$mem" = "historical" ]; then
                        for ssim in "${shared_memory_ssim[@]}"; do
                            if [ "$partition" = "ours" ]; then
                                torchrun --nnodes "$nnodes" --node_rank "$node_rank" --nproc-per-node "$node_per" --master-addr "$addr" --master-port 9445 train_boundery.py --dataname "$data" --mode "$model" --partition "$partition" --topk 0.1 --sample_type "$sample" --memory_type "$mem" --shared_memory_ssim "$ssim" --seed "$seed" > all_"$seed"/"$data"/"$model"/"$partitions"-ours_shared-0.01-"$mem"-"$ssim"-"$sample".out &
                                wait
                            fi
                            if [ "$partition" = "metis" ]; then
                                torchrun --nnodes "$nnodes" --node_rank "$node_rank" --nproc-per-node "$node_per" --master-addr "$addr" --master-port 9445 train_boundery.py --dataname "$data" --mode "$model" --partition "$partition" --topk 0.1 --sample_type "$sample" --memory_type "$mem" --shared_memory_ssim "$ssim" --seed "$seed" > all_"$seed"/"$data"/"$model"/"$partitions"-ours_shared-0.01-"$mem"-"$ssim"-"$sample".out &
                                wait
                            fi
                        done
                    elif [ "$mem" = "all_reduce" ]; then
                        if [ "$partition" = "ours" ]; then
                            torchrun --nnodes "$nnodes" --node_rank "$node_rank" --nproc-per-node "$node_per" --master-addr "$addr" --master-port 9445 train_boundery.py --dataname "$data" --mode "$model" --partition "$partition" --topk 0.1 --sample_type "$sample" --memory_type "$mem"  --seed "$seed" > all_"$seed"/"$data"/"$model"/"$partitions"-ours_shared-0.01-"$mem"-"$sample".out &
                            wait
                        fi
                    else
                        torchrun --nnodes "$nnodes" --node_rank "$node_rank" --nproc-per-node "$node_per" --master-addr "$addr" --master-port 9445 train_boundery.py --dataname "$data" --mode "$model" --partition "$partition" --topk 0 --sample_type "$sample" --memory_type "$mem" --seed "$seed" > all_"$seed"/"$data"/"$model"/"$partitions"-"$partition"-0-"$mem"-"$sample".out &
                        wait
                        if [ "$partition" = "ours" ] && [ "$mem" != "all_local" ]; then
                            torchrun --nnodes "$nnodes" --node_rank "$node_rank" --nproc-per-node "$node_per" --master-addr "$addr" --master-port 9445 train_boundery.py --dataname "$data" --mode "$model" --partition "$partition" --topk 0.1 --sample_type "$sample" --memory_type "$mem" --seed "$seed" > all_"$seed"/"$data"/"$model"/"$partitions"-ours_shared-0.01-"$mem"-"$sample".out &
                            wait
                        fi
                        if [ "$partition" = "metis" ] && [ "$mem" != "all_local" ]; then
                            torchrun --nnodes "$nnodes" --node_rank "$node_rank" --nproc-per-node "$node_per" --master-addr "$addr" --master-port 9445 train_boundery.py --dataname "$data" --mode "$model" --partition "$partition" --topk 0.1 --sample_type "$sample" --memory_type "$mem" --seed "$seed" > all_"$seed"/"$data"/"$model"/"$partitions"-ours_shared-0.01-"$mem"-"$sample".out &
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
                                     torchrun --nnodes "$nnodes" --node_rank "$node_rank" --nproc-per-node "$node_per" --master-addr "$addr" --master-port 9445 train_boundery.py --dataname "$data" --mode "$model" --partition "$partition" --topk 0.1 --sample_type "$sample" --probability "$pro" --memory_type "$mem" --shared_memory_ssim "$ssim" --seed "$seed" > all_"$seed"/"$data"/"$model"/"$partitions"-ours_shared-0.01-"$mem"-"$ssim"-"$sample"-"$pro".out &
                                     wait
                                 fi
                             done
                        elif [ "$mem" = "all_reduce" ]; then
                            if [ "$partition" = "ours"]; then
                                torchrun --nnodes "$nnodes" --node_rank "$node_rank" --nproc-per-node "$node_per" --master-addr "$addr" --master-port 9445 train_boundery.py --dataname "$data" --mode "$model" --partition "$partition" --topk 0.1 --sample_type "$sample" --probability "$pro" --memory_type "$mem"  --seed "$seed" > all_"$seed"/"$data"/"$model"/"$partitions"-ours_shared-0.01-"$mem"-"$sample"-"$pro".out&
                                wait
                            fi
                        else
                            torchrun --nnodes "$nnodes" --node_rank "$node_rank" --nproc-per-node "$node_per" --master-addr "$addr" --master-port 9445 train_boundery.py --dataname "$data" --mode "$model" --partition "$partition" --topk 0 --sample_type "$sample" --probability "$pro" --memory_type "$mem" --seed "$seed" > all_"$seed"/"$data"/"$model"/"$partitions"-"$partition"-0-"$mem"-"$sample"-"$pro".out &
                            wait
                            if [ "$partition" = "ours" ] && [ "$mem" != "all_local" ]; then
                                torchrun --nnodes "$nnodes" --node_rank "$node_rank" --nproc-per-node "$node_per" --master-addr "$addr" --master-port 9445 train_boundery.py --dataname "$data" --mode "$model" --partition "$partition" --topk 0.1 --sample_type "$sample" --probability "$pro" --memory_type "$mem" --seed "$seed" > all_"$seed"/"$data"/"$model"/"$partitions"-ours_shared-0.01-"$mem"-"$sample"-"$pro".out &
                                wait
                            fi
                            if [ "$partition" = "metis" ] && [ "$mem" != "all_local" ]; then
                                torchrun --nnodes "$nnodes" --node_rank "$node_rank" --nproc-per-node "$node_per" --master-addr "$addr" --master-port 9445 train_boundery.py --dataname "$data" --mode "$model" --partition "$partition" --topk 0.1 --sample_type "$sample" --probability "$pro" --memory_type "$mem" --seed "$seed" > all_"$seed"/"$data"/"$model"/"$partitions"-ours_shared-0.01-"$mem"-"$sample"-"$pro".out &
                                wait
                            fi
                        fi
                    done
                done
            fi
        done
    done
done


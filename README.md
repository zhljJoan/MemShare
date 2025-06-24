# Effective and Efficient Distributed Temporal Graph Learning through Hotspot Memory Sharing

## Overview

This repo is the open-sourced code for our work *TGL: A General Framework for Temporal Graph Training on Billion-Scale Graphs*.

## Requirements
- cuda=11.8
- python >= 3.8
- pytorch >= 1.8.1
- pandas >= 1.1.5
- numpy >= 1.19.5
- dgl >= 0.6.1
- pyyaml >= 5.4.1
- tqdm >= 4.61.0
- pybind11 >= 2.6.2
- g++ >= 7.5.0
- openmp >= 201511
- Install the required environment using requirements.txt
Our temporal sampler is implemented using C++, please compile the sampler first with the following command
> bash install.sh

## Dataset

In our paper, we utilized four datasets: LASTFM, WIKITALK, STACKOVERFLOW, and GDELT. To obtain the LASTFM and GDELT datasets, you can download them from the TGL project(https://github.com/amazon-science/tgl).
For WIKITALK and STACKOVERFLOW
WIKITALK: http://snap.stanford.edu/data/wiki-talk-temporal.html
STACKOVERFLOW: https://snap.stanford.edu/data/sx-stackoverflow.html
To generate TGL format data, we used the preprocessing code from ETC (Efficient Training of Temporal Graph Neural Networks over Large-scale Dynamic Graphs). You can find the code at https://github.com/eddiegaoo/ETC.


## Configuration Files

The format of configuration files is follow TGL and are located at `/config/` 

## Run

### MultiGPU Link Prediction
>cd examples

>torchrun --nnodes \<NumberOfMachines> --node_rank \<RankOfGPUs> --nproc-per-node \<NumberOfGPUsperMachine> --master-addr \<addr> --master-port \<port>  train_boundery.py --dataname \<NameOfYourDATA> --mode \<Model> --partition \<PartitionFile> --topk \<RatioOfHotNodes> --sample_type \<PartitionMethod(boundary_recent_decayl/recent/local)> --probability \<BoundaryDecaySamplingProbability> --memory_type \<MemoryUpdateType(historical/all_update/local)> --shared_memory_ssim \<MemoryInspectionThreshold> --seed \<RandomSeed>  

\<PartitionFile> contains three files, including:
> node_i = '{data}/{seed}/{data}_{NumOfGPUs}parts_top{RatioOfHotNOdes}/output{RANK}.txt', which is the node ID stored in each GPU.
> shared_node = '{data}/{seed}/{data}_{NumOfGPUs}parts_top{RatioOfHotNOdes}/outputshared.txt'.
> edge_i = '{data}/{seed}/{data}_{NumOfGPUs}parts_top{RatioOfHotNOdes}/edge_output{RANK}.txt', which is the edge ID stored in each GPU.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

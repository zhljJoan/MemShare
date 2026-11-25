#!/usr/bin/env bash
set -eu
experiment=$(cd "$(dirname "$0")" && pwd)
root=${MEMSHARE_TRAIN_ROOT:?set to the coordinated-sampling training checkout}
mode=${1:?native or hybrid}
output=$(realpath -m "${2:?fresh absolute output directory}")
mkdir "$output"
# The coordinated training checkout contains the native extension and the
# current model. Overlay only this branch's optional loader in an isolated copy.
code="$root"
if [[ ${MEMSHARE_PREVIOUS_EDGE_FEATURES:-0} == 1 ]]; then
    code="$output/source"
    mkdir "$code"
    cp -a "$root/starrygl" "$code/"
    cp "$experiment/../../starrygl/sample/"{data_loader,previous_edge_features}.py "$code/starrygl/sample/"
fi
cd "$root/examples"
export PYTHONPATH="$code:$root:$experiment:${PYTHONPATH:-}"
export OMP_NUM_THREADS=10 MKL_NUM_THREADS=10
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-lo}
exec python -m torch.distributed.run --standalone --nproc_per_node=4 \
    "$experiment/train_entry.py" "$mode" --model TGN_large --dataname LASTFM \
    --partition /home/zlj/SPEED/partition/divided_nodes_seed_starrygl --topk 0.1 \
    --memory_type historical --probability 0.1 --shared_memory_ssim 0.3 --seed 6773 \
    --sample_type boundery_recent_decay_coordinated --coordinated_group_size 8 \
    --block_dedup on --state_dedup on --compact_memory_ops on --max_epoch 3 \
    --freeze_adaptive_parameters --skip_eval --hierarchical_compaction off \
    --compact_gru_update off --fused_segment_attention off --compact_kv_projection off \
    --output_dir "$output" --metrics_path "$output/epochs.jsonl"

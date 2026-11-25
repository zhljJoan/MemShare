#!/usr/bin/env bash
set -eu
root=${MEMSHARE_TRAIN_ROOT:?set to the coordinated training snapshot}
dataset=${1:?dataset}
model=${2:?model configuration}
output=$(realpath -m "${3:?fresh output directory}")
epochs=${4:-3}
mode=fixed
if [[ $model == JODIE_large ]]; then mode=native; fi
experiment=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$output"
test ! -e "$output/epochs.jsonl"
export PYTHONPATH="$root:$experiment:${PYTHONPATH:-}" OMP_NUM_THREADS=10 MKL_NUM_THREADS=10
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-lo}
cd "$root/examples"
exec python -m torch.distributed.run --standalone --nproc_per_node=4 "$experiment/train_entry.py" "$mode" \
    --model "$model" --dataname "$dataset" --partition "${MEMSHARE_PARTITIONS:-/home/zlj/SPEED/partition/divided_nodes_seed_starrygl}" \
    --topk 0.1 --memory_type historical --probability 0.1 --shared_memory_ssim 0.3 --seed 6773 \
    --sample_type boundery_recent_decay_coordinated --coordinated_group_size 8 --block_dedup on \
    --state_dedup on --compact_memory_ops on --max_epoch "$epochs" --freeze_adaptive_parameters \
    --skip_eval --hierarchical_compaction off --compact_gru_update off --fused_segment_attention off \
    --compact_kv_projection off --output_dir "$output" --metrics_path "$output/epochs.jsonl"

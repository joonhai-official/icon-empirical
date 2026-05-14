#!/bin/bash
# scripts/run.sh
#
# Launch one runner process per GPU on a p4d.24xlarge (8x A100 40 GB).
# Each process handles one shard of the condition list independently.
# All shards write to separate JSONL files under results/.
#
# Usage
#   bash scripts/run.sh [all|main|sigma|temp|epoch]
#   EXP=sigma bash scripts/run.sh   # alternative env-var form
#
# After all shards finish:
#   python experiments/aggregate.py

EXP=${1:-${EXP:-all}}
OUT_DIR=${OUT_DIR:-./results}
DATA_ROOT=${DATA_ROOT:-./data}
N_SHARDS=8

mkdir -p "$OUT_DIR" logs

echo "========================================"
echo "Icon_Empirical  exp=$EXP  gpus=$N_SHARDS"
echo "out=$OUT_DIR  data=$DATA_ROOT"
echo "========================================"

PIDS=()
for SHARD in $(seq 0 $((N_SHARDS-1))); do
    GPU=$SHARD
    OUT_FILE="$OUT_DIR/shard_${SHARD}.jsonl"
    LOG_FILE="logs/shard_${SHARD}.log"

    # GPU isolation: CUDA_VISIBLE_DEVICES limits each process to one physical
    # GPU; --gpu 0 then refers to that GPU within the process's view.
    # Append to log so prior runs are not lost on resume.
    CUDA_VISIBLE_DEVICES=$GPU python3 experiments/runner.py \
        --gpu       0 \
        --shard     $SHARD \
        --n_shards  $N_SHARDS \
        --out       "$OUT_FILE" \
        --data_root "$DATA_ROOT" \
        --exp       "$EXP" \
        >> "$LOG_FILE" 2>&1 &

    PIDS+=($!)
    echo "  shard $SHARD -> GPU $GPU  PID=${PIDS[-1]}  log=$LOG_FILE"
done

echo ""
echo "Monitor progress:"
echo "  tail -f logs/shard_0.log"
echo "  watch 'wc -l results/*.jsonl | tail -1'"
echo ""

# Wait for all shards and collect exit codes.
FAILED=0
for PID in "${PIDS[@]}"; do
    wait "$PID" || FAILED=$((FAILED + 1))
done

if [ "$FAILED" -gt 0 ]; then
    echo "WARNING: $FAILED shard(s) exited with non-zero status."
    echo "Check logs/ for details.  Completed records are safe to aggregate."
    exit 1
fi

echo "All shards done.  Run: python experiments/aggregate.py"

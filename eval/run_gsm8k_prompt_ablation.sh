#!/usr/bin/env bash
set -euo pipefail

# Evaluation-only prompt ablation for the nine matched step-300 checkpoints.
# Each worker runs its assigned models sequentially on one GPU. Completed
# model/variant files are detected by line count and skipped on restart.

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-/home/lian/miniconda3/envs/vllm-serve/bin/python}"
log_dir="${repo_dir}/eval/results_prompt_ablation/logs"
mkdir -p "${log_dir}"

export PATH="/home/lian/miniconda3/envs/vllm-serve/bin:${PATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

run_worker() {
    local gpu="$1"
    shift
    local model
    for model in "$@"; do
        CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" \
            "${repo_dir}/eval/eval_gsm8k_prompt_ablation.py" \
            --model "${model}" --max_tokens 8192
        CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" \
            "${repo_dir}/eval/eval_gsm8k_prompt_open_tag_ablation.py" \
            --model "${model}" --max_tokens 8192
    done
}

# A GPU index followed by model keys runs one worker directly; no arguments
# launches the complete four-worker schedule below.
if (( $# > 0 )); then
    run_worker "$@"
    exit 0
fi

run_worker 0 matched_ppo_s7_step300 matched_dapo_s7_step300 \
    >"${log_dir}/gpu0.log" 2>&1 &
pid0=$!
run_worker 1 matched_ppo_s8_step300 matched_dapo_s8_step300 \
    >"${log_dir}/gpu1.log" 2>&1 &
pid1=$!
run_worker 2 matched_ppo_s9_step300 matched_dapo_s9_step300 \
    >"${log_dir}/gpu2.log" 2>&1 &
pid2=$!
run_worker 3 matched_grpo_s7_step300 matched_grpo_s8_step300 matched_grpo_s9_step300 \
    >"${log_dir}/gpu3.log" 2>&1 &
pid3=$!

printf 'workers: %s %s %s %s\n' "${pid0}" "${pid1}" "${pid2}" "${pid3}"
wait "${pid0}" "${pid1}" "${pid2}" "${pid3}"

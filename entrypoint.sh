#!/usr/bin/env bash
# =============================================================================
# entrypoint.sh — wraps `vllm serve` for DeepSeek-V4-Flash (SM89 fork).
#
# All tuning knobs are read from environment variables so Kubernetes operators
# can override them in the Deployment / ConfigMap without rebuilding the image.
# =============================================================================
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/models/DeepSeek-V4-Flash}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

# Base command — model path is the positional arg to `vllm serve`
ARGS=(serve "$MODEL_PATH")
ARGS+=(--host "$HOST" --port "$PORT")

# --- Service identity & model architecture -----------------------------------
ARGS+=(--served-model-name "${SERVED_MODEL_NAME:-deepseek-v4-flash}")

# --- Hardware / memory tuning (defaults tuned for 8x L40S, TP=8) ------------
ARGS+=(--tensor-parallel-size "${TENSOR_PARALLEL_SIZE:-8}")
ARGS+=(--gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.92}")

# --- Context / KV cache (the "max context length" knob) ----------------------
ARGS+=(--max-model-len "${MAX_MODEL_LEN:-262144}")        # 256K; model arch max is 1M
ARGS+=(--max-num-seqs "${MAX_NUM_SEQS:-16}")
ARGS+=(--block-size "${BLOCK_SIZE:-256}")
ARGS+=(--kv-cache-dtype "${KV_CACHE_DTYPE:-fp8_ds_mla}")

# --- DeepSeek-V4 / SM89-specific backends and parsers ------------------------
ARGS+=(--attention-backend "${ATTENTION_BACKEND:-FLASHINFER_MLA_SPARSE_DSV4}")
ARGS+=(--reasoning-parser "${REASONING_PARSER:-deepseek_v4}")
ARGS+=(--tool-call-parser "${TOOL_CALL_PARSER:-deepseek_v4}")

# --- Optional flags (only added when set) ------------------------------------
[[ -n "${MAX_NUM_BATCHED_TOKENS:-}" ]] && ARGS+=(--max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS")
# Optional DSpark speculative decoding: '{"method":"dspark","num_speculative_tokens":6,"draft_sample_method":"greedy"}'
[[ -n "${SPECULATIVE_CONFIG:-}" ]]        && ARGS+=(--speculative-config "$SPECULATIVE_CONFIG")
[[ -n "${DATA_PARALLEL_SIZE:-}" ]]        && ARGS+=(--data-parallel-size "$DATA_PARALLEL_SIZE")
[[ -n "${ENABLE_PREFIX_CACHING:-}" ]] && [[ "${ENABLE_PREFIX_CACHING}" == "true" ]] && ARGS+=(--enable-prefix-caching)

# --- Feature toggles ----------------------------------------------------------
[[ "${ENABLE_AUTO_TOOL_CHOICE:-true}" == "true" ]] && ARGS+=(--enable-auto-tool-choice)
[[ "${TRUST_REMOTE_CODE:-true}"    == "true" ]] && ARGS+=(--trust-remote-code)
[[ "${ENABLE_REASONING:-true}"     == "true" ]] && ARGS+=(--enable-reasoning)

# --- Arbitrary extra vLLM flags (operator escape hatch) -----------------------
if [[ -n "${EXTRA_ARGS:-}" ]]; then
  read -r -a EXTRA <<< "$EXTRA_ARGS"
  ARGS+=("${EXTRA[@]}")
fi

echo "[entrypoint] launching: vllm ${ARGS[*]}" >&2
exec vllm "${ARGS[@]}"

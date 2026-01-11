#!/usr/bin/env bash
set -euo pipefail

# Example launcher for teacher SFT generation.
# Usage:
#   bash scripts/run_generate_teacher_sft.sh \
#     /path/to/input.jsonl \
#     /path/to/output.jsonl \
#     gemini-2.5-flash \
#     http://localhost:8000/v1 \
#     gpt-4.1 \
#     2

INPUT_JSONL="${1:-}"
OUTPUT_JSONL="${2:-}"
MODEL_NAME="${3:-}"
API_BASE="${4:-}"
JUDGE_MODEL="${5:-}"
MAX_CORRECTIONS="${6:-1}"

if [[ -z "${INPUT_JSONL}" || -z "${OUTPUT_JSONL}" || -z "${MODEL_NAME}" || -z "${API_BASE}" ]]; then
  echo "Usage: $0 <input_jsonl> <output_jsonl> <model_name> <api_base>"
  exit 1
fi

export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export JUDGE_API_KEY="${JUDGE_API_KEY:-${OPENAI_API_KEY}}"

python scripts/generate_teacher_sft.py \
  --input "${INPUT_JSONL}" \
  --output "${OUTPUT_JSONL}" \
  --model "${MODEL_NAME}" \
  --api-base "${API_BASE}" \
  --api-key "${OPENAI_API_KEY}" \
  --mcp-server-path "examples/video_tools/mcp_server.py" \
  --tool-output-dir "./tool_outputs" \
  --frame-output-dir "./frame_outputs" \
  --workers 8 \
  --max-tokens 2048 \
  --temperature 0.2 \
  --fps 1 \
  --max-frames 512 \
  --max-pixels 50176 \
  --max-payload-mb 20 \
  --overlap-threshold 0.3 \
  --max-corrections "${MAX_CORRECTIONS}" \
  --answer-iou-threshold 0.5 \
  ${JUDGE_MODEL:+--judge-model "${JUDGE_MODEL}"} \
  ${JUDGE_MODEL:+--judge-api-base "${API_BASE}"} \
  ${JUDGE_MODEL:+--judge-api-key "${JUDGE_API_KEY}"} \
  ${JUDGE_MODEL:+--judge-max-tokens 16} \
  ${JUDGE_MODEL:+--judge-temperature 0.0}

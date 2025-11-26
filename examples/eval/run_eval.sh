#!/bin/bash

# Environment variables for LLM Judge
export OPENAI_API_BASE="http://localhost:8000/v1"
export OPENAI_MODEL_NAME="judge"
export OPENAI_BASE_URL="http://your-judge-server:8000/v1"
export OPENAI_API_KEY="EMPTY"
export USE_LLM_JUDGE=True

# Cache settings
export DECORD_EOF_RETRY_MAX=409600
export LMMS_EVAL_USE_CACHE=True
export LMMS_EVAL_HOME=""  # Set your cache directory

# Arguments
CKPT_PATH=$1        # Path to model checkpoint
TASK_NAME=$2        # Evaluation task name
IS_QWEN3_VL=$3      # Whether using Qwen3-VL model (True/False)
MAX_FRAME_NUM=${4:-768}

# Path to MCP server for tool calling
MCP_PATH="./examples/video_tools/mcp_server.py"

# Start vLLM server
# Qwen3 VL does not need additional chat template
if [ "$IS_QWEN3_VL" == "False" ]; then
    vllm serve $CKPT_PATH \
        --chat-template ./examples/eval/tool_call_qwen2_5_vl.jinja \
        --tool-call-parser hermes \
        --enable-auto-tool-choice \
        --data-parallel-size 8 \
        --trust-remote-code &
else
    vllm serve $CKPT_PATH \
        --tool-call-parser hermes \
        --enable-auto-tool-choice \
        --data-parallel-size 8 \
        --trust-remote-code &
fi
sleep 240

source path/to/lmms-eval/.venv/bin/activate

# Run evaluation
accelerate launch --num_processes=8 --main_process_port 12345 -m lmms_eval \
    --model async_openai \
    --model_args model_version=$CKPT_PATH,mcp_server_path=$MCP_PATH,fps=1,max_frames=$MAX_FRAME_NUM,max_pixels=50176,base_url=$OPENAI_API_BASE,api_key=$OPENAI_API_KEY,num_cpus=1,timeout=12000,is_qwen3_vl=$IS_QWEN3_VL \
    --tasks $TASK_NAME \
    --batch_size 1 \
    --output_path ./eval_logs \
    --log_samples \
    --include_path ./lmms_eval_tasks


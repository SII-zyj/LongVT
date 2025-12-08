#!/bin/bash
# -*- coding: utf-8 -*-
#
# Text Summary Script
# 
# This script performs hierarchical summarization of video captions using LLM.
#
# Usage:
#   ./scripts/run_text_summary.sh --input-dir /path/to/captions --output-file /path/to/output.json
#
# Environment Variables:
#   OPENAI_API_KEY      - OpenAI API key (required)
#   OPENAI_BASE_URL     - OpenAI-compatible API base URL (optional)
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$(dirname "$SCRIPT_DIR")"

# Default configuration
MODEL="gpt-4o"
GROUP_SIZE=8
TOTAL_SHARDS=1

# Help message
show_help() {
    echo "Text Summary Script"
    echo ""
    echo "Usage: $0 [options]"
    echo ""
    echo "Required Options:"
    echo "  --input-dir DIR      Input directory containing caption JSON files"
    echo "  --output-file FILE   Output JSON file for video summaries"
    echo ""
    echo "Optional Options:"
    echo "  --model MODEL        LLM model to use (gpt-4o or o3, default: gpt-4o)"
    echo "  --group-size N       Group size for long videos (default: 8)"
    echo "  --shard-id N         Current shard ID for parallel processing (0-based)"
    echo "  --total-shards N     Total number of shards (default: 1)"
    echo "  --log-level LEVEL    Log level (DEBUG, INFO, WARNING, ERROR, default: INFO)"
    echo "  -h, --help           Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  OPENAI_API_KEY       OpenAI API key (required)"
    echo "  OPENAI_BASE_URL      OpenAI-compatible API base URL (optional)"
    echo ""
    echo "Examples:"
    echo "  # Basic usage"
    echo "  $0 --input-dir ./captions --output-file ./summary.json"
    echo ""
    echo "  # With parallel processing"
    echo "  $0 --input-dir ./captions --output-file ./summary.json --shard-id 0 --total-shards 4"
    echo ""
    echo "  # Custom group size"
    echo "  $0 --input-dir ./captions --output-file ./summary.json --group-size 10"
    echo ""
}

# Parse arguments
PYTHON_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --input-dir)
            INPUT_DIR="$2"
            PYTHON_ARGS+=(--input-dir "$2")
            shift 2
            ;;
        --output-file)
            OUTPUT_FILE="$2"
            PYTHON_ARGS+=(--output-file "$2")
            shift 2
            ;;
        --model)
            MODEL="$2"
            PYTHON_ARGS+=(--model "$2")
            shift 2
            ;;
        --group-size)
            GROUP_SIZE="$2"
            PYTHON_ARGS+=(--group-size "$2")
            shift 2
            ;;
        --shard-id)
            PYTHON_ARGS+=(--shard-id "$2")
            shift 2
            ;;
        --total-shards)
            PYTHON_ARGS+=(--total-shards "$2")
            shift 2
            ;;
        --log-level)
            PYTHON_ARGS+=(--log-level "$2")
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$INPUT_DIR" ]] || [[ -z "$OUTPUT_FILE" ]]; then
    echo "Error: --input-dir and --output-file are required"
    show_help
    exit 1
fi

# Check API key
if [[ -z "$OPENAI_API_KEY" ]]; then
    echo "Error: OPENAI_API_KEY environment variable is not set"
    exit 1
fi

# Check input directory
if [[ ! -d "$INPUT_DIR" ]]; then
    echo "Error: Input directory does not exist: $INPUT_DIR"
    exit 1
fi

# Create output directory if needed
OUTPUT_DIR="$(dirname "$OUTPUT_FILE")"
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "Text Summary Configuration"
echo "=========================================="
echo "Input Directory:  $INPUT_DIR"
echo "Output File:      $OUTPUT_FILE"
echo "Model:            $MODEL"
echo "Group Size:       $GROUP_SIZE"
echo "=========================================="

# Run text summary
python "$DATA_DIR/launch/text_summary.py" "${PYTHON_ARGS[@]}"

echo "Text summary complete!"


import datetime
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import yaml
from decord import VideoReader, cpu
from loguru import logger as eval_logger

import lmms_eval.tasks._task_utils.file_utils as file_utils
from custom_rewards.lmms_lab_recipe import extract_anwser_tag

# with open(Path(__file__).parent / "_default_template.yaml", "r") as f:
#     raw_data = f.readlines()
#     safe_data = []
#     for i, line in enumerate(raw_data):
#         # remove function definition since yaml load cannot handle it
#         if "!function" not in line:
#             safe_data.append(line)

#     config = yaml.safe_load("".join(safe_data))

SYSTEM_PROMPT = (
    "You are a helpful assistant. When the user asks a question, your response must include two parts: "
    "first, the reasoning process enclosed in <think>...</think> tags, then the final answer enclosed in <answer>...</answer> tags."
    "Please provide a clear, concise response within <answer> </answer> tags that directly addresses the question."
)

TOOL_PROMPT = (
    "Think first, call **crop_video** if needed, then answer. Format strictly as:  <think>...</think>  "
    "<tool_call>...</tool_call> (if tools needed)  <answer>...</answer>."
)


hf_home = os.getenv("HF_HOME", "~/.cache/huggingface/")
# cache_dir = os.path.join(hf_home, cache_dir)
# base_cache_dir = config["dataset_kwargs"]["cache_dir"]
base_cache_dir = os.path.expanduser(hf_home)
with open(Path(__file__).parent / "charades_reasoning.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)

cache_name = yaml.safe_load("".join(safe_data))["dataset_kwargs"]["cache_dir"]


# DATA_LIST = {
#     "charades": 'your_data_dir/Charades/',
# }
# Pass in video path here
# Can only work correctly with video llm
def temporal_grounding_doc_to_visual(doc, lmms_eval_specific_kwargs=None):
    video_path = doc["video"]
    cache_dir = os.path.join(base_cache_dir, cache_name)
    video_path = os.path.join(cache_dir, "Charades_v1_480", video_path)
    if os.path.exists(video_path):
        video_path = video_path
    elif "s3://" not in video_path:
        sys.exit(f"video path:{video_path} does not exist, please check")

    return [video_path]


# This is the place where you format your question
def temporal_grounding_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    if "pre_prompt" in lmms_eval_specific_kwargs:
        pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    if "post_prompt" in lmms_eval_specific_kwargs:
        post_prompt = lmms_eval_specific_kwargs["post_prompt"]

    question = doc["caption"]

    return f"{pre_prompt}{question}. {post_prompt}"

def temporal_grounding_doc_to_messages(doc, lmms_eval_specific_kwargs=None):
    question = temporal_grounding_doc_to_text(doc, lmms_eval_specific_kwargs)
    visuals = temporal_grounding_doc_to_visual(doc)
    system_messages = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]}]
    messages = [{"role": "user", "content": []}]
    messages[0]["content"].append({"type": "video", "url": visuals[0]})
    messages[0]["content"].append({"type": "text", "text": question.strip()})
    messages = system_messages + messages
    return messages


def temporal_grounding_tool_doc_to_messages(doc, lmms_eval_specific_kwargs=None):
    question = temporal_grounding_doc_to_text(doc, lmms_eval_specific_kwargs)
    visuals = temporal_grounding_doc_to_visual(doc)
    messages = [{"role": "user", "content": []}]
    messages[0]["content"].append({"type": "video", "url": visuals[0]})
    messages[0]["content"].append({"type": "text", "text": question.strip() + " " + TOOL_PROMPT})
    return messages


def temporal_grounding_doc_to_answer(doc):
    return doc["timestamp"]


# Process result for mcq answer generation
def temporal_grounding_process_results_generation(doc, result):
    pred = result[0]
    pred = extract_anwser_tag(pred)
    return {"submission": {f'{doc["video"]}>>>{doc["caption"]}>>>{doc["timestamp"]}': pred}}


def temporal_grounding_aggregate_charades(results, args):
    temporal_grounding_aggregate_submissions(results, args, "charades")


def temporal_grounding_aggregate_submissions(results, args, task):
    now_date_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    submission_file_name = f"inference_results_temporal_grounding_{task}_{now_date_time}.json"
    path = file_utils.generate_submission_file(submission_file_name, args)

    # results is a list of 5031 dict,
    # need to convert results into a single dict with 5031 key-value pairs
    combined_submission = {}

    for submission_dict in results:
        combined_submission.update(submission_dict)

    with open(path, "w") as f:
        json.dump(combined_submission, f, indent=4)

    eval_logger.info(f"Submission file saved to {path}")

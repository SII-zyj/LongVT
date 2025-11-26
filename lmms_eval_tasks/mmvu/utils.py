import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Union

import yaml
from loguru import logger as eval_logger

from custom_rewards.lmms_lab_recipe import compute_score

hf_home = os.getenv("HF_HOME", "~/.cache/huggingface")

base_cache_dir = os.path.expanduser(hf_home)


with open(Path(__file__).parent / "mmvu_thinking.yaml", "r") as f:
    raw_data_val = f.readlines()
    safe_data_val = []
    for i, line in enumerate(raw_data_val):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data_val.append(line)
cache_name_val = yaml.safe_load("".join(safe_data_val))["dataset_kwargs"]["cache_dir"]
cache_dir_val = os.path.join(base_cache_dir, cache_name_val)


def mmvu_doc_to_visual_val(doc):
    video_path = doc["video_path"]
    video_path = os.path.join(cache_dir_val, video_path)
    if os.path.exists(video_path):
        video_path = video_path
    else:
        sys.exit(f"video path:{video_path} does not exist, please check")
    return [video_path]

SYSTEM_PROMPT = (
    "You are a helpful assistant. When the user asks a question, your response must include two parts: "
    "first, the reasoning process enclosed in <think>...</think> tags, then the final answer enclosed in <answer>...</answer> tags."
    "Please provide a clear, concise response within <answer> </answer> tags that directly addresses the question."
)

multiple_choice_prompt_cot = """
Question:{question}
A: {a}
B: {b}
C: {c}
D: {d}
E: {e}
"""

open_ended_prompt_cot = """
Question:{question}
"""


def mmvu_doc_to_text_cot(doc, lmms_eval_specific_kwargs=None):
    question_type = doc["question_type"]
    if question_type == "multiple-choice":
        question = doc["question"]
        choices = doc["choices"]
        full_prompt = multiple_choice_prompt_cot.format(question=question, a=choices["A"], b=choices["B"], c=choices["C"], d=choices["D"], e=choices["E"])
    else:
        question = doc["question"]
        full_prompt = open_ended_prompt_cot.format(question=question)
    return full_prompt

def mmvu_doc_to_messages_cot(doc, lmms_eval_specific_kwargs=None):
    question = mmvu_doc_to_text_cot(doc, lmms_eval_specific_kwargs)
    visuals = mmvu_doc_to_visual_val(doc)
    system_messages = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]}]
    messages = [{"role": "user", "content": []}]
    messages[0]["content"].append({"type": "video", "url": visuals[0]})
    messages[0]["content"].append({"type": "text", "text": question.strip()})
    messages = system_messages + messages
    return messages


def construct_question_prompt(doc):
    """Construct the question prompt for evaluation"""
    question_type = doc["question_type"]
    if question_type == "multiple-choice":
        choices = doc["choices"]
        return f"""Question: {doc["question"]}
A: {choices["A"]}
B: {choices["B"]}
C: {choices["C"]}
D: {choices["D"]}
E: {choices["E"]}"""
    else:
        return f"Question: {doc['question']}"


def extract_category(doc):
    category = doc["video_path"].split("/")[-2]
    return category


def mmvu_process_results(doc, results):
    acc_score = 0
    format_score = 0
    question = mmvu_doc_to_text_cot(doc, None)
    extra_info = {"question": question}
    for pred in results:
        score_dict = compute_score(data_source="mmvu", solution_str=pred.strip(), ground_truth=doc["answer"], extra_info=extra_info)
        acc_score += score_dict["acc_score"]
        format_score += score_dict.get("format_reward_score", 0.0)

    return {"acc_score": acc_score / len(results) if results else 0.0, "format_score": format_score / len(results) if results else 0.0}

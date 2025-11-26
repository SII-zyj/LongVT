import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Union

import decord
import numpy as np
import torch
import yaml
from decord import VideoReader, cpu
from loguru import logger as eval_logger
from PIL import Image

from custom_rewards.lmms_lab_recipe import compute_score

SYSTEM_PROMPT = (
    "You are a helpful assistant. When the user asks a question, your response must include two parts: "
    "first, the reasoning process enclosed in <think>...</think> tags, then the final answer enclosed in <answer>...</answer> tags."
    "Please provide a clear, concise response within <answer> </answer> tags that directly addresses the question."
)

TOOL_PROMPT = (
    "Think first, call **crop_video** if needed, then answer. Format strictly as:  <think>...</think>  "
    "<tool_call>...</tool_call> (if tools needed)  <answer>...</answer>."
)

def timestamp_to_seconds(timestamp):
    # Split the timestamp into hours, minutes, and seconds
    h, m, s = timestamp.split(":")
    # Convert hours, minutes, and total seconds (including fractions) to float and compute total seconds
    total_seconds = int(h) * 3600 + int(m) * 60 + float(s)
    return total_seconds


def load_video(video_file, duration, max_num_frames=16):
    from decord import VideoReader

    vr = VideoReader(video_file, ctx=cpu(0), num_threads=1)
    fps = vr.get_avg_fps()
    total_valid_frames = int(duration * fps)
    num_frames = min(max_num_frames, int(duration))

    frame_indices = [int(total_valid_frames / num_frames) * i for i in range(num_frames)]

    frames = vr.get_batch(frame_indices)
    if isinstance(frames, torch.Tensor):
        frames = frames.numpy()
    else:
        frames = frames.asnumpy()
    frame_timestamps = [frame_index / fps for frame_index in frame_indices]

    return [Image.fromarray(fr).convert("RGB") for fr in frames]


def compute_frame_timestamps(duration, max_num_frames=16):
    if duration > max_num_frames:
        return [duration / max_num_frames * i for i in range(max_num_frames)]
    else:
        return [i for i in range(int(duration))]


def insert_subtitles_into_frames(frame_timestamps, subtitles, starting_timestamp_for_subtitles, duration):
    interleaved_list = []
    cur_i = 0

    for subtitle in subtitles:
        if "timestamp" in subtitle:
            start, end = subtitle["timestamp"]

            if not isinstance(end, float):
                end = duration

            start -= starting_timestamp_for_subtitles
            end -= starting_timestamp_for_subtitles

            subtitle_timestamp = (start + end) / 2
            subtitle_text = subtitle["text"]
        else:
            start, end = subtitle["start"], subtitle["end"]
            start = timestamp_to_seconds(start)
            end = timestamp_to_seconds(end)
            start -= starting_timestamp_for_subtitles
            end -= starting_timestamp_for_subtitles

            subtitle_timestamp = (start + end) / 2
            subtitle_text = subtitle["line"]

        for i, frame_timestamp in enumerate(frame_timestamps[cur_i:]):
            if frame_timestamp <= subtitle_timestamp:
                # print("frame:", frame_timestamp)
                interleaved_list.append("<image>")
                cur_i += 1
            else:
                break

        if end - start < 1:
            end = subtitle_timestamp + 0.5
            start = subtitle_timestamp - 0.5

        covering_frames = False
        for frame_timestamp in frame_timestamps:
            if frame_timestamp < end and frame_timestamp > start:
                covering_frames = True
                break

        if covering_frames:
            # print("subtitle:", subtitle_timestamp, start, end)
            interleaved_list.append(subtitle_text)
        else:
            pass
            # print("leaving out subtitle:", start, end)

    for i, frame_timestamp in enumerate(frame_timestamps[cur_i:]):
        # print(frame_timestamp)
        interleaved_list.append("<image>")

    return "\n".join(interleaved_list)


def longvideobench_doc_to_text(doc, lmms_eval_specific_kwargs):
    candidates = []

    for i in range(5):
        candidate = doc.get(f"option{i}")
        if candidate != "N/A":
            candidates.append(candidate)

    question = doc["question"] + "\n" + "\n".join([". ".join([chr(ord("A") + i), candidate]) for i, candidate in enumerate(candidates)])
    pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    post_prompt = lmms_eval_specific_kwargs["post_prompt"]

    if lmms_eval_specific_kwargs.get("insert_interleave_subtitles", False):
        with open(Path(__file__).parent / "longvideobench_val_i.yaml", "r") as f:
            raw_data = f.readlines()
            safe_data = []
            for i, line in enumerate(raw_data):
                # remove function definition since yaml load cannot handle it
                if "!function" not in line:
                    safe_data.append(line)
        cache_name = yaml.safe_load("".join(safe_data))["dataset_kwargs"]["cache_dir"]
        subtitle_subdir_name = yaml.safe_load("".join(safe_data))["dataset_kwargs"].get("subtitle_subdir", "subtitles")
        cache_dir = os.path.join(base_cache_dir, cache_name, subtitle_subdir_name)
        with open(os.path.join(cache_dir, doc["subtitle_path"])) as f:
            subtitles = json.load(f)

        max_num_frames = yaml.safe_load("".join(safe_data))["dataset_kwargs"].get("max_num_frames", 16)

        frame_timestamps = compute_frame_timestamps(doc["duration"], max_num_frames)
        interleaved_prefix = insert_subtitles_into_frames(frame_timestamps, subtitles, doc["starting_timestamp_for_subtitles"], doc["duration"])
        return f"{pre_prompt}{interleaved_prefix}\n{question}\n{post_prompt}"
    else:
        return f"{pre_prompt}{question}\n{post_prompt}"


hf_home = os.getenv("HF_HOME", "~/.cache/huggingface/")
base_cache_dir = os.path.expanduser(hf_home)


def longvideobench_doc_to_visual_v(doc):
    with open(Path(__file__).parent / "longvideobench_val_v.yaml", "r") as f:
        raw_data = f.readlines()
        safe_data = []
        for i, line in enumerate(raw_data):
            # remove function definition since yaml load cannot handle it
            if "!function" not in line:
                safe_data.append(line)
    cache_name = yaml.safe_load("".join(safe_data))["dataset_kwargs"]["cache_dir"]
    vid_subdir_name = yaml.safe_load("".join(safe_data))["dataset_kwargs"].get("video_subdir", "videos/")
    cache_dir = os.path.join(base_cache_dir, cache_name, vid_subdir_name)
    video_path = doc["video_path"]
    video_path = os.path.join(cache_dir, video_path)
    return [video_path]


def longvideobench_doc_to_messages_v(doc, lmms_eval_specific_kwargs=None):
    question = longvideobench_doc_to_text(doc, lmms_eval_specific_kwargs)
    visuals = longvideobench_doc_to_visual_v(doc)
    system_messages = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]}]
    messages = [{"role": "user", "content": []}]
    messages[0]["content"].append({"type": "video", "url": visuals[0]})
    messages[0]["content"].append({"type": "text", "text": question.strip()})
    messages = system_messages + messages
    return messages


def longvideobench_doc_to_text_no_visual(doc, lmms_eval_specific_kwargs=None):
    candidates = []
    for i in range(5):
        candidate = doc.get(f"option{i}")
        if candidate != "N/A":
            candidates.append(candidate)

    question = doc["question"] + "\n" + "\n".join([". ".join([chr(ord("A") + i), candidate]) for i, candidate in enumerate(candidates)])
    return question + " Answer with the option letter directly."


def longvideobench_doc_to_messages_no_visual(doc, lmms_eval_specific_kwargs=None):
    question = longvideobench_doc_to_text_no_visual(doc, lmms_eval_specific_kwargs)
    messages = [{"role": "user", "content": []}]
    messages[0]["content"].append({"type": "text", "text": question.strip()})
    return messages


def longvideobench_process_docs(dataset):
    def map(example):
        candidates = []
        for i in range(5):
            candidate = example.get(f"option{i}")
            if candidate != "N/A":
                candidates.append(candidate)
        
        correct_choice = int(example['correct_choice'])
        correct_candidate = candidates[correct_choice]
        
        # Shuffle candidates
        random.shuffle(candidates)
        
        # Find new position of correct choice after shuffling
        new_correct_choice = candidates.index(correct_candidate)
        
        # Update the example with shuffled options
        return_dict = {}
        for i, candidate in enumerate(candidates):
            return_dict[f"option{i}"] = candidate
        
        # Fill remaining options with "N/A"
        for i in range(len(candidates), 5):
            return_dict[f"option{i}"] = "N/A"
        
        return_dict["correct_choice"] = new_correct_choice
        return return_dict
    new_dataset = dataset.map(map)
    return new_dataset

def longvideobench_doc_to_messages_random_choice(doc, lmms_eval_specific_kwargs=None):
    question = longvideobench_doc_to_text(doc, lmms_eval_specific_kwargs)
    visuals = longvideobench_doc_to_visual_v(doc)
    messages = [{"role": "user", "content": []}]
    messages[0]["content"].append({"type": "video", "url": visuals[0]})
    messages[0]["content"].append({"type": "text", "text": question.strip() + " Answer with the option letter directly."})
    return messages


def longvideobench_tool_doc_to_messages(doc, lmms_eval_specific_kwargs=None):
    question = longvideobench_doc_to_text(doc, lmms_eval_specific_kwargs)
    visuals = longvideobench_doc_to_visual_v(doc)
    messages = [{"role": "user", "content": []}]
    messages[0]["content"].append({"type": "video", "url": visuals[0]})
    messages[0]["content"].append({"type": "text", "text": question.strip() + " " + TOOL_PROMPT})
    return messages


def longvideobench_process_results(doc, results):
    acc_score = 0
    format_score = 0
    question = longvideobench_doc_to_text(doc, lmms_eval_specific_kwargs={"pre_prompt": "", "post_prompt": ""})
    extra_info = {"question": question}
    answer = ['A', 'B', 'C', 'D', 'E'][doc['correct_choice']]
    for pred in results:
        score_dict = compute_score(data_source="longvideobench", solution_str=pred.strip(), ground_truth=answer, extra_info=extra_info)
        acc_score += score_dict["acc_score"]
        format_score += score_dict.get("format_reward_score", 0.0)

    return {"acc_score": acc_score / len(results) if results else 0.0, "format_score": format_score / len(results) if results else 0.0}

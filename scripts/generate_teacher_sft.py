#!/usr/bin/env python3
"""
Generate teacher trajectories with rejection sampling for temporal grounding.

Features:
- OpenAI-compatible API via openai library
- Multi-process generation
- Resume from existing output
- Streaming writes with progress bar

Example:
  python scripts/generate_teacher_sft.py \
    --input /path/to/input.jsonl \
    --output /path/to/output.jsonl \
    --model gemini-2.5-flash \
    --api-base http://localhost:8000/v1 \
    --api-key EMPTY
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import base64
import json
import math
import os
import re
import sys
import time
import mimetypes
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import OpenAI
from tqdm import tqdm


SYSTEM_PROMPT = """You are a helpful assistant for long-video understanding and reasoning.

You may call tools to inspect video segments. Use the tool only when necessary.

<tools>
{"type":"function","function":{"name":"crop_video","description":"Crop a video to a specified duration (return the exact start/end timestamps you selected; no images).","parameters":{"type":"object","properties":{"video_path":{"type":"string","description":"Path to the video file"},"start_time":{"type":"number","description":"Start time in seconds"},"end_time":{"type":"number","description":"End time in seconds, must be > start_time"}},"required":["video_path","start_time","end_time"]},"strict":false}}
</tools>

For every function call, wrap a JSON object with the function name and its arguments inside <tool_call></tool_call> tags.
Do NOT output any <tool_response> tags. The tool response will be injected by the system after execution.
The JSON inside <tool_call> must include a "name" key and an "arguments" object.
You will receive a series of image frames sampled from the video (at most 512 frames) for reference.

Input you receive:
VIDEO_PATH: {video_path}
GROUND_TRUTH_TIME: [{gt_start:.3f}, {gt_end:.3f}]
GROUND_TRUTH_ANSWER: {gt_answer}

Thinking requirements:
- Each <think> must be non-empty prose (3–6 sentences) with clear evidence and integration.
- Mention time anchors in natural language when you refer to video evidence.
- Use plain ASCII punctuation; avoid placeholders and gibberish.

We will follow a coarse-to-fine multi-stage approach:
Phase 1 (global skim & planning — first <think> block):
- Reconstruct the visual storyline of the entire video by interpreting the sequence of provided frames (silent video). Do not mention that you are looking at static images or frames; narrate it as a continuous video scene.
- In ≈ 4–6 flowing sentences, narrate what the camera shows across the whole video (settings, actors, transitions).
- Timestamp during thinking: As you narrate, sprinkle human-readable time anchors for key moments (not only the final windows). Allowed styles include: ≈297s, around 298–300s, from 4:56 to 5:15, 295–300s, or [296.34s – 320.76s].

First-turn decision rule:
- Think first. If you need to call a tool, output exactly one <tool_call> and stop.
- If you already have enough evidence to answer, output a single answer (no explanations) and stop.

Output format:
<think>...</think>
<tool_call>...</tool_call>
<answer>YOUR_FINAL_ANSWER</answer>
"""

TRAJECTORY_SYSTEM_PROMPT = """You are a helpful assistant.
# Tools
You may call one or more functions to assist with the user query.
You are provided with function signatures within <tools></tools> XML tags:
<tools>{"type": "function", "function": {"name": "crop_video", "description": "Crop a video to a specified duration.", "parameters": {"type": "object", "properties": {"video_path": {"type": "string", "description": "Path to the video file", "enum": null}, "start_time": {"type": "number", "description": "Start time in seconds", "enum": null}, "end_time": {"type": "number", "description": "End time in seconds, must be > start_time", "enum": null}}, "required": []}, "strict": false}}
{"type": "function", "function": {"name": "get_frame", "description": "Extract a single frame from a video at a specified timestamp.", "parameters": {"type": "object", "properties": {"video_path": {"type": "string", "description": "Path to the video file", "enum": null}, "timestamp": {"type": "number", "description": "Timestamp in seconds for the desired frame", "enum": null}}, "required": []}, "strict": false}}</tools>
For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>{"name": <function-name>, "arguments": <args-json-object>}</tool_call>"""

TOOL_RESPONSE_OK = "The tool executed successfully."

_FRAME_CACHE: Dict[str, Tuple[List[Dict[str, Any]], int]] = {}


USER_TEMPLATE = """QUESTION: {question}"""

TRAJECTORY_USER_TEMPLATE = (
    "{question} Think first, call **crop_video** or **get_frame** if needed, then answer. "
    "Format strictly as:  <think>...</think>  <tool_call>...</tool_call> "
    "(if tools needed)  <answer>...</answer>. The Video path for this video is: {video_path}"
)


FINE_INSPECTION_TEMPLATE = """You are now in Phase 2 (fine-grained inspection), round {round_idx}.

Continue the existing response without repeating earlier content. First append exactly one <think> block,
then decide whether to output a <tool_call> or a final <answer>. Use 3–6 sentences in <think> with evidence,
integration, and reflection. Mention time anchors in natural language. If evidence is sufficient, output a
complete <answer>...</answer> block (with both opening and closing tags) and stop; otherwise output exactly
one <tool_call>...</tool_call> block and stop. If you still cannot answer after a crop, keep calling the tool
with new segments until you can answer.

This round includes:
- Attached frames: images from the video segment of this interval (low resolution, ~224px).
- The original QUESTION (for reference): {question}

In the <think> block you append this round, include three parts (as prose, not bullet labels):
1) Evidence: what this window shows that helps answer the question.
2) Integration: how this confirms or revises your earlier hypothesis (mark outdated bits as "revised: …").
3) Self-reflection: whether this window was mis-localized; if so, how you would correct it; otherwise note
   that it suffices for its subgoal.
"""

JUDGE_PROMPT_GROUNDING = """You are a strict judge for temporal grounding.

Question:
{question}

Ground truth time range: [{gt_start:.3f}, {gt_end:.3f}]
Predicted time range: [{pred_start:.3f}, {pred_end:.3f}]

Decide if the predicted time range correctly answers the question.
Respond with only one token: YES or NO.
"""

JUDGE_PROMPT_VQA = """You are a strict judge for VQA.

Question:
{question}

Ground truth answer: {gt_answer}
Predicted answer: {pred_answer}

Decide if the predicted answer is semantically consistent with the ground truth.
Respond with only one token: YES or NO.
"""


@dataclass
class GenerationConfig:
    model: str
    api_base: Optional[str]
    api_key: str
    max_tokens: int
    temperature: float
    timeout_s: Optional[float]
    mcp_server_path: Optional[str]
    tool_output_dir: Optional[str]
    fps: int
    max_frames: int
    max_pixels: int
    max_payload_mb: float
    frame_output_dir: Optional[str]


def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def read_input(path: str) -> Iterable[Dict[str, Any]]:
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                f.seek(0)
                yield from read_jsonl(path)
                return
        if isinstance(data, list):
            for item in data:
                yield item
        else:
            yield data
        return
    yield from read_jsonl(path)


def render_system_prompt(prompt: str, **kwargs: Any) -> str:
    rendered = prompt
    for key, value in kwargs.items():
        rendered = rendered.replace(f"{{{key}}}", str(value))
    return rendered


def load_existing_ids(path: str) -> set[str]:
    if not os.path.exists(path):
        return set()
    ids: set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                clip_id = data.get("clip_id")
                if not clip_id and isinstance(data.get("metadata"), dict):
                    clip_id = data["metadata"].get("clip_id")
                if not clip_id:
                    clip_id = data.get("id")
                if clip_id:
                    ids.add(clip_id)
            except json.JSONDecodeError:
                continue
    return ids


def parse_tool_calls(text: str, tool_name: str) -> Optional[Dict[str, Any]]:
    tool_call_pattern = r"<tool_call>(.*?)</tool_call>"
    tool_calls = re.findall(tool_call_pattern, text, re.DOTALL)
    for tool_call in reversed(tool_calls):
        tool_call = tool_call.strip()
        try:
            data = json.loads(tool_call)
        except json.JSONDecodeError:
            try:
                data = ast.literal_eval(tool_call)
            except (ValueError, SyntaxError):
                continue
        if isinstance(data, dict) and data.get("name") == tool_name:
            return data.get("arguments", {})
    return None


def parse_answer_interval(text: str) -> Optional[Tuple[float, float]]:
    m = re.search(r"<answer>(.*?)</answer>", text, flags=re.S | re.I)
    if not m:
        return None
    content = m.group(1).strip()
    try:
        interval = ast.literal_eval(content)
        if not isinstance(interval, list) or len(interval) != 2:
            return None
        return float(interval[0]), float(interval[1])
    except (ValueError, SyntaxError, TypeError):
        return None


def parse_answer_text(text: str) -> Optional[str]:
    m = re.search(r"<answer>(.*?)</answer>", text, flags=re.S | re.I)
    if not m:
        return None
    return m.group(1).strip()


def parse_gt_interval(answer: Any) -> Optional[Tuple[float, float]]:
    if isinstance(answer, list) and len(answer) == 2:
        try:
            return float(answer[0]), float(answer[1])
        except (ValueError, TypeError):
            return None
    if isinstance(answer, str):
        try:
            interval = ast.literal_eval(answer)
            if isinstance(interval, list) and len(interval) == 2:
                return float(interval[0]), float(interval[1])
        except (ValueError, SyntaxError, TypeError):
            return None
    return None


def _get_cached_frames(
    video_path: str,
    config: GenerationConfig,
    clip_id: str,
) -> Tuple[List[Dict[str, Any]], List[str], int]:
    print(f"[INFO] encode frames start clip_id={clip_id} video_path={video_path}", flush=True)
    cached = _FRAME_CACHE.get(video_path)
    if cached:
        frame_contents, total_bytes = cached
        return frame_contents, [], total_bytes
    frame_contents, frame_paths, total_bytes = encode_video_frames(
        video_path,
        fps=config.fps,
        max_frames=config.max_frames,
        max_pixels=config.max_pixels,
        max_payload_mb=config.max_payload_mb,
        output_dir=config.frame_output_dir,
        clip_id=clip_id,
    )
    _FRAME_CACHE[video_path] = (frame_contents, total_bytes)
    return frame_contents, frame_paths, total_bytes


def build_messages(
    sample: Dict[str, Any],
    config: GenerationConfig,
) -> Tuple[List[Dict[str, Any]], List[str], int, str]:
    print(
        "[INFO] build_messages start clip_id="
        f"{sample.get('clip_id')} video_path={sample.get('video_path')}",
        flush=True,
    )
    qa_start = float(sample.get("qa_start_time", sample.get("clip_start_time", 0.0)))
    qa_end = float(sample.get("qa_end_time", sample.get("clip_end_time", 0.0)))
    gt_answer = sample.get("answer")
    user_text = USER_TEMPLATE.format(
        question=sample["question"],
    )
    system_prompt = render_system_prompt(
        SYSTEM_PROMPT,
        video_path=sample["video_path"],
        gt_start=f"{qa_start:.3f}",
        gt_end=f"{qa_end:.3f}",
        gt_answer=gt_answer,
    )
    frame_contents, frame_paths, total_bytes = _get_cached_frames(
        sample["video_path"],
        config,
        sample["clip_id"],
    )
    user_content = [{"type": "text", "text": user_text}] + frame_contents
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ], frame_paths, total_bytes, user_text


def encode_video_frames(
    video_path: str,
    fps: int,
    max_frames: int,
    max_pixels: int,
    max_payload_mb: float,
    output_dir: Optional[str],
    clip_id: str,
) -> Tuple[List[Dict[str, Any]], List[str], int]:
    import cv2
    import torch
    from qwen_vl_utils import fetch_video
    from torchvision.transforms.functional import to_pil_image

    cap = cv2.VideoCapture(video_path)
    duration = None
    if cap.isOpened():
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
        if video_fps > 0 and total_frames > 0:
            duration = total_frames / video_fps
    cap.release()

    if duration and duration > 0:
        if duration >= 512:
            target_max_frames = min(max_frames, 512)
            target_fps = target_max_frames / duration
        else:
            target_fps = 1.0
            target_max_frames = min(max_frames, max(1, int(math.ceil(duration))))
    else:
        target_fps = float(fps)
        target_max_frames = max_frames

    max_payload_bytes = int(max_payload_mb * 1024 * 1024)
    current_max_frames = target_max_frames
    while True:
        if duration and duration > 0:
            effective_fps = min(target_fps, current_max_frames / duration)
        else:
            effective_fps = target_fps
        video_ele = {
            "type": "video",
            "video": f"file://{video_path}",
            "fps": effective_fps,
            "min_frames": 1,
            "max_frames": current_max_frames,
            "min_pixels": 28 * 28,
            "max_pixels": max_pixels,
        }
        video_frames = fetch_video(video_ele)
        video_frames = video_frames.to(torch.uint8)
        images = [to_pil_image(frame) for frame in video_frames]

        image_contents = []
        total_bytes = 0
        for img in images:
            from io import BytesIO

            buffer = BytesIO()
            img.save(buffer, format="PNG")
            raw = buffer.getvalue()
            total_bytes += len(raw)
            b64 = base64.b64encode(raw).decode("utf-8")
            image_contents.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})

        if total_bytes <= max_payload_bytes or current_max_frames <= 1:
            frame_paths = []
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                clip_dir = os.path.join(output_dir, clip_id)
                os.makedirs(clip_dir, exist_ok=True)
                for idx, img in enumerate(images, start=1):
                    frame_path = os.path.join(clip_dir, f"frame_{idx:03d}.png")
                    img.save(frame_path, format="PNG")
                    frame_paths.append(frame_path)
            return image_contents, frame_paths, total_bytes

        reduction = max(1, current_max_frames // 10)
        current_max_frames = max(1, current_max_frames - reduction)


def _validate_response(text: str) -> bool:
    tool_call_pattern = r"<tool_call>(.*?)</tool_call>"
    tool_calls = re.findall(tool_call_pattern, text, re.DOTALL)
    if not tool_calls:
        return "<answer>" in text and "</answer>" in text
    for tool_call in tool_calls:
        tool_call = tool_call.strip()
        try:
            data = json.loads(tool_call)
        except json.JSONDecodeError:
            try:
                data = ast.literal_eval(tool_call)
            except (ValueError, SyntaxError):
                return False
        if not isinstance(data, dict) or "name" not in data or "arguments" not in data:
            return False
    return True


def call_model(client: OpenAI, config: GenerationConfig, messages: List[Dict[str, Any]]) -> str:
    delay_s = 1.0
    while True:
        try:
            resp = client.chat.completions.create(
                model=config.model,
                messages=messages,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                timeout=config.timeout_s,
            )
            content = resp.choices[0].message.content or ""
            if _validate_response(content):
                return content
            print("[WARN] Invalid response format. Retrying...", file=sys.stderr)
        except Exception as exc:
            print(f"[WARN] API call failed: {exc}. Retrying in {delay_s:.1f}s...", file=sys.stderr)
            time.sleep(delay_s)
            delay_s = min(delay_s * 1.5, 30.0)


async def _call_mcp_tool(server_path: str, tool_name: str, tool_args: Dict[str, Any]):
    server_params = StdioServerParameters(command="python", args=[server_path])
    async with stdio_client(server=server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            return await session.call_tool(tool_name, tool_args)


def run_mcp_tool(server_path: str, tool_name: str, tool_args: Dict[str, Any]):
    delay_s = 1.0
    while True:
        try:
            return asyncio.run(_call_mcp_tool(server_path, tool_name, tool_args))
        except Exception as exc:
            print(f"[WARN] MCP tool call failed: {exc}. Retrying in {delay_s:.1f}s...", file=sys.stderr)
            time.sleep(delay_s)
            delay_s = min(delay_s * 1.5, 30.0)


def _strip_tool_responses(text: str) -> str:
    return re.sub(r"<tool_response>.*?</tool_response>", "", text, flags=re.DOTALL)


def apply_tool_responses(
    text: str,
    tool_name: str,
    tool_runs: List[Dict[str, Any]],
    config: GenerationConfig,
    sample: Dict[str, Any],
    turn_idx: int,
) -> Tuple[str, List[Dict[str, Any]]]:
    tool_call_pattern = r"(<tool_call>.*?</tool_call>)"
    tool_calls = re.findall(tool_call_pattern, text, re.DOTALL)
    if not tool_calls:
        return text, []

    cleaned = _strip_tool_responses(text)
    response_runs: List[Dict[str, Any]] = []
    for tool_call in tool_calls:
        tool_args = None
        inner = re.search(r"<tool_call>(.*?)</tool_call>", tool_call, re.DOTALL)
        if inner:
            try:
                tool_args = json.loads(inner.group(1).strip())
            except json.JSONDecodeError:
                try:
                    tool_args = ast.literal_eval(inner.group(1).strip())
                except (ValueError, SyntaxError):
                    tool_args = None
        tool_response_text = TOOL_RESPONSE_OK
        success = False
        image_count = 0
        saved_images: List[str] = []
        if (
            tool_args
            and isinstance(tool_args, dict)
            and tool_args.get("name") == tool_name
            and isinstance(tool_args.get("arguments"), dict)
            and config.mcp_server_path
        ):
            try:
                result = run_mcp_tool(config.mcp_server_path, tool_name, tool_args["arguments"])
                image_parts = [part for part in result.content if part.type == "image"]
                image_count = len(image_parts)
                if config.tool_output_dir:
                    saved_images = save_tool_images(
                        config.tool_output_dir,
                        sample["clip_id"],
                        tool_name,
                        image_parts,
                        turn_idx,
                    )
                success = True
            except Exception as exc:
                tool_response_text = f"Tool execution failed: {exc}"
        run_record = {
            "name": tool_name,
            "arguments": tool_args["arguments"] if tool_args and "arguments" in tool_args else {},
            "image_count": image_count,
            "saved_images": saved_images,
            "success": success,
            "response_text": tool_response_text,
        }
        tool_runs.append(run_record)
        response_runs.append(run_record)
    return cleaned, response_runs


def save_tool_images(
    output_dir: str,
    clip_id: str,
    tool_name: str,
    image_contents: List[Any],
    turn_idx: int,
) -> List[str]:
    os.makedirs(output_dir, exist_ok=True)
    clip_dir = os.path.join(output_dir, clip_id, f"turn_{turn_idx:02d}")
    os.makedirs(clip_dir, exist_ok=True)
    saved = []
    for idx, part in enumerate(image_contents, start=1):
        if getattr(part, "type", None) != "image":
            continue
        data = getattr(part, "data", None)
        if not data:
            continue
        image_path = os.path.join(clip_dir, f"{tool_name}_{idx:03d}.png")
        with open(image_path, "wb") as f:
            f.write(base64.b64decode(data))
        saved.append(image_path)
    return saved


def _append_assistant(messages: List[Dict[str, Any]], text: str) -> None:
    messages.append({"role": "assistant", "content": text})


def _append_user(messages: List[Dict[str, Any]], text: str) -> None:
    messages.append({"role": "user", "content": text})


def _strip_previous_tool_responses(messages: List[Dict[str, Any]]) -> None:
    messages[:] = [
        message
        for message in messages
        if not (
            message.get("role") == "user"
            and isinstance(message.get("content"), list)
            and any(
                part.get("type") == "text"
                and str(part.get("text", "")).lstrip().startswith("<tool_response>")
                for part in message["content"]
            )
        )
    ]


def _build_tool_response_content(
    tool_runs: List[Dict[str, Any]],
    *,
    encode_images: bool,
) -> List[Dict[str, Any]]:
    response_text = "\n".join(
        run.get("response_text", TOOL_RESPONSE_OK) for run in tool_runs
    )
    content: List[Dict[str, Any]] = [
        {"type": "text", "text": f"<tool_response>\n{response_text}"},
    ]
    for run in tool_runs:
        for image_path in run.get("saved_images", []):
            if encode_images:
                data_url = _image_file_to_data_url(image_path)
                if data_url:
                    content.append({"type": "image_url", "image_url": {"url": data_url}})
            else:
                content.append({"type": "image_url", "image_url": {"url": image_path}})
    content.append({"type": "text", "text": "</tool_response>"})
    return content


def _image_file_to_data_url(image_path: str) -> Optional[str]:
    if not image_path or not os.path.exists(image_path):
        return None
    mime_type, _ = mimetypes.guess_type(image_path)
    if not mime_type:
        mime_type = "image/png"
    with open(image_path, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def build_longvt_output(
    sample: Dict[str, Any],
    response_steps: List[Tuple[str, List[Dict[str, Any]]]],
    transcript: str,
    frame_paths: List[str],
    frame_bytes: int,
) -> Dict[str, Any]:
    user_text = TRAJECTORY_USER_TEMPLATE.format(
        question=sample["question"],
        video_path=sample["video_path"],
    )
    system_prompt = TRAJECTORY_SYSTEM_PROMPT
    messages: List[Dict[str, Any]] = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}],
        },
        {
            "role": "user",
            "content": [
                {"type": "video_url", "video_url": {"url": sample["video_path"]}},
                {"type": "text", "text": user_text},
            ],
        },
    ]
    for response_text, tool_runs in response_steps:
        messages.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": response_text}],
            }
        )
        if tool_runs:
            messages.append(
                {
                    "role": "user",
                    "content": _build_tool_response_content(tool_runs, encode_images=False),
                }
            )

    return {
        "id": sample["clip_id"],
        "messages": messages,
        "metadata": {
            "clip_id": sample["clip_id"],
            "video_name": sample.get("video_name"),
            "video_path": sample.get("video_path"),
            "question": sample.get("question"),
            "answer_gt": sample.get("answer"),
            "transcript": transcript,
            "frame_paths": frame_paths,
            "frame_bytes": frame_bytes,
        },
    }


def generate_sample(sample: Dict[str, Any], config: GenerationConfig) -> Dict[str, Any]:
    client = OpenAI(api_key=config.api_key, base_url=config.api_base)
    messages, frame_paths, frame_bytes, user_text = build_messages(sample, config)
    qa_start = sample.get("qa_start_time", sample.get("clip_start_time"))
    qa_end = sample.get("qa_end_time", sample.get("clip_end_time"))
    gt_answer = sample.get("answer")
    gt_answer_interval = parse_gt_interval(gt_answer)
    is_grounding = gt_answer_interval is not None

    transcript = ""
    tool_runs: List[Dict[str, Any]] = []
    response_steps: List[Tuple[str, List[Dict[str, Any]]]] = []

    while True:
        response = call_model(client, config, messages)
        transcript += response

        answer_interval = parse_answer_interval(transcript)
        answer_text = parse_answer_text(transcript)
        if is_grounding and answer_interval:
            response_steps.append((response, []))
            _append_assistant(messages, response)
            return build_longvt_output(
                sample,
                response_steps,
                transcript,
                frame_paths,
                frame_bytes,
            )
        if not is_grounding and answer_text:
            response_steps.append((response, []))
            _append_assistant(messages, response)
            return build_longvt_output(
                sample,
                response_steps,
                transcript,
                frame_paths,
                frame_bytes,
            )

        tool_args = parse_tool_calls(transcript, "crop_video")
        if tool_args:
            break

        print(
            "[WARN] Response missing <answer> and <tool_call>. Retrying...",
            file=sys.stderr,
        )
        transcript = ""

    response_clean, response_tool_runs = apply_tool_responses(
        response,
        "crop_video",
        tool_runs,
        config,
        sample,
        1,
    )
    response_steps.append((response_clean, response_tool_runs))
    transcript = response_clean
    _append_assistant(messages, response_clean)
    if len(messages) > 1 and isinstance(messages[1].get("content"), list):
        messages[1]["content"] = [{"type": "text", "text": user_text}]
    if response_tool_runs:
        _strip_previous_tool_responses(messages)
        messages.append(
            {
                "role": "user",
                "content": _build_tool_response_content(response_tool_runs, encode_images=True),
            }
        )

    max_rounds = 2
    rounds_used = 1
    while rounds_used < max_rounds:
        round_prompt = FINE_INSPECTION_TEMPLATE.format(
            round_idx=rounds_used + 1,
            question=sample["question"],
        )
        _append_user(messages, round_prompt)
        response = call_model(client, config, messages)
        response_clean, response_tool_runs = apply_tool_responses(
            response,
            "crop_video",
            tool_runs,
            config,
            sample,
            rounds_used + 1,
        )
        response_steps.append((response_clean, response_tool_runs))
        transcript += "\n" + response_clean
        _append_assistant(messages, response_clean)
        if response_tool_runs:
            _strip_previous_tool_responses(messages)
            messages.append(
                {
                    "role": "user",
                    "content": _build_tool_response_content(response_tool_runs, encode_images=True),
                }
            )

        answer_interval = parse_answer_interval(transcript)
        answer_text = parse_answer_text(transcript)
        if is_grounding and answer_interval:
            return build_longvt_output(
                sample,
                response_steps,
                transcript,
                frame_paths,
                frame_bytes,
            )
        if not is_grounding and answer_text:
            return build_longvt_output(
                sample,
                response_steps,
                transcript,
                frame_paths,
                frame_bytes,
            )
        rounds_used += 1

    return build_longvt_output(
        sample,
        response_steps,
        transcript,
        frame_paths,
        frame_bytes,
    )


def worker(entry: Dict[str, Any], config_dict: Dict[str, Any]) -> Dict[str, Any]:
    config = GenerationConfig(**config_dict)
    clip_id = entry.get("clip_id")
    video_path = entry.get("video_path")
    print(f"[INFO] worker start clip_id={clip_id} video_path={video_path}", flush=True)
    try:
        return generate_sample(entry, config)
    except Exception as exc:
        return {
            "clip_id": clip_id,
            "error": str(exc),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Teacher trajectory generation with rejection sampling.")
    parser.add_argument("--input", required=True, help="Input JSONL file.")
    parser.add_argument("--output", required=True, help="Output JSONL file.")
    parser.add_argument("--model", required=True, help="Model name.")
    parser.add_argument("--api-base", default=None, help="OpenAI-compatible base URL.")
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", "EMPTY"), help="API key.")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Max tokens per request.")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature.")
    parser.add_argument("--timeout", type=float, default=None, help="Request timeout in seconds.")
    parser.add_argument("--workers", type=int, default=4, help="Number of processes.")
    parser.add_argument("--mcp-server-path", default=None, help="Path to MCP server script for tool calls.")
    parser.add_argument("--tool-output-dir", default=None, help="Directory to save tool images.")
    parser.add_argument("--fps", type=int, default=1, help="Frames per second for video sampling.")
    parser.add_argument("--max-frames", type=int, default=512, help="Max frames to sample from the video.")
    parser.add_argument("--max-pixels", type=int, default=50176, help="Max pixels per frame (e.g. 224*224=50176).")
    parser.add_argument("--max-payload-mb", type=float, default=20.0, help="Max total frame payload size (MB).")
    parser.add_argument("--frame-output-dir", default=None, help="Directory to save sampled frames.")
    parser.add_argument(
        "--no-validate-first",
        action="store_true",
        help="Skip validating the first generated trajectory format.",
    )
    parser.add_argument(
        "--preload-frames",
        dest="preload_frames",
        action="store_true",
        help="Preload base64 frames for all videos before generation.",
    )
    parser.add_argument(
        "--no-preload-frames",
        dest="preload_frames",
        action="store_false",
        help="Disable preloading base64 frames before generation.",
    )
    parser.set_defaults(preload_frames=False)
    parser.add_argument(
        "--executor",
        choices=("process", "thread", "inline"),
        default="process",
        help="Execution backend for workers.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    existing_ids = load_existing_ids(args.output)

    config = GenerationConfig(
        model=args.model,
        api_base=args.api_base,
        api_key=args.api_key,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        timeout_s=args.timeout,
        mcp_server_path=args.mcp_server_path,
        tool_output_dir=args.tool_output_dir,
        fps=args.fps,
        max_frames=args.max_frames,
        max_pixels=args.max_pixels,
        max_payload_mb=args.max_payload_mb,
        frame_output_dir=args.frame_output_dir,
    )

    if args.preload_frames:
        print("[INFO] Preloading base64 frames for input videos...")
        for entry in read_input(args.input):
            if entry.get("clip_id") in existing_ids:
                continue
            _get_cached_frames(entry["video_path"], config, entry["clip_id"])
        print(f"[INFO] Preloaded {len(_FRAME_CACHE)} videos.")

    entries = [e for e in read_input(args.input) if e.get("clip_id") not in existing_ids]
    validate_first = not args.no_validate_first

    if not entries:
        print("No new entries to process.")
        return

    with open(args.output, "a", encoding="utf-8") as out_f:
        with tqdm(total=len(entries), desc="Generating") as progress:
            if args.executor == "inline":
                first_checked = False
                for entry in entries:
                    result = worker(entry, config.__dict__)
                    if validate_first and not first_checked:
                        transcript = result.get("metadata", {}).get("transcript", "")
                        if transcript and not _validate_response(transcript):
                            result["error"] = "invalid first trajectory format"
                            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                            out_f.flush()
                            print("[ERROR] First trajectory failed validation; aborting.", file=sys.stderr)
                            return
                        first_checked = True
                    out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    out_f.flush()
                    progress.update(1)
            else:
                executor_cls = ProcessPoolExecutor
                if args.executor == "thread":
                    executor_cls = ThreadPoolExecutor
                with executor_cls(max_workers=args.workers) as executor:
                    futures = [
                        executor.submit(worker, entry, config.__dict__)
                        for entry in entries
                    ]
                    first_checked = False
                    for future in as_completed(futures):
                        result = future.result()
                        if validate_first and not first_checked:
                            transcript = result.get("metadata", {}).get("transcript", "")
                            if transcript and not _validate_response(transcript):
                                result["error"] = "invalid first trajectory format"
                                out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                                out_f.flush()
                                print(
                                    "[ERROR] First trajectory failed validation; aborting.",
                                    file=sys.stderr,
                                )
                                return
                            first_checked = True
                        out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                        out_f.flush()
                        progress.update(1)


if __name__ == "__main__":
    main()

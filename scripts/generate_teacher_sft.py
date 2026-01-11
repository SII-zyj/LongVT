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
import os
import re
import sys
import time
import mimetypes
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

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

First-turn decision rule:
- Think first. If you need to call a tool, output exactly one <tool_call> and stop (no <answer> in that response).
- If you already have enough evidence to answer, output a single <answer>...</answer> and stop (no <tool_call>).

Thinking requirements:
- Each <think> must be non-empty prose (3–6 sentences) with clear evidence and integration.
- Mention time anchors in natural language when you refer to video evidence.
- Use plain ASCII punctuation; avoid placeholders and gibberish.
- If you output a <tool_call> in the first round, do not output <answer> in that same response.
- In the first round, either output a <tool_call> (no <answer>), or output a complete <answer>...</answer> block.

We will follow a coarse-to-fine multi-stage approach:
Phase 1 (global skim & planning — first <think> block):
- Reconstruct the visual storyline of the entire video by interpreting the sequence of provided frames (silent video). Do not mention that you are looking at static images or frames; narrate it as a continuous video scene.
- In ≈ 4–6 flowing sentences, narrate what the camera shows across the whole video (settings, actors, transitions).

Output format:
<think>...</think>
<tool_call>...</tool_call>
<answer>...</answer>

Repeat <think> and <tool_call> until you have enough evidence to answer, then output <answer>.
If no tool is needed, omit <tool_call>.
"""

TOOL_RESPONSE_OK = "The tool executed successfully."

_FRAME_CACHE: Dict[str, Tuple[List[Dict[str, Any]], int]] = {}


USER_TEMPLATE = """You are now in Phase 1 (global skim & planning). Follow the system instructions.
VIDEO_PATH: {video_path}
QUESTION: {question}"""


HINT_TEMPLATE = """Hint: The relevant time range is [{gt_start:.3f}, {gt_end:.3f}].
Do not mention this hint in your reasoning. Continue the trajectory and reach a final <answer>.
"""

FINE_INSPECTION_TEMPLATE = """You are now in Phase 2 (fine-grained inspection), round {round_idx}.

Continue the existing response without repeating earlier content. Append exactly one <tool_call> block,
then exactly one <think> block. Use 3–6 sentences in <think> with evidence and integration. Mention
time anchors in natural language. Only output <answer> when you have enough evidence.
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
    overlap_threshold: float
    max_corrections: int
    answer_iou_threshold: float
    judge_model: Optional[str]
    judge_api_base: Optional[str]
    judge_api_key: Optional[str]
    judge_max_tokens: int
    judge_temperature: float
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


def overlap_ratio(pred: Tuple[float, float], gt: Tuple[float, float]) -> float:
    pred_start, pred_end = pred
    gt_start, gt_end = gt
    intersection = max(0.0, min(pred_end, gt_end) - max(pred_start, gt_start))
    gt_len = max(0.0, gt_end - gt_start)
    if gt_len <= 0:
        return 0.0
    return intersection / gt_len


def iou_ratio(pred: Tuple[float, float], gt: Tuple[float, float]) -> float:
    pred_start, pred_end = pred
    gt_start, gt_end = gt
    intersection = max(0.0, min(pred_end, gt_end) - max(pred_start, gt_start))
    union = max(pred_end, gt_end) - min(pred_start, gt_start)
    if union <= 0:
        return 0.0
    return intersection / union


def _get_cached_frames(
    video_path: str,
    config: GenerationConfig,
    clip_id: str,
) -> Tuple[List[Dict[str, Any]], List[str], int]:
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
    user_text = USER_TEMPLATE.format(
        video_path=sample["video_path"],
        clip_start=float(sample["clip_start_time"]),
        clip_end=float(sample["clip_end_time"]),
        question=sample["question"],
    )
    frame_contents, frame_paths, total_bytes = _get_cached_frames(
        sample["video_path"],
        config,
        sample["clip_id"],
    )
    user_content = [{"type": "text", "text": user_text}] + frame_contents
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
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
    import torch
    from qwen_vl_utils import fetch_video
    from torchvision.transforms.functional import to_pil_image

    max_payload_bytes = int(max_payload_mb * 1024 * 1024)
    current_max_frames = max_frames
    while True:
        video_ele = {
            "type": "video",
            "video": f"file://{video_path}",
            "fps": fps,
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

        current_max_frames = max(1, current_max_frames // 2)


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


def judge_answer(
    config: GenerationConfig,
    question: str,
    gt_answer: str,
    pred_answer: str,
) -> bool:
    if not config.judge_model:
        return False
    client = OpenAI(
        api_key=config.judge_api_key or config.api_key,
        base_url=config.judge_api_base or config.api_base,
    )
    prompt = JUDGE_PROMPT_VQA.format(
        question=question,
        gt_answer=gt_answer,
        pred_answer=pred_answer,
    )
    messages = [{"role": "system", "content": prompt}]
    delay_s = 1.0
    while True:
        try:
            resp = client.chat.completions.create(
                model=config.judge_model,
                messages=messages,
                temperature=config.judge_temperature,
                max_tokens=config.judge_max_tokens,
                timeout=config.timeout_s,
            )
            content = (resp.choices[0].message.content or "").strip().upper()
            return content.startswith("YES")
        except Exception as exc:
            print(f"[WARN] Judge call failed: {exc}. Retrying in {delay_s:.1f}s...", file=sys.stderr)
            time.sleep(delay_s)
            delay_s = min(delay_s * 1.5, 30.0)


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


def _replace_answer_text(text: str, new_answer: str) -> str:
    return re.sub(r"<answer>.*?</answer>", f"<answer>{new_answer}</answer>", text, flags=re.S | re.I)


def _maybe_override_answer(
    response_steps: List[Tuple[str, List[Dict[str, Any]]]],
    transcript: str,
    ok: bool,
    gt_answer: str,
) -> Tuple[List[Tuple[str, List[Dict[str, Any]]]], str]:
    if not ok or not response_steps:
        return response_steps, transcript
    last_text, last_tools = response_steps[-1]
    if "<answer>" not in last_text:
        return response_steps, transcript
    response_steps[-1] = (_replace_answer_text(last_text, gt_answer), last_tools)
    transcript = _replace_answer_text(transcript, gt_answer)
    return response_steps, transcript


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
    user_text = USER_TEMPLATE.format(
        video_path=sample["video_path"],
        clip_start=float(sample["clip_start_time"]),
        clip_end=float(sample["clip_end_time"]),
        question=sample["question"],
    )
    messages: List[Dict[str, Any]] = [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
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
    gt_interval = (float(qa_start), float(qa_end))
    gt_answer = sample.get("answer")
    gt_answer_text = str(gt_answer) if gt_answer is not None else ""
    gt_answer_interval = parse_gt_interval(gt_answer)
    is_grounding = gt_answer_interval is not None

    transcript = ""
    corrections = 0
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
            answer_iou = iou_ratio(answer_interval, gt_interval)
            ok = answer_iou >= config.answer_iou_threshold
            response_steps, transcript = _maybe_override_answer(
                response_steps,
                transcript,
                ok,
                f"[{gt_interval[0]}, {gt_interval[1]}]",
            )
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
            if config.judge_model:
                ok = judge_answer(config, sample["question"], gt_answer_text, answer_text)
            else:
                ok = answer_text.strip().lower() == gt_answer_text.strip().lower()
            response_steps, transcript = _maybe_override_answer(
                response_steps,
                transcript,
                ok,
                gt_answer_text,
            )
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

    max_rounds = max(1, config.max_corrections + 1)
    rounds_used = 1
    while rounds_used < max_rounds:
        tool_args = parse_tool_calls(transcript, "crop_video")
        if tool_args and "start_time" in tool_args and "end_time" in tool_args:
            pred_interval = (float(tool_args["start_time"]), float(tool_args["end_time"]))
            overlap = overlap_ratio(pred_interval, gt_interval)
            if overlap < config.overlap_threshold and corrections < config.max_corrections:
                hint = HINT_TEMPLATE.format(gt_start=gt_interval[0], gt_end=gt_interval[1])
                _append_user(messages, hint)
                corrections += 1

        round_prompt = FINE_INSPECTION_TEMPLATE.format(
            round_idx=rounds_used + 1,
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
            answer_iou = iou_ratio(answer_interval, gt_interval)
            ok = answer_iou >= config.answer_iou_threshold
            response_steps, transcript = _maybe_override_answer(
                response_steps,
                transcript,
                ok,
                f"[{gt_interval[0]}, {gt_interval[1]}]",
            )
            return build_longvt_output(
                sample,
                response_steps,
                transcript,
                frame_paths,
                frame_bytes,
            )
        if not is_grounding and answer_text:
            if config.judge_model:
                ok = judge_answer(config, sample["question"], gt_answer_text, answer_text)
            else:
                ok = answer_text.strip().lower() == gt_answer_text.strip().lower()
            response_steps, transcript = _maybe_override_answer(
                response_steps,
                transcript,
                ok,
                gt_answer_text,
            )
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
    try:
        return generate_sample(entry, config)
    except Exception as exc:
        return {
            "clip_id": entry.get("clip_id"),
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
    parser.add_argument("--overlap-threshold", type=float, default=0.3, help="GT overlap threshold for correction.")
    parser.add_argument("--max-corrections", type=int, default=1, help="Max correction attempts per sample.")
    parser.add_argument("--answer-iou-threshold", type=float, default=0.5, help="IoU threshold for acceptance.")
    parser.add_argument("--judge-model", default=None, help="Optional judge model for acceptance.")
    parser.add_argument("--judge-api-base", default=None, help="Optional judge API base URL.")
    parser.add_argument("--judge-api-key", default=None, help="Optional judge API key.")
    parser.add_argument("--judge-max-tokens", type=int, default=16, help="Max tokens for judge response.")
    parser.add_argument("--judge-temperature", type=float, default=0.0, help="Judge temperature.")
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
        overlap_threshold=args.overlap_threshold,
        max_corrections=args.max_corrections,
        answer_iou_threshold=args.answer_iou_threshold,
        judge_model=args.judge_model,
        judge_api_base=args.judge_api_base,
        judge_api_key=args.judge_api_key,
        judge_max_tokens=args.judge_max_tokens,
        judge_temperature=args.judge_temperature,
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
            for idx, entry in enumerate(entries):
                result = worker(entry, config.__dict__)
                if idx == 0 and validate_first:
                    transcript = result.get("metadata", {}).get("transcript", "")
                    if transcript and not _validate_response(transcript):
                        result["error"] = "invalid first trajectory format"
                        out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                        out_f.flush()
                        print("[ERROR] First trajectory failed validation; aborting.", file=sys.stderr)
                        return
                out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                out_f.flush()
                progress.update(1)


if __name__ == "__main__":
    main()

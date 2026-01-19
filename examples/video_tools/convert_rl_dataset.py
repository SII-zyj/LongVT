# Copyright 2025 Individual Contributor: Kaichen Zhang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


INSTRUCTION = (
    "Think first, call **crop_video** or **get_frame** if needed, then answer. "
    "Format strictly as:  <think>...</think>  <tool_call>...</tool_call> (if tools needed)  <answer>...</answer>."
)


def _load_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() in {".jsonl", ".jsonlines"}:
        records = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, list):
        return data
    return data.get("data", [])


def _video_segment(record: dict[str, Any]) -> list[float] | None:
    if "qa_start_time" in record and "qa_end_time" in record:
        return [float(record["qa_start_time"]), float(record["qa_end_time"])]
    if "clip_start_time" in record and "clip_end_time" in record:
        return [float(record["clip_start_time"]), float(record["clip_end_time"])]
    return None


def _frame_time(record: dict[str, Any]) -> float | None:
    if "anchor_frame_time" in record:
        return float(record["anchor_frame_time"])
    return None


def _build_prompt(question: str, video_path: str) -> str:
    return f"<video>{question} {INSTRUCTION} The Video path for this video is: {video_path}"


def _iter_output(
    records: Iterable[dict[str, Any]],
    data_source: str,
    fps: int,
    max_frames: int | None,
    max_pixels: int,
    split: str,
) -> Iterable[dict[str, Any]]:
    for idx, record in enumerate(records, start=1):
        question = record.get("question", "").strip()
        answer = record.get("answer", "").strip()
        video_path = record.get("video_path") or record.get("video") or record.get("video_name", "")
        segment = _video_segment(record)
        frame_time = _frame_time(record)

        video_dict = {
            "type": "video",
            "video": f"file://{video_path}",
            "fps": fps,
            "min_frames": 1,
            "max_frames": max_frames,
            "max_pixels": max_pixels,
        }
        if max_frames is None:
            video_dict.pop("max_frames")

        extra_info = {
            "answer": answer,
            "index": idx,
            "need_tools_kwargs": True,
            "question": question,
            "split": split,
            "tools_kwargs": {
                "crop_video": {"create_kwargs": {"dummy": "dummy"}},
                "get_frame": {"create_kwargs": {"dummy": "dummy"}},
            },
        }
        if segment is not None:
            extra_info["video_segment"] = segment
        if frame_time is not None:
            extra_info["frame_time"] = frame_time

        yield {
            "data_source": data_source,
            "prompt": [{"content": _build_prompt(question, video_path), "role": "user"}],
            "videos": [video_dict],
            "ability": "video_tool",
            "reward_model": {"ground_truth": answer, "style": "model"},
            "extra_info": extra_info,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert video QA datasets into RL-ready format.")
    parser.add_argument("--input", required=True, help="Path to input JSON or JSONL file.")
    parser.add_argument("--output", required=True, help="Path to output JSONL file.")
    parser.add_argument("--data-source", required=True, help="Data source name for the RL dataset.")
    parser.add_argument("--fps", type=int, default=1, help="FPS value for the videos.")
    parser.add_argument("--max-frames", type=int, default=512, help="Max frames for the video sampler.")
    parser.add_argument("--max-pixels", type=int, default=50176, help="Max pixels for the video sampler.")
    parser.add_argument("--split", default="train", help="Split name to store in extra_info.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = _load_records(Path(args.input))

    max_frames = None if args.max_frames <= 0 else args.max_frames
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for item in _iter_output(records, args.data_source, args.fps, max_frames, args.max_pixels, args.split):
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

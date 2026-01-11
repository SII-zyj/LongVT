#!/usr/bin/env python3
"""
Clean teacher-input JSON/JSONL:
- Drop samples whose clip coverage over the full video exceeds a threshold.
- Clamp negative timestamps to 0.
"""

import argparse
import json
from pathlib import Path
from typing import Iterable

import cv2
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean teacher input data.")
    parser.add_argument("--input", required=True, help="Input JSON or JSONL file.")
    parser.add_argument("--output", required=True, help="Output JSONL file.")
    parser.add_argument("--threshold", type=float, default=0.7, help="Max clip coverage ratio.")
    return parser.parse_args()


def iter_input(path: Path) -> Iterable[dict]:
    if path.suffix == ".json":
        data = json.loads(path.read_text("utf-8"))
        if isinstance(data, list):
            yield from data
        else:
            yield data
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def get_video_duration(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / fps if fps and fps > 0 else 0.0
    cap.release()
    if duration <= 0:
        raise RuntimeError(f"Invalid duration for video: {video_path}")
    return duration


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_duration = 0.0
    total_items = 0
    kept_items = 0
    dropped_items = 0

    items = list(iter_input(input_path))
    with output_path.open("w", encoding="utf-8") as out_f:
        for item in tqdm(items, desc="Cleaning", unit="item"):
            try:
                duration = get_video_duration(item["video_path"])
            except Exception:
                dropped_items += 1
                continue
            total_duration += duration
            total_items += 1

            clip_start = max(0.0, float(item.get("clip_start_time", 0.0)))
            clip_end = max(0.0, float(item.get("clip_end_time", 0.0)))
            if clip_end < clip_start:
                clip_end = clip_start

            clip_len = max(0.0, clip_end - clip_start)
            if clip_len / duration > args.threshold:
                dropped_items += 1
                continue

            item["clip_start_time"] = clip_start
            item["clip_end_time"] = clip_end
            item["video_duration"] = duration

            answer = item.get("answer")
            is_grounding = isinstance(answer, str) and answer.strip().startswith("[")
            if is_grounding:
                item["answer"] = f"[{clip_start:.3f}, {clip_end:.3f}]"
                item.pop("qa_start_time", None)
                item.pop("qa_end_time", None)

            out_f.write(json.dumps(item, ensure_ascii=False) + "\n")
            kept_items += 1

    print(
        f"[SUMMARY] total_items={total_items} kept={kept_items} dropped={dropped_items} "
        f"total_duration_sec={total_duration:.2f}"
    )


if __name__ == "__main__":
    main()

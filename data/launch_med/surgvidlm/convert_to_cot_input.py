#!/usr/bin/env python3
"""
Convert SurgVidLM JSONL to minimal CoT input JSON.

Expected output fields:
  clip_id, video_name, video_path,
  clip_start_time, clip_end_time,
  qa_start_time, qa_end_time,
  question, answer
"""

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert JSONL to CoT input JSON.")
    parser.add_argument("--input", required=True, help="Input JSONL file.")
    parser.add_argument("--output", required=True, help="Output JSON file.")
    return parser.parse_args()


def build_clip_id(data: dict) -> str:
    clip_id = data.get("clip_id")
    if clip_id:
        return clip_id
    aug_key = data.get("_aug_key")
    if aug_key:
        return aug_key
    video_name = data.get("video_name", "unknown.mp4")
    clip_start = float(data.get("clip_start_time", data.get("qa_start_time", 0.0)))
    clip_end = float(data.get("clip_end_time", data.get("qa_end_time", 0.0)))
    clip_index = data.get("clip_index", "unknown")
    return f"{video_name}__{clip_start:.3f}_{clip_end:.3f}__idx{clip_index}"


def main() -> None:
    args = parse_args()
    rows = []
    input_path = Path(args.input)
    output_path = Path(args.output)

    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)

            tg_question = None
            tg_start = None
            tg_end = None
            tg_raw = data.get("_tg_raw")
            if tg_raw:
                try:
                    tg_data = json.loads(tg_raw)
                except json.JSONDecodeError:
                    try:
                        tg_data = json.loads(tg_raw.strip())
                    except json.JSONDecodeError:
                        try:
                            tg_data = json.loads(tg_raw.replace("\n", " ").strip())
                        except json.JSONDecodeError:
                            try:
                                tg_data = json.loads(tg_raw.replace("\r", " ").strip())
                            except json.JSONDecodeError:
                                try:
                                    tg_data = json.loads(json.dumps(tg_raw))
                                except json.JSONDecodeError:
                                    try:
                                        tg_data = json.loads(str(tg_raw))
                                    except json.JSONDecodeError:
                                        try:
                                            import ast

                                            tg_data = ast.literal_eval(tg_raw)
                                        except Exception:
                                            tg_data = None
                if isinstance(tg_data, dict):
                    tg_question = tg_data.get("tg_question")
                    tg_start = tg_data.get("tg_start_time")
                    tg_end = tg_data.get("tg_end_time")

            question = tg_question or data.get("question")
            if question is None:
                raise KeyError("Missing question field.")

            qa_start_time = tg_start if tg_start is not None else data.get("qa_start_time")
            qa_end_time = tg_end if tg_end is not None else data.get("qa_end_time")
            if qa_start_time is None or qa_end_time is None:
                raise KeyError("Missing qa_start_time/qa_end_time fields.")

            answer = data.get("answer")
            if answer is None and tg_start is not None and tg_end is not None:
                answer = f"[{float(tg_start):.3f}, {float(tg_end):.3f}]"

            rows.append(
                {
                    "clip_id": build_clip_id(data),
                    "video_name": data.get("video_name", ""),
                    "video_path": data.get("video_path", ""),
                    "clip_start_time": data.get("clip_start_time", qa_start_time),
                    "clip_end_time": data.get("clip_end_time", qa_end_time),
                    "qa_start_time": qa_start_time,
                    "qa_end_time": qa_end_time,
                    "question": question,
                    "answer": answer,
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

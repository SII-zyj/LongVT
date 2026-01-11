#!/usr/bin/env python3
"""
Convert source JSONL to minimal teacher-input JSON.

Output fields:
  clip_id, video_name, video_path,
  clip_start_time, clip_end_time,
  qa_start_time, qa_end_time,
  question, answer
"""

import argparse
import json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert JSONL to teacher-input JSON.")
    parser.add_argument("--input", required=True, help="Input JSONL file.")
    parser.add_argument("--output", required=True, help="Output JSON file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    with open(args.input, "r", encoding="utf-8") as f:
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
                    tg_question = tg_data.get("tg_question")
                    tg_start = tg_data.get("tg_start_time")
                    tg_end = tg_data.get("tg_end_time")
                except json.JSONDecodeError:
                    pass

            question = tg_question or data["question"]
            qa_start_time = tg_start if tg_start is not None else data["qa_start_time"]
            qa_end_time = tg_end if tg_end is not None else data["qa_end_time"]
            answer = data.get("answer")
            if answer is None and tg_start is not None and tg_end is not None:
                answer = f"[{float(tg_start):.3f}, {float(tg_end):.3f}]"

            rows.append(
                {
                    "clip_id": data["clip_id"],
                    "video_name": data["video_name"],
                    "video_path": data["video_path"],
                    "clip_start_time": data["clip_start_time"],
                    "clip_end_time": data["clip_end_time"],
                    "qa_start_time": qa_start_time,
                    "qa_end_time": qa_end_time,
                    "question": question,
                    "answer": answer,
                }
            )

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

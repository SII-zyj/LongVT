#!/usr/bin/env python3
"""Check evidence rewards for surgvidlm, MedVideoCap, and ophvl examples."""

from __future__ import annotations

from custom_rewards import vl_agent


def compute_reward(predict_str: str, extra_info: dict) -> tuple[float, float, float, float, float, float]:
    kwargs = {
        "use_iou_reward": True,
        "use_frame_reward": True,
        "iou_refine_h0": 0.5,
        "iou_refine_delta": 0.05,
        "iou_refine_eta": 0.1,
        "iou_refine_base": 1.0,
        "frame_reward_window": 1.0,
    }
    score, acc_reward, format_reward, evidence_reward, crop_reward, frame_reward = vl_agent.compute_score(
        predict_str=predict_str,
        ground_truth=extra_info["answer"],
        extra_info=extra_info,
        **kwargs,
    )
    return score, acc_reward, format_reward, evidence_reward, crop_reward, frame_reward


def format_predict(tool_calls: list[str]) -> str:
    tool_response = "<tool_response>ok</tool_response>"
    return (
        "<think>t</think>"
        + "".join([f"<tool_call>{call}</tool_call>{tool_response}" for call in tool_calls])
        + "<answer></answer>"
    )


def main() -> None:
    examples = [
        {
            "name": "surgvidlm",
            "extra_info": {
                "answer": "Undue tension.",
                "question": "What is being avoided during dissection to prevent damage to the peritoneum and "
                "surrounding structures?",
                "video_segment": [435.0, 600.0],
            },
            "tool_calls": [
                '{"name":"crop_video","arguments":{"start_time":440.0,"end_time":500.0}}',
                '{"name":"get_frame","arguments":{"timestamp":500.0}}',
            ],
        },
        {
            "name": "MedVideoCap",
            "extra_info": {
                "answer": "Pipettes.",
                "question": "What specific type of glassware is prominently visible on the workspace in the "
                "foreground, in addition to beakers and flasks?",
                "frame_time": 4.5,
            },
            "tool_calls": [
                '{"name":"crop_video","arguments":{"start_time":4.0,"end_time":5.0}}',
                '{"name":"get_frame","arguments":{"timestamp":4.5}}',
            ],
        },
        {
            "name": "ophvl",
            "extra_info": {
                "answer": "A haptic.",
                "question": "What specific component is being carefully inserted during the ocular surgery?",
                "video_segment": [372.639, 374.639],
            },
            "tool_calls": [
                '{"name":"crop_video","arguments":{"start_time":372.0,"end_time":375.0}}',
                '{"name":"get_frame","arguments":{"timestamp":373.0}}',
            ],
        },
    ]

    for example in examples:
        predict_str = format_predict(example["tool_calls"])
        score, acc_reward, format_reward, evidence_reward, crop_reward, frame_reward = compute_reward(
            predict_str, example["extra_info"]
        )
        print(f"[{example['name']}] score={score:.3f} acc={acc_reward:.3f} format={format_reward:.3f}")
        print(
            f"[{example['name']}] evidence={evidence_reward:.3f} crop={crop_reward:.3f} "
            f"frame={frame_reward:.3f}"
        )


if __name__ == "__main__":
    main()

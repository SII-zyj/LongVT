#!/usr/bin/env python3
"""
Minimal tests for frame reward computation.

Usage:
  python tests/test_frame_reward.py
"""

import math

from custom_rewards import vl_agent


def run_frame_reward(predict_str: str, extra_info: dict, window: float = 1.0) -> float:
    kwargs = {
        "use_frame_reward": True,
        "frame_reward_window": window,
    }
    full_extra_info = {
        "question": "dummy question",
        "answer": "dummy answer",
    }
    full_extra_info.update(extra_info)
    score, acc_reward, format_reward, frame_reward = vl_agent.compute_score(
        predict_str=predict_str,
        ground_truth="dummy",
        extra_info=full_extra_info,
        **kwargs,
    )
    return score, acc_reward, format_reward, frame_reward


def main() -> None:
    predict_template = (
        "<think>t</think>"
        "<tool_call>{\"name\":\"get_frame\",\"arguments\":{\"timestamp\":%s}}</tool_call>"
        "<tool_response>ok</tool_response>"
        "<answer>a</answer>"
    )

    score, acc_reward, format_reward, frame_reward = run_frame_reward(
        predict_template % "10.0",
        {"frame_time": 10.0},
        window=2.0,
    )
    assert math.isclose(frame_reward, 1.0, rel_tol=1e-6), frame_reward
    assert math.isclose(score, acc_reward + format_reward + frame_reward, rel_tol=1e-6)

    score, acc_reward, format_reward, frame_reward = run_frame_reward(
        predict_template % "11.0",
        {"frame_time": 10.0},
        window=2.0,
    )
    assert math.isclose(frame_reward, 0.5, rel_tol=1e-6), frame_reward
    assert math.isclose(score, acc_reward + format_reward + frame_reward, rel_tol=1e-6)

    score, acc_reward, format_reward, frame_reward = run_frame_reward(
        predict_template % "13.0",
        {"frame_time": 10.0},
        window=2.0,
    )
    assert math.isclose(frame_reward, 0.0, rel_tol=1e-6), frame_reward
    assert math.isclose(score, acc_reward + format_reward + frame_reward, rel_tol=1e-6)

    print("frame reward tests passed")


if __name__ == "__main__":
    main()

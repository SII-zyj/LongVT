# Copyright 2025 Individual Contributor: Sudong Wang, Zuhao Yang, Kaichen Zhang
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

import ast
import os
import random
import re

import requests

# from math_verify import parse, verify
from openai import OpenAI

# from rouge_score import rouge_scorer

openai_api_key = "EMPTY"
openai_api_base_list = [
    os.environ.get("LLM_AS_A_JUDGE_BASE", "YOUR_API_BASE"), # e.g. http://localhost:8000/v1
]

client_list = []
for api_base in openai_api_base_list:
    client = OpenAI(
        api_key=openai_api_key,
        base_url=api_base,
    )
    client_list.append(client)
model_name_list = []
for client in client_list:
    response = requests.get(f"{api_base}/models")
    # print(response)
    models = response.json()
    model_name_list.append(models["data"][0]["id"])


def get_chat_template():
    #     chat_template = """
    # Below are two answers to a question. Question is [Question], [Standard Answer] is the
    # standard answer to the question,
    # and [Model_answer] is the answer extracted from a model's output to this question.
    # Determine whether these two answers are consistent.
    # Note that [Model Answer] is consistent with [Standard Answer] whenever they are essentially the same.
    # If the meaning is expressed in the same way, it is considered consistent, for example, 'pink' and 'it is pink'.
    # If they are consistent, Judement is 1; if they are different, Judement is 0.
    # Just output Judement and don't output anything else.\n\n
    # """
    chat_template = """
Below are two answers to a question. Question is [Question], [Standard Answer] is the standard answer to the question, 
and [Model_answer] is the answer extracted from a model's output to this question. 

Judge how consistent the two answers are.

Scoring rules  
• 1    — Fully consistent: they convey the same meaning (e.g., “pink” vs. “it is pink”).  
• 0.5 — Partially consistent: they overlap on some key points but not all.  
• 0    — Inconsistent: they conflict or share no essential overlap.

Output **only** one of the following numbers: 1, 0.5, or 0.
"""
    return chat_template


def get_gpt4_score_ICE():
    example_1 = """
[Question]: Is the countertop tan or blue?
[Standard Answer]: The countertop is tan.
[Model_answer] : tan
Judgement: 1
"""  # noqa

    example_2 = """
[Question]: On which side of the picture is the barrier?
[Standard Answer]: The barrier is on the left side of the picture.
[Model_answer] : left
Judgement: 1
"""  # noqa

    example_3 = """
[Question]: What happens immediately after the fireworks illuminate the sky?
[Standard Answer]: The crowd cheers loudly and waves flags.
[Model_answer] : The crowd cheers.
Judgement: 0.5
"""  # noqa

    example_4 = """
[Question]: What items does the waitress hand to the customer?
[Standard Answer]: She hands over a sandwich and a cup of coffee.
[Model_answer] : She hands over a sandwich and a cup of tea.
Judgement: 0.5
"""  # noqa

    example_5 = """
[Question]: Where is the cat sitting when the dog first walks into the kitchen?
[Standard Answer]: On top of the kitchen counter.
[Model_answer] : In the kitchen, sitting on the floor near the counter.
Judgement: 0.5
"""  # noqa

    example_6 = """
[Question]: Is the man phone both blue and closed?
[Standard Answer]: Yes, the man phone is both blue and closed.
[Model_answer] : No.
Judgement: 0
"""  # noqa

    example_7 = """
[Question]: What color is the towel in the center of the picture?
[Standard Answer]: The towel in the center of the picture is blue.
[Model_answer] : The towel in the center of the picture is pink.
Judgement: 0
"""  # noqa

    return [example_1, example_2, example_3, example_4, example_5, example_6, example_7]


COMMON_VERIFY_PROMPT = """# CONTEXT #
I am a teacher, and I have some high-level reasoning problems. 
I am tasked with evaluating the correctness of a student's answer. 
Below, I am provided with a problem and a reference answer. Additionally, a student's answer is provided. 
My job is to assess whether the student's answer captures the same meaning as the reference answer, 
even when expressed with different wording or format.

# OBJECTIVE #
I need you to judge whether the student's answer is correct given the ground truth answer.

Your tasks include:
1. Identify Semantic Equivalence: Carefully examine the expression in both answers. 
Confirm whether the semantic meaning of student's final answer is equivalent to the reference answer, 
even when expressed with different wording or format.

# TONE #
Professional, scientific.

# RESPONSE: MARKDOWN REPORT #
## Equivalence Judgement
[Whether the student's answer share the same meaning with the reference answer. (TRUE or FALSE)]

# ATTENTION #
 - The reference answer is ALWAYS correct. 
 You should carefully judge whether the student gives the same answer as reference answer.
 - The Equivalence Judgement is only TRUE or FALSE. 
 The answer is FALSE even if the student's final answer almost correct with a minor mistakes.
 - Don't give extra explanation.

**Question**:
{query}

**Reference Answer**
{gold_ans}

## Student Final Answer
{pred_ans}"""


MATH_VERIFY_PROMPT = """# CONTEXT #
I am a teacher, and I have some high-level math problems. 
I am tasked with evaluating the correctness of a student's answer. 
Below, I am provided with a problem and a reference answer. Additionally, a student's answer is provided. 
My job is to assess whether the student's answer captures the same meaning as the reference answer, 
even when expressed with different wording or format.

# OBJECTIVE #
I need you to judge whether the student's answer is correct given the ground truth answer.

Your tasks include:
1. Identify Mathematical or Notational Equivalence: Pay special attention to any LaTeX expressions in both answers. 
Confirm that the mathematical relationships, variables, and operations conveyed are equivalent.

# TONE #
Professional, scientific.

# RESPONSE: MARKDOWN REPORT #
## Equivalence Judgement
[Whether the student's answer share the same meaning with the reference answer. (TRUE or FALSE)]

# ATTENTION #
 - The reference answer is ALWAYS correct. You should carefully judge whether 
 the student gives the same answer as reference answer.
 - The Equivalence Judgement is only TRUE or FALSE. The answer is 
 FALSE even if the student's final answer almost correct with a minor mistakes.
 - Don't give extra explanation.

**Question**:
{query}

**Reference Answer**
{gold_ans}

## Student Final Answer
{pred_ans}"""


def get_prompt(predict_str, ground_truth, question):
    examples = get_gpt4_score_ICE()
    chat_template = get_chat_template()
    demo_prompt = chat_template
    for example in examples:
        demo_prompt += example + "\n\n"
    test_prompt = f"""
[Question]: {question}
[Standard Answer]: {ground_truth}
[Model_answer] : {predict_str}
Judgement:"""
    full_prompt = f"{demo_prompt}{test_prompt}"

    return full_prompt


def extract_answer(text):
    # 使用非贪婪模式匹配<answer>和</answer>之间的内容
    pattern = r"<answer>(.*?)</answer>"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def compute_score(predict_str: str, ground_truth: str, extra_info=None, **kwargs) -> float:
    is_format_error = False

    # 基本标签匹配检查
    count_think_1 = predict_str.count("<think>")
    count_think_2 = predict_str.count("</think>")
    count_tool_call_1 = predict_str.count("<tool_call>")
    count_tool_call_2 = predict_str.count("</tool_call>")
    count_answer_1 = predict_str.count("<answer>")
    count_answer_2 = predict_str.count("</answer>")

    # 检查标签是否匹配
    if count_think_1 != count_think_2 or count_think_1 == 0:
        is_format_error = True
    if count_tool_call_1 != count_tool_call_2:
        is_format_error = True
    if count_answer_1 != count_answer_2 or count_answer_1 != 1:  # 必须有且仅有一个answer
        is_format_error = True

    # 严格格式检查
    if not is_format_error:
        if count_tool_call_1 == 0:
            # 不使用tool的情况：<think>...</think><answer>...</answer>
            # 允许前后有空白符
            pattern = r"^\s*<think>.*?</think>\s*<answer>.*?</answer>\s*$"
            if not re.match(pattern, predict_str, re.DOTALL):
                is_format_error = True
        else:
            # 使用tool的情况：必须严格按照 <think></think><tool_call></tool_call><think></think>... 的交替模式
            # 使用更精确的方法检查结构
            stripped_str = predict_str.strip()

            # 检查是否以<think>开头，以</answer>结尾
            if not (stripped_str.startswith("<think>") and stripped_str.endswith("</answer>")):
                is_format_error = True
            else:
                # 分析标签序列，确保tool_call和think正确交替
                # 找到所有开始标签的位置和类型
                tag_pattern = r"<(think|tool_call|answer)>"
                tags = re.findall(tag_pattern, stripped_str)

                # 期望的模式：think, (tool_call, think)*, answer
                expected_pattern = ["think"]
                for _ in range(count_tool_call_1):
                    expected_pattern.extend(["tool_call", "think"])
                expected_pattern.append("answer")

                if tags != expected_pattern:
                    is_format_error = True

    if count_answer_1 == 0 or count_answer_2 == 0:
        answer_text = ""
    else:
        answer_text = predict_str.split("<answer>")[-1].split("</answer>")[0].strip()

    # skip the case that the answer is empty
    if answer_text == "":
        acc_reward = 0.0
    else:
        question_text = extra_info["question"]
        full_prompt = get_prompt(answer_text, ground_truth, question_text)

        client_idx = random.randint(0, len(client_list) - 1)
        client = client_list[client_idx]
        model_name = model_name_list[client_idx]

        chat_response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": full_prompt},
            ],
            seed=random.randint(0, 1000000),
            temperature=0.3,
        )
        response = chat_response.choices[0].message.content.strip()
        if "Judgement:" in response:
            response = response.split("Judgement:")[-1].strip()
            if "1" in response:
                acc_reward = 1.0
            elif "0.5" in response:
                acc_reward = 0.5
            elif "0" in response:
                acc_reward = 0.0
            else:
                print(f" [WARNING] resp format error {response=}")
                acc_reward = 0.0
        else:
            if response == "1":
                acc_reward = 1.0
            elif response == "0.5":
                acc_reward = 0.5
            elif response == "0":
                acc_reward = 0.0
            else:
                print(f" [WARNING] resp format error {response=}")
                acc_reward = 0.0

    # Penalize for model trying to predict longer answer to hack llm-as-judge
    if len(answer_text) >= 1000:
        acc_reward = 0.0
        is_format_error = True

    format_reward = 0.0 if is_format_error else 1.0

    # args definition
    tool_use_reward = kwargs.get("tool_use_reward", False)  # tool reawrd
    use_time_reward = kwargs.get("use_time_reward", False)  # recall reward
    use_iou_reward = kwargs.get("use_iou_reward", False)  # iou reward
    use_new_reward = kwargs.get("use_new_reward", False)  # acc:format = 1:1
    use_frame_reward = kwargs.get("use_frame_reward", False)  # frame reward (linear decay)

    tool_reward = 0.0
    if tool_use_reward:
        count_vision_response_1 = predict_str.count("<tool_response>")
        tool_reward = 1.0 if count_vision_response_1 > 0 and acc_reward >= 0.5 else 0.0
        return (1.0 * acc_reward + 1.0 * format_reward + 1.0 * tool_reward, acc_reward, format_reward, tool_reward)
    def _extract_last_tool_call(tool_name: str):
        tool_call_pattern = r"<tool_call>(.*?)</tool_call>"
        tool_calls = re.findall(tool_call_pattern, predict_str, re.DOTALL)
        for tool_call in reversed(tool_calls):
            try:
                tool_data = ast.literal_eval(tool_call.strip())
                if isinstance(tool_data, dict) and tool_data.get("name") == tool_name:
                    return tool_data.get("arguments", {})
            except (ValueError, SyntaxError, KeyError):
                continue
        return None

    if use_time_reward or use_iou_reward or use_frame_reward:
        count_vision_response_1 = predict_str.count("<tool_response>")
        crop_reward = 0.0
        frame_reward = 0.0

        if count_vision_response_1 > 0:
            if use_time_reward or use_iou_reward:
                ground_truth_time = extra_info.get("video_segment")

                try:
                    arguments = _extract_last_tool_call("crop_video")
                    last_crop_video = None
                    if arguments and "start_time" in arguments and "end_time" in arguments:
                        last_crop_video = (float(arguments["start_time"]), float(arguments["end_time"]))

                    if last_crop_video:
                        pred_start, pred_end = last_crop_video
                        if isinstance(ground_truth_time, list) and len(ground_truth_time) == 2:
                            gt_start, gt_end = float(ground_truth_time[0]), float(ground_truth_time[1])

                            intersection_start = max(pred_start, gt_start)
                            intersection_end = min(pred_end, gt_end)
                            intersection = max(0, intersection_end - intersection_start)
                            gt_length = gt_end - gt_start

                            if use_iou_reward:
                                union_start = min(pred_start, gt_start)
                                union_end = max(pred_end, gt_end)
                                union = union_end - union_start

                                if union > 0:
                                    iou = intersection / union
                                else:
                                    iou = 1.0 if intersection > 0 else 0.0

                                iou_refine_h0 = float(kwargs.get("iou_refine_h0", 0.5))
                                iou_refine_delta = float(kwargs.get("iou_refine_delta", 0.05))
                                iou_refine_eta = float(kwargs.get("iou_refine_eta", 0.1))
                                iou_refine_base = float(kwargs.get("iou_refine_base", 1.0))

                                if iou > 0:
                                    sign = 1.0 if iou >= iou_refine_h0 else -1.0
                                    step_bonus = 0.0
                                    if iou >= iou_refine_h0 and iou_refine_delta > 0:
                                        step_bonus = iou_refine_eta * int((iou - iou_refine_h0) / iou_refine_delta)
                                    crop_reward = iou_refine_base * sign + step_bonus
                                else:
                                    crop_reward = 0.0
                            else:
                                if gt_length > 0:
                                    crop_reward = intersection / gt_length
                                else:
                                    crop_reward = 1.0 if intersection > 0 else 0.0
                        else:
                            gt_time = None
                            for key in ("frame_time", "frame_timestamp", "gt_frame_time"):
                                if key in extra_info:
                                    gt_time = float(extra_info[key])
                                    break
                            if gt_time is not None:
                                crop_reward = 1.0 if pred_start <= gt_time <= pred_end else 0.0
                except Exception:
                    crop_reward = 0.0

            if use_frame_reward:
                try:
                    arguments = _extract_last_tool_call("get_frame")
                    pred_time = None
                    if arguments and "timestamp" in arguments:
                        pred_time = float(arguments["timestamp"])

                    if pred_time is not None:
                        segment = extra_info.get("video_segment")
                        if isinstance(segment, list) and len(segment) == 2:
                            gt_start, gt_end = float(segment[0]), float(segment[1])
                            frame_reward = 1.0 if gt_start <= pred_time <= gt_end else 0.0
                        else:
                            gt_time = None
                            for key in ("frame_time", "frame_timestamp", "gt_frame_time"):
                                if key in extra_info:
                                    gt_time = float(extra_info[key])
                                    break
                            if gt_time is not None:
                                frame_window = float(kwargs.get("frame_reward_window", 1.0))
                                if frame_window > 0:
                                    frame_reward = max(0.0, 1.0 - abs(pred_time - gt_time) / frame_window)
                except Exception:
                    frame_reward = 0.0

        evidence_reward = crop_reward + frame_reward
        return (
            1.0 * acc_reward + 1.0 * format_reward + 1.0 * evidence_reward,
            acc_reward,
            format_reward,
            evidence_reward,
            crop_reward,
            frame_reward,
        )

    if use_new_reward:
        return (1.0 * acc_reward + 1.0 * format_reward, acc_reward, format_reward)
    else:
        return (0.8 * acc_reward + 0.2 * format_reward, acc_reward, format_reward)

def compute_score_time_r1(predict_str: str, ground_truth: str, extra_info=None, use_recall=False) -> float:
    is_format_error = False
    # predict_str = "<think>" + predict_str
    count_think_1 = predict_str.count("<think>")
    count_think_2 = predict_str.count("</think>")
    if count_think_1 != count_think_2 or count_think_1 == 0:  # reward hacking
        is_format_error = True

    count_tool_call_1 = predict_str.count("<tool_call>")
    count_tool_call_2 = predict_str.count("</tool_call>")
    if count_tool_call_1 != count_tool_call_2:
        is_format_error = True

    predict_no_think = predict_str.split("</think>")[-1].strip()
    count_answer_1 = predict_no_think.count("<answer>")
    count_answer_2 = predict_no_think.count("</answer>")
    if count_answer_1 != count_answer_2 or count_answer_1 == 0:
        is_format_error = True

    # more strict format check
    if not is_format_error:
        if count_tool_call_1 == 0:
            # <think>...</think><answer>...</answer>
            pattern = r"^\s*<think>.*?</think>\s*<answer>.*?</answer>\s*$"
            if not re.match(pattern, predict_str, re.DOTALL):
                is_format_error = True
        else:
            # use tool case: must strictly follow the
            # alternating pattern of <think></think><tool_call></tool_call><think></think>...

            stripped_str = predict_str.strip()

            if not (stripped_str.startswith("<think>") and stripped_str.endswith("</answer>")):
                is_format_error = True

    # extract answer content from answer tag
    if count_answer_1 == 0 or count_answer_2 == 0:
        answer_content = ""
    else:
        answer_content = predict_str.split("<answer>")[-1].split("</answer>")[0].strip()

    if answer_content == "":
        acc_reward = 0.0
    else:
        try:
            predicted_interval = ast.literal_eval(answer_content)
            if not isinstance(predicted_interval, list) or len(predicted_interval) != 2:
                acc_reward = 0.0
            else:
                pred_start, pred_end = float(predicted_interval[0]), float(predicted_interval[1])

                ground_truth_interval = ast.literal_eval(ground_truth)
                if not isinstance(ground_truth_interval, list) or len(ground_truth_interval) != 2:
                    acc_reward = 0.0
                else:
                    gt_start, gt_end = float(ground_truth_interval[0]), float(ground_truth_interval[1])

                    if use_recall:
                        # Recall = intersection / ground_truth_length
                        intersection_start = max(pred_start, gt_start)
                        intersection_end = min(pred_end, gt_end)
                        intersection = max(0, intersection_end - intersection_start)

                        gt_length = gt_end - gt_start
                        if gt_length > 0:
                            acc_reward = intersection / gt_length
                        else:
                            acc_reward = 1.0 if intersection > 0 else 0.0
                    else:
                        # compute IoU (Intersection over Union)
                        # compute intersection
                        intersection_start = max(pred_start, gt_start)
                        intersection_end = min(pred_end, gt_end)
                        intersection = max(0, intersection_end - intersection_start)

                        # compute union
                        union_start = min(pred_start, gt_start)
                        union_end = max(pred_end, gt_end)
                        union = union_end - union_start

                        # compute IoU
                        if union > 0:
                            acc_reward = intersection / union
                        else:
                            acc_reward = 1.0 if intersection > 0 else 0.0

        except (SyntaxError, ValueError, TypeError):
            acc_reward = 0.0

    format_reward = 0.0 if is_format_error else 1.0

    return 1.0 * acc_reward + 1.0 * format_reward, acc_reward, format_reward

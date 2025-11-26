
from custom_rewards.lmms_lab_recipe import compute_score

SYSTEM_PROMPT = (
    "You are a helpful assistant. When the user asks a question, your response must include two parts: "
    "first, the reasoning process enclosed in <think>...</think> tags, then the final answer enclosed in <answer>...</answer> tags."
    "Please provide a clear, concise response within <answer> </answer> tags that directly addresses the question."
)

def doc_to_text(doc):
    choices = doc['choices']
    question = doc['question']
    question = f"Question: {question.strip()}\n(A) {choices[0]} (B) {choices[1]} (C) {choices[2]} (D) {choices[3]}\n"
    return question

def doc_to_messages(doc: dict) -> list[dict]:
    question = doc_to_text(doc)
    system_messages = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]}]
    messages = [{"role": "user", "content": []}]
    messages[0]["content"].append({"type": "text", "text": question.strip()})
    messages = system_messages + messages
    return messages

def process_results(doc, results):
    acc_score = 0
    format_score = 0
    question = doc_to_text(doc)
    extra_info = {"question": question}
    answer = ['A', 'B', 'C', 'D'][doc['answer']]
    for pred in results:
        score_dict = compute_score(data_source="mmlu", solution_str=pred.strip(), ground_truth=answer, extra_info=extra_info)
        acc_score += score_dict["acc_score"]
        format_score += score_dict.get("format_reward_score", 0.0)

    return {"acc_score": acc_score / len(results) if results else 0.0, "format_score": format_score / len(results) if results else 0.0}
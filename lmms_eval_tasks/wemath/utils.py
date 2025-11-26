
from custom_rewards.lmms_lab_recipe import compute_score
import pandas as pd

SYSTEM_PROMPT = (
    "You are a helpful assistant. When the user asks a question, your response must include two parts: "
    "first, the reasoning process enclosed in <think>...</think> tags, then the final answer enclosed in <answer>...</answer> tags."
    "Please provide a clear, concise response within <answer> </answer> tags that directly addresses the question."
)


def wemath_doc_to_text_cot(doc, lmms_eval_specific_kwargs=None):
    return doc['question'] + "\n" + doc['option']


def wemath_doc_to_visual(doc):
    return [doc["image_path"].convert("RGB")]


def wemath_doc_to_messages_cot(doc, lmms_eval_specific_kwargs=None):
    question = wemath_doc_to_text_cot(doc, lmms_eval_specific_kwargs)
    visuals = wemath_doc_to_visual(doc)
    system_messages = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]}]
    messages = [{"role": "user", "content": []}]
    messages[0]["content"].append({"type": "image", "url": visuals[0]})
    messages[0]["content"].append({"type": "text", "text": question.strip()})
    messages = system_messages + messages
    return messages

def wemath_reasoning_process_results(doc, results):
    acc_score = 0
    format_score = 0
    question = wemath_doc_to_text_cot(doc, None)
    extra_info = {"question": question}
    for pred in results:
        score_dict = compute_score(data_source="wemath", solution_str=pred.strip(), ground_truth=doc["answer"], extra_info=extra_info)
        acc_score += score_dict["acc_score"]
        format_score += score_dict.get("format_reward_score", 0.0)
    
    data_dict = {
        "ID": doc['ID'],
        "split": doc['split'],
        "knowledge concept": doc['knowledge concept'],
        "question": doc['question'],
        "option": doc['option'],
        "answer": doc['answer'],
        # "image_path": doc['image_path'],
        "key": doc['key'],
        "question number": doc['question number'],
        "knowledge concept description": doc['knowledge concept description'],
        "acc_score": acc_score
    }

    return {"wemath_loose": data_dict, "wemath_strict": data_dict, "acc_score": acc_score / len(results) if results else 0.0, "format_score": format_score / len(results) if results else 0.0}

# Function to process steps data and merge results
def process_steps_data(df, steps):
    steps_data = {f'{steps}steps_{i}': df[df['key'] == f'{steps}steps_{i}'] for i in range(1, steps + 1)}
    steps_data[f'{steps}steps_multi'] = df[df['key'] == f'{steps}steps_multi']
    for key, data in steps_data.items():
        data.columns = [col + f'_{key.split("_")[-1]}' for col in data.columns]
    merged_data = steps_data[f'{steps}steps_1']
    for i in range(2, steps + 1):
        merged_data = pd.merge(merged_data, steps_data[f'{steps}steps_{i}'], left_on=f'ID_1', right_on=f'ID_{i}', how='left')
    merged_data = pd.merge(merged_data, steps_data[f'{steps}steps_multi'], left_on=f'ID_1', right_on='ID_multi', how='left')
    return merged_data


# Function to calculate evaluation metrics
def calculate_metrics(merged_2steps, merged_3steps):
    metrics = {}
    metrics['steps2_filtered_rows_1_loose'] = merged_2steps[((merged_2steps['joker_1'] == False) & (merged_2steps['joker_2'] == False)) & (merged_2steps['joker_multi'] == True)]
    metrics['steps2_filtered_rows_1_strict'] = merged_2steps[((merged_2steps['joker_1'] == False) | (merged_2steps['joker_2'] == False)) & (merged_2steps['joker_multi'] == True)]
    metrics['steps2_filtered_rows_2'] = merged_2steps[((merged_2steps['joker_1'] == True) & (merged_2steps['joker_2'] == True)) & (merged_2steps['joker_multi'] == False)]
    metrics['steps2_filtered_rows_3'] = merged_2steps[((merged_2steps['joker_1'] == False) | (merged_2steps['joker_2'] == False)) & (merged_2steps['joker_multi'] == False)]
    metrics['steps2_filtered_rows_4_loose'] = merged_2steps[((merged_2steps['joker_1'] == True) | (merged_2steps['joker_2'] == True)) & (merged_2steps['joker_multi'] == True)]
    metrics['steps2_filtered_rows_4_strict'] = merged_2steps[((merged_2steps['joker_1'] == True) & (merged_2steps['joker_2'] == True)) & (merged_2steps['joker_multi'] == True)]
    metrics['steps3_filtered_rows_1_loose'] = merged_3steps[((merged_3steps['joker_1'] == False) & (merged_3steps['joker_2'] == False) & (merged_3steps['joker_3'] == False)) & (merged_3steps['joker_multi'] == True)]
    metrics['steps3_filtered_rows_1_strict'] = merged_3steps[((merged_3steps['joker_1'] == False) | (merged_3steps['joker_2'] == False) | (merged_3steps['joker_3'] == False)) & (merged_3steps['joker_multi'] == True)]
    metrics['steps3_filtered_rows_2'] = merged_3steps[((merged_3steps['joker_1'] == True) & (merged_3steps['joker_2'] == True) & (merged_3steps['joker_3'] == True)) & (merged_3steps['joker_multi'] == False)]
    metrics['steps3_filtered_rows_3'] = merged_3steps[((merged_3steps['joker_1'] == False) | (merged_3steps['joker_2'] == False) | (merged_3steps['joker_3'] == False)) & (merged_3steps['joker_multi'] == False)]
    metrics['steps3_filtered_rows_4_loose'] = merged_3steps[((merged_3steps['joker_1'] == True) | (merged_3steps['joker_2'] == True) | (merged_3steps['joker_3'] == True)) & (merged_3steps['joker_multi'] == True)]
    metrics['steps3_filtered_rows_4_strict'] = merged_3steps[((merged_3steps['joker_1'] == True) & (merged_3steps['joker_2'] == True) & (merged_3steps['joker_3'] == True)) & (merged_3steps['joker_multi'] == True)]
    # metrics.to_csv("/Users/mac/Desktop/测试结果/error_anal/csv/gpt4o-0626.csv", index = False)
    return metrics

# Function to compute evaluation rates and final scores
def compute_final_scores(metrics, total_count):
    total_counts = {
        'InadequateGeneralization': len(metrics['steps2_filtered_rows_2']) + len(metrics['steps3_filtered_rows_2']),
        'InsufficientKnowledge': len(metrics['steps2_filtered_rows_3']) + len(metrics['steps3_filtered_rows_3']),
        'CompleteMastery_loose': len(metrics['steps2_filtered_rows_4_loose']) + len(metrics['steps3_filtered_rows_4_loose']),
        'CompleteMastery_strict': len(metrics['steps2_filtered_rows_4_strict']) + len(metrics['steps3_filtered_rows_4_strict']),
        'RoteMemorization_loose': len(metrics['steps2_filtered_rows_1_loose']) + len(metrics['steps3_filtered_rows_1_loose']),
        'RoteMemorization_strict': len(metrics['steps2_filtered_rows_1_strict']) + len(metrics['steps3_filtered_rows_1_strict'])
    }
    rates = {
        'InadequateGeneralization_rate': "{:.2%}".format(total_counts['InadequateGeneralization'] / total_count),
        'InsufficientKnowledge_rate': "{:.2%}".format(total_counts['InsufficientKnowledge'] / total_count),
        'CompleteMastery_loose_rate': "{:.2%}".format(total_counts['CompleteMastery_loose'] / total_count),
        'CompleteMastery_strict_rate': "{:.2%}".format(total_counts['CompleteMastery_strict'] / total_count),
        'RoteMemorization_loose_rate': "{:.2%}".format(total_counts['RoteMemorization_loose'] / (total_counts['CompleteMastery_loose'] + total_counts['RoteMemorization_loose'])),
        'RoteMemorization_strict_rate': "{:.2%}".format(total_counts['RoteMemorization_strict'] / (total_counts['CompleteMastery_strict'] + total_counts['RoteMemorization_strict']))
    }
    return total_counts, rates

# Function to update main results DataFrame
def update_main_results_df(total_counts, rates):

    final_score_loose = "{:.2%}".format((525 - 0.5 * total_counts['InadequateGeneralization'] - total_counts['RoteMemorization_loose'] - total_counts['InsufficientKnowledge']) / 525)
    final_score_strict = "{:.2%}".format((525 - 0.5 * total_counts['InadequateGeneralization'] - total_counts['RoteMemorization_strict'] - total_counts['InsufficientKnowledge']) / 525)

    new_row = {
        'Score (Strict)': final_score_strict,
        'InsufficientKnowledge (Strict)': f"{rates['InsufficientKnowledge_rate']} ({total_counts['InsufficientKnowledge']})",
        'InadequateGeneralization (Strict)': f"{rates['InadequateGeneralization_rate']} ({total_counts['InadequateGeneralization']})",
        'CompleteMastery (Strict)': f"{rates['CompleteMastery_strict_rate']} ({total_counts['CompleteMastery_strict']})",
        'RoteMemorization (Strict)': f"{rates['RoteMemorization_strict_rate']} ({total_counts['RoteMemorization_strict']})",

        'Score (Loose)': final_score_loose,
        'InsufficientKnowledge (Loose)': f"{rates['InsufficientKnowledge_rate']} ({total_counts['InsufficientKnowledge']})",
        'InadequateGeneralization (Loose)': f"{rates['InadequateGeneralization_rate']} ({total_counts['InadequateGeneralization']})",
        'CompleteMastery (Loose)': f"{rates['CompleteMastery_loose_rate']} ({total_counts['CompleteMastery_loose']})",
        'RoteMemorization (Loose)': f"{rates['RoteMemorization_loose_rate']} ({total_counts['RoteMemorization_loose']})"
    }
    return new_row

def wemath_aggregate_results(results, metric_name):
    data = pd.DataFrame(results)
    data['joker'] = data['acc_score'] == 1.0
    data_2steps = data[data['key'].str.contains('2steps')]
    data_3steps = data[data['key'].str.contains('3steps')]
    merged_2steps = process_steps_data(data_2steps, 2)
    merged_3steps = process_steps_data(data_3steps, 3)
    metrics = calculate_metrics(merged_2steps, merged_3steps)
    total_counts, rates = compute_final_scores(metrics, total_count=525)
    score_dict = update_main_results_df(total_counts, rates)
    if metric_name == "wemath_loose":
        return score_dict['Score (Loose)']
    elif metric_name == "wemath_strict":
        return score_dict['Score (Strict)']
    else:
        raise ValueError(f"Invalid metric name: {metric_name}")
    
def wemath_aggregate_results_loose(results):
    return wemath_aggregate_results(results, "wemath_loose")

def wemath_aggregate_results_strict(results):
    return wemath_aggregate_results(results, "wemath_strict")

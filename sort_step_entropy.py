import json


def thinking_step_score(file_name):
    """
    Reads a JSON file containing thinking steps and their scores, and returns a list of dictionaries
    with the index, step, and score for each thinking step.
    
    Returns:
        list[dict]: A list of dictionaries containing the index, step, and score for each thinking step.
    """
    import json
    with open(file_name, "r", encoding="utf-8") as f:
        data = json.load(f)  

    new_tokens = []
    new_tokens_h = []
    for key, value in data.items():
        new_tokens.append(key)
        new_tokens_h.append(value)
    dict1=[]
    for i in range(len(new_tokens)):
        dict1.append({'index': i, 'step': new_tokens[i], 'score': new_tokens_h[i]})
    print(len(dict1))
    return dict1

# dict1 = thinking_step_score("/data/jianyuan/zeju_code/think_ada/data_step_entropy.json")


def split_by_double_newline(text):
    return text.split("\n\n")

def extract_thinking_content(file_name):
    answer_thinking=[]
    with open(file_name, 'r') as f:
        for line in f:
            problem = json.loads(line)
            answer_thinking.append({
                'question': problem['question'],
                'generated_answer': problem['generated_answer']
            })
    thinking = answer_thinking[0]['generated_answer'].split("</think>")[0].strip()
    return thinking+'\n</think>\n\n'


# file_name = '/data/jianyuan/zeju_code/think_ada/test_2.jsonl'

# thinking = extract_thinking_content(file_name)


# result = split_by_double_newline(thinking)

# for i in range(len(result)):
#     result[i] = result[i]+"\n\n"

# result = result[:-1]  # Remove the last empty string if it exists




import random

def random_mask_thinking_steps(thinking_steps: list[dict], mask_percentage: float = 0.4) -> list[dict]:
    """
    Randomly masks a percentage of thinking steps.

    Args:
        thinking_steps (list[dict]): A list of dictionaries, where each dict represents a step
                                     and has 'index', 'step', and 'score' keys.
        mask_percentage (float): The percentage of steps to mask (e.g., 0.2 for 20%).

    Returns:
        list[dict]: A new list of dictionaries with some steps masked.
    """
    if not (0.0 <= mask_percentage <= 1.0):
        raise ValueError("mask_percentage must be between 0.0 and 1.0")

    num_steps = len(thinking_steps)
    num_to_mask = int(num_steps * mask_percentage)

    # Create a list of indices to mask
    indices_to_mask = random.sample(range(num_steps), num_to_mask)

    masked_steps = []
    for i in range(len(thinking_steps)):
        if i in indices_to_mask:
            thinking_steps[i]['step'] = '[SKIP]'
        
    content = ''.join([step['step'] for step in thinking_steps])
    return content

def score_mask_thinking_steps(thinking_steps: list[dict], mask_percentage: float) -> list[dict]:
    """
    Masks thinking steps based on their score relative to a given threshold.

    Args:
        thinking_steps (list[dict]): A list of dictionaries, where each dict represents a step
                                     and has 'index', 'step', and 'score' keys.
        score_threshold (float): The threshold score. Steps with scores below/above this threshold
                                 will be masked.
        mask_low_scores (bool): If True, mask steps where score <= score_threshold.
                                If False, mask steps where score > score_threshold.

    Returns:
        list[dict]: A new list of dictionaries with some steps masked.
    """


    if not (0.0 <= mask_percentage <= 1.0):
        raise ValueError("mask_percentage must be between 0.0 and 1.0")

    num_steps = len(thinking_steps)
    num_to_mask = int(num_steps * mask_percentage)

    
    score_list =[] 
    for i in range(len(thinking_steps)):
        score_list.append(thinking_steps[i]['score'])

    
    # Sort from lowest to mask_ratio, which is the low entropy
    indices_to_mask =  sorted(range(len(score_list)), key=lambda i: score_list[i], reverse=False)[:num_to_mask]

    for i in range(len(thinking_steps)):
        if i in indices_to_mask:
            thinking_steps[i]['step'] = '[SKIP]'
        
    content = ''.join([step['step'] for step in thinking_steps])
    return content



def score_mask_low_entropy_sthinking_steps(thinking_steps: list[dict], mask_percentage: float = 0.4) -> list[dict]:
    """
    Masks thinking steps based on their score relative to a given threshold.

    Args:
        thinking_steps (list[dict]): A list of dictionaries, where each dict represents a step
                                     and has 'index', 'step', and 'score' keys.
        score_threshold (float): The threshold score. Steps with scores below/above this threshold
                                 will be masked.
        mask_low_scores (bool): If True, mask steps where score <= score_threshold.
                                If False, mask steps where score > score_threshold.

    Returns:
        list[dict]: A new list of dictionaries with some steps masked.
    """


    if not (0.0 <= mask_percentage <= 1.0):
        raise ValueError("mask_percentage must be between 0.0 and 1.0")

    num_steps = len(thinking_steps)
    num_to_mask = int(num_steps * mask_percentage)

    
    score_list =[] 
    for i in range(len(thinking_steps)):
        score_list.append(thinking_steps[i]['score'])

    
    # Sort from lowest to mask_ratio, which is the low entropy
    indices_to_mask =  sorted(range(len(score_list)), key=lambda i: score_list[i], reverse=False)[:num_to_mask]

    for i in range(len(thinking_steps)):
        if i in indices_to_mask:
            thinking_steps[i]['step'] = '[SKIP]'
        
    content = ''.join([step['step'] for step in thinking_steps])
    return content
# dict1 = thinking_step_score("/data/jianyuan/zeju_code/think_ada/data_step_entropy.json")

# b = random_mask_thinking_steps(dict1, mask_percentage=0)
# c = score_mask_thinking_steps(dict1, mask_percentage=0)


# print(b)

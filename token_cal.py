import json
from transformers import AutoTokenizer

# 加载 tokenizer
tokenizer_path = "/data/jianyuan/DeepSeek-R1-Distill-Qwen-7B"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

# 读取 JSONL 文件并统计 tokens
jsonl_path = "/data/jianyuan/zeju_code/think_ada/inference_masking_thinking_cot_aime2024.jsonl"
total_tokens = 0
num_answers = 0

with open(jsonl_path, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        if "mask_thinking" in data:
            answer = data["mask_thinking"]
            tokens = tokenizer.tokenize(answer)
            token_count = len(tokens)
            total_tokens += token_count
            num_answers += 1
        # print(f"Answer {num_answers}: {token_count} tokens")

print(f"\nTotal answers: {num_answers}")
print(f"Total tokens: {total_tokens}")
print(f"Average tokens per answer: {total_tokens / num_answers if num_answers > 0 else 0}")



jsonl_path = "/data/jianyuan/zeju_code/think_ada/inference_complete_thinking_cot_aime2024.jsonl"
total_tokens = 0
num_answers = 0

with open(jsonl_path, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        full_thinking=''
        if "step_entropy_dict1" in data:
            answer = data["step_entropy_dict1"]
            for i in answer:
                full_thinking += i['step']
        tokens = tokenizer.tokenize(full_thinking)
        token_count = len(tokens)
        total_tokens += token_count
        num_answers += 1
        # print(f"Answer {num_answers}: {token_count} tokens")

print(f"\nTotal answers: {num_answers}")
print(f"Total tokens: {total_tokens}")
print(f"Average tokens per answer: {total_tokens / num_answers if num_answers > 0 else 0}")


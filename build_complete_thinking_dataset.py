import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import asyncio
import aiofiles
from tqdm import tqdm
import os
import argparse
from openai import AsyncOpenAI
from math_verify import parse
from evaluate import load
from transformers import AutoTokenizer
from sort_step_entropy import thinking_step_score, random_mask_thinking_steps, score_mask_thinking_steps




    

def create_logprob_heatmap(tokens, logprobs, save_path='logprob_heatmap.png'):
    """
    创建token logprob的折线图
    
    Args:
        tokens: token列表
        logprobs: 对应的logprob值列表
        save_path: 保存图片的路径
    """
    # 设置图形大小
    plt.figure(figsize=(50, 3))
    
    # 创建折线图
    plt.plot(range(len(logprobs)), logprobs, marker='o', linestyle='-', linewidth=1, markersize=4)
    
    # 设置标题和标签
    plt.title('Token Log Probability Trend', fontsize=14)
    plt.xlabel('Token Position', fontsize=12)
    plt.ylabel('Log Probability', fontsize=12)
    
    # 设置y轴范围
    plt.ylim(-2.0, 0.0)
    
    # 添加网格线
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 设置x轴刻度
    plt.xticks(range(len(tokens)), tokens, rotation=45, ha='right')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

import math as math_old


def calculate_sentence_joint_entropy(tokens_list, logprobs_list, sentence_delimiter='ĊĊ'):
    """
    计算给定 token 序列中每个句子的联合概率分布熵 (H_joint(S))。
    H_joint(S) = - sum(log P(o_t | q, o_<t)) for tokens in S

    Args:
        tokens_list (list): 包含所有 token 的列表。
        logprobs_list (list): 包含每个 token 对应 logprob 的列表。
                             注意：这里假设 logprobs_list 中的值是 log(P)，即对数概率。
        sentence_delimiter (str): 句子结束的标识符。默认为 'ĊĊ'。

    Returns:
        list: 一个字典列表，每个字典包含 'sentence' 和 'joint_entropy'。
              如果输入不合法（如长度不匹配），返回空列表。
    """
    if len(tokens_list) != len(logprobs_list):
        print("错误：token 列表和 logprob 列表的长度不匹配。")
        return []

    sentences_data = []
    current_sentence_tokens = []
    current_sentence_logprobs_sum = 0.0 # 累计负对数概率

    for i in range(len(tokens_list)):
        token = tokens_list[i]
        logprob = logprobs_list[i]

        current_sentence_tokens.append(token)
        # 联合概率分布熵 = - sum(log P)
        # 所以我们需要累加 log P，然后在句子结束时取负
        current_sentence_logprobs_sum += logprob

        # 检查是否达到句子结束符
        if sentence_delimiter in token:
            # 移除结束符本身，或者决定是否包含在句子文本中
            sentence_text = "".join(current_sentence_tokens) # 不包含结束符
            # joint_entropy = - (累加的 logprob)
            joint_entropy = -current_sentence_logprobs_sum 
            
            sentences_data.append({
                'sentence': sentence_text.strip(),
                'joint_entropy': joint_entropy
            })
            
            # 重置，开始下一个句子
            current_sentence_tokens = []
            current_sentence_logprobs_sum = 0.0

    # 处理可能在列表末尾没有结束符的最后一个句子
    if current_sentence_tokens:
        sentence_text = "".join(current_sentence_tokens).strip()
        joint_entropy = -current_sentence_logprobs_sum
        sentences_data.append({
            'sentence': sentence_text,
            'joint_entropy': joint_entropy
        })
        
    return sentences_data

def calculate_average_token_entropy_per_sentence(tokens: list[str], token_entropies: list[float]) -> dict[str, float]:
    """
    计算每个句子的平均 Token 熵。
    一个句子的结束由 'ĊĊ' 标识。

    Args:
        tokens: 一个字符串列表，包含按顺序的每个 token。
        token_entropies: 一个浮点数列表，对应 tokens 中每个 token 的信息熵 H_t。
                         长度必须与 tokens 列表相同。

    Returns:
        一个字典，键是句子的内容（包含 'ĊĊ'），值是该句子的平均 Token 熵。
        如果句子只有一个 'ĊĊ'（即分隔符本身构成一个“句子”），则该句子的熵为 0。
        注意：如果存在内容完全相同的句子，字典会保留最后一次计算的熵值。
              如果需要所有实例，请修改返回类型。
    """
    if len(tokens) != len(token_entropies):
        raise ValueError("tokens 列表和 token_entropies 列表的长度必须相同。")

    sentence_results = [] # 存储 (sentence_content, avg_entropy) 元组
    current_sentence_tokens = []
    current_sentence_entropies = []

    for i, token in enumerate(tokens):
        # 核心改动：精确匹配 'ĊĊ' 作为句子结束符
        if token == 'ĊĊ':
            # 如果当前句子不为空，计算其平均熵
            if current_sentence_tokens:
                # 在这里，我们将 'ĊĊ' 也视为句子的一部分，并将其加入内容
                # 但不计入平均熵的计算（因为它只是分隔符）
                sentence_content = "".join(current_sentence_tokens) + token # 将ĊĊ添加到内容
                
                if not current_sentence_entropies: # 避免除以零
                    avg_entropy = 0.0
                else:
                    avg_entropy = sum(current_sentence_entropies) / len(current_sentence_entropies)
                
                sentence_results.append((sentence_content.strip(), avg_entropy))
                
                current_sentence_tokens = []
                current_sentence_entropies = []
            else:
                # 如果当前句子为空，且直接遇到 'ĊĊ'，将其视为一个只包含分隔符的句子
                # 根据你的描述：“如果句子只有一个 'ĊĊ'，则该句子的熵为 0。”
                sentence_results.append((token.strip(), 0.0))
                
        else:
            current_sentence_tokens.append(token)
            current_sentence_entropies.append(token_entropies[i])

    # 处理最后一个句子，如果它不是以 'ĊĊ' 结尾
    if current_sentence_tokens:
        sentence_content = "".join(current_sentence_tokens).strip()
        if not current_sentence_entropies:
            avg_entropy = 0.0
        else:
            avg_entropy = sum(current_sentence_entropies) / len(current_sentence_entropies)
        sentence_results.append((sentence_content, avg_entropy))

    # 将结果转换为字典格式，处理可能重复的句子内容
    result_dict = {}
    for content, entropy in sentence_results:
        result_dict[content] = entropy 
    
    return result_dict


def compute_neg_plogp(log_p_list):
    """
    计算列表中每个 log p 对应的 -p log p
    
    参数:
    log_p_list: list of float, 包含 log p 的列表
    
    返回:
    list of float: 包含 -p log p 的列表
    """
    neg_plogp_list = []
    for log_p in log_p_list:
        p = math_old.exp(log_p)
        neg_plogp = -p * log_p
        neg_plogp_list.append(neg_plogp)
    return neg_plogp_list


def plot_neg_plogp_sentence(plogp, tokens, avg_or_joint, bar_color='skyblue', edge_color='black', 
                   title="Bar Plot of -p log p", xlabel="sentence", 
                   ylabel="-p log p", show_values=False, grid=True):
    """
    绘制 -p log p 的柱状图

    参数:
    plogp     : list of float, 包含 log p 的列表
    bar_color      : str, 柱状图颜色 (默认 'skyblue')
    edge_color     : str, 柱状图边框颜色 (默认 'black')
    title          : str, 图表标题 (默认 "Bar Plot of -p log p")
    xlabel         : str, x轴标签 (默认 "Probability (p)")
    ylabel         : str, y轴标签 (默认 "-p log p")
    show_values    : bool, 是否在柱顶显示数值 (默认 True)
    grid           : bool, 是否显示网格线 (默认 True)
    """
    # 计算 -p log p
    neg_plogp_list = plogp

    # 创建图表
    plt.figure(figsize=(50, 5))
    bars = plt.bar(range(len(neg_plogp_list)), neg_plogp_list, 
                   color=bar_color, edgecolor=edge_color)

    # 设置坐标轴标签和标题
    # plt.xticks(range(len(neg_plogp_list)), [f"p = {p:.2f}" for p in p_values])
    plt.xticks(range(len(neg_plogp_list)), tokens, rotation=45, ha='right')
    plt.xlabel(xlabel)
    
    plt.ylabel(ylabel)
    plt.title(title)

    # 显示柱顶数值
    if show_values:
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                     f"{height:.4f}", ha='center', va='bottom')

    # 显示网格线
    if grid:
        plt.grid(axis='y', linestyle='--', alpha=0.7)
    save_path = avg_or_joint + '_47.jpg'
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


os.environ["NO_PROXY"] = "localhost,127.0.0.1"
tokenizer = AutoTokenizer.from_pretrained("/data/jianyuan/DeepSeek-R1-Distill-Qwen-7B", trust_remote_code=True)
math = load("competition_math")


# async def generate_single_answer(client, question: str, model_name: str) -> str:
#     """Generate a single answer for a question using the language model."""
    
#     problem = question
    
#     cot_prompt = f"""
#     The following is a math problem:

#     [Math Problem]

#     {problem}

#     Your task is to solve it step by step. Please put your final answer in \\boxed{{}}.

#     """

#     thinking_mode = 0
#     if thinking_mode == 1:
#         try:
#             response = await client.chat.completions.create(
#                 model=model_name,
#                 messages=[
#                     {"role": "user", "content": cot_prompt}
#                 ],
#                 max_tokens=32768,
#                 temperature=0.6,
#                 top_p=0.95,
#                 n=1,
#                 logprobs=True, 
#                 top_logprobs=5 
#             )
#             #print(response.choices[0].message.content.strip())
#             output_tokens = len(tokenizer.encode(response.choices[0].message.content.strip()))
#             print('output_tokens',output_tokens)
#             if response.choices[0].logprobs:
#                 print("\nToken Logprobs:")
#                 # 收集tokens和logprobs
#                 tokens = []
#                 logprobs = []
#                 for token_logprob_data in response.choices[0].logprobs.content:
#                     token_text = token_logprob_data.token
#                     tokens.append(token_text)
#                     logprobs.append(token_logprob_data.logprob)
#                     # print(f"  Token: '{token_text}', Logprob: {token_logprob_data.logprob:.4f}")
                
#                 # 创建热力图
#                 index1 = tokens.index('</think>')
#                 think_index = tokens.index('</think>')
#                 tokens_example2 = tokens[:think_index]
#                 logit_scores_example2 = logprobs[:think_index]
#                 # create_logprob_heatmap(tokens[:index1], logprobs[:index1])
                
#                 result = compute_neg_plogp(logit_scores_example2)

#                 ccc = calculate_average_token_entropy_per_sentence(tokens_example2, result)
#                 ccc1_joint = calculate_sentence_joint_entropy(tokens_example2, logit_scores_example2)

#                 with open("data_step_entropy.json", "w", encoding="utf-8") as f:
#                     json.dump(ccc, f, indent=4, ensure_ascii=False)  # indent=4 让 JSON 格式化输出

#                 new_tokens = []
#                 new_tokens_h = []
#                 for key, value in ccc.items():
#                     new_tokens.append(key)
#                     new_tokens_h.append(value)

#                 new_tokens1=[]
#                 new_tokens_h1 = []
#                 for i in ccc1_joint:
#                     new_tokens1.append(i['sentence'])
#                     new_tokens_h1.append(i['joint_entropy'])

#                 dict1=[]
#                 for i in range(len(new_tokens)):
#                     dict1.append({'index': i, 'step': new_tokens[i], 'score': new_tokens_h[i]})

#                 print('number of steps', len(dict1))

#                 plot_neg_plogp_sentence(new_tokens_h, new_tokens, 'avg_sentence_entropy')
#                 plot_neg_plogp_sentence(new_tokens_h1, new_tokens1,'joint_entropy')
               

#             return response.choices[0].message.content.strip()
#         except Exception as e:
#             print(f"Error in generate_single_answer: {str(e)}")
#             return None
#     if thinking_mode == 0:
#         try:
#             messages=[
#                     {"role": "user", "content": cot_prompt}
#                 ],
#             prompt_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
#             #prompt_text = [prompt_text[0] + '\nOkay, I think I have finished thinking.\n</think>\n\n']
            
#             # thinking ="\nOkay, I think I have finished thinking.\n</think>\n\n"
#             dict1 = thinking_step_score("/data/jianyuan/zeju_code/think_ada/data_step_entropy.json")
#             # thinking = score_mask_thinking_steps(dict1, 0.75)
#             thinking = random_mask_thinking_steps(dict1, 0.8)
#             thinking = thinking + '\n</think>\n\n'
#             # print(thinking)
#             prompt_text = [prompt_text[0] + thinking] # y1 [pad]+y2
        
#             response = await client.completions.create(
#                 model=model_name,
#                 prompt=prompt_text,
#                 max_tokens=32768,
#                 temperature=0.6,
#                 top_p=0.95,
#                 n=1
#             )
#             return response.choices[0].text
#         except Exception as e:
#             print(f"Error in generate_single_answer: {str(e)}")
#             return None


async def generate_single_answer(client, question: str, model_name: str) -> str:
    """Generate a single answer for a question using the language model."""
    
    problem = question
    
    cot_prompt = f"""
    The following is a math problem:

    [Math Problem]

    {problem}

    Your task is to solve it step by step. Please put your final answer in \\boxed{{}}.

    """

    response = await client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "user", "content": cot_prompt}
        ],
        max_tokens=32768,
        temperature=0.6,
        top_p=0.95,
        n=1,
        logprobs=True, 
        top_logprobs=5 
    )
    #print(response.choices[0].message.content.strip())
    output_tokens = len(tokenizer.encode(response.choices[0].message.content.strip()))
    print('output_tokens',output_tokens)
    if response.choices[0].logprobs:
        print("\nToken Logprobs:")
        # 收集tokens和logprobs
        tokens = []
        logprobs = []
        for token_logprob_data in response.choices[0].logprobs.content:
            token_text = token_logprob_data.token
            tokens.append(token_text)
            logprobs.append(token_logprob_data.logprob)
            # print(f"  Token: '{token_text}', Logprob: {token_logprob_data.logprob:.4f}")
        
        # 创建热力图
        index1 = tokens.index('</think>')
        think_index = tokens.index('</think>')
        tokens_example2 = tokens[:think_index]
        logit_scores_example2 = logprobs[:think_index]
        # create_logprob_heatmap(tokens[:index1], logprobs[:index1])
        
        result = compute_neg_plogp(logit_scores_example2)

        ccc = calculate_average_token_entropy_per_sentence(tokens_example2, result)
        ccc1_joint = calculate_sentence_joint_entropy(tokens_example2, logit_scores_example2)
        new_tokens = []
        new_tokens_h = []
        for key, value in ccc.items():
            new_tokens.append(key)
            new_tokens_h.append(value)

        new_tokens1=[]
        new_tokens_h1 = []
        for i in ccc1_joint:
            new_tokens1.append(i['sentence'])
            new_tokens_h1.append(i['joint_entropy'])

        dict1=[]
        for i in range(len(new_tokens)):
            dict1.append({'index': i, 'step': new_tokens[i], 'score': new_tokens_h[i]})

        print('number of steps', len(dict1))

        # plot_neg_plogp_sentence(new_tokens_h, new_tokens, 'avg_sentence_entropy')
        # plot_neg_plogp_sentence(new_tokens_h1, new_tokens1,'joint_entropy')
    return response.choices[0].message.content.strip(), dict1




async def evaluate_single_problem(
    prob: dict,
    client: AsyncOpenAI,
    model_name: str,
    sem: asyncio.Semaphore
) -> dict:
    async with sem:
        try:
            # print("Evaluating problem: {}".format(prob["question"]))
            
            # Generate single answer
            answer, step_entropy_dict1 = await generate_single_answer(client, prob["question"], model_name)
            if answer is None:
                return None
            
            # Extract answer and check correctness
            extracted_ans = parse(answer)
            if isinstance(extracted_ans, list):
                extracted_ans = str(extracted_ans[-1])  # 取最后一个元素并转为字符串
            else:
                extracted_ans = str(extracted_ans)

            # 确保 references 也是字符串格式
            # references = str(prob["expected_answer"])
            print(extracted_ans)

            pass_at_1 = 1 if math.compute(references=[str(prob["expected_answer"])], predictions=[extracted_ans])["accuracy"] > 0.99 else 0
            
            print("------------------------------------------------------------")
            print("Question:", prob["question"])
            print("Expected answer:", prob["expected_answer"])
            print("Generated answer:", answer)
            print("Pass@1:", pass_at_1)
            
            result = {
                "question": prob["question"],
                "expected_answer": prob["expected_answer"],
                "generated_answer": answer,
                "pass@1": pass_at_1,
                "step_entropy_dict1": step_entropy_dict1
            }
            return result
        except Exception as e:
            print(f"Error in evaluate_single_problem: {str(e)}")
            return None


async def save_results_async(output_file: str, data: dict):
    async with aiofiles.open(output_file, 'a') as f:
        await f.write(json.dumps(data) + '\n')
# async def save_results_async(output_file: str, data: dict):
#     try:
#         # 确保目录存在
#         os.makedirs(os.path.dirname(output_file), exist_ok=True)
#         async with aiofiles.open(output_file, 'a') as f:
#             await f.write(json.dumps(data) + '\n')
#         print(f"Results saved to {output_file}")
#     except Exception as e:
#         print(f"Error saving results: {str(e)}")

async def main(debug: bool = False, resume: bool = False):
    # Initialize the AsyncOpenAI client
    client = AsyncOpenAI(
        base_url="http://localhost:8018/v1",
        api_key="token-abc123"
    )
    
    model_name = "DeepSeek-R1-Distill-Qwen-7B"
    
    # Load problems from test500.jsonl
    problems = []
    # with open('/data/jianyuan/zeju_code/think_ada/dataset/math500.json', 'r') as f:
    #     problem = json.load(f)
    #     for i in problem:
    #         problems.append({
    #             'question': i['problem'],
    #             'expected_answer': i['answer']
    #         })
    with open('/data/jianyuan/zeju_code/think_ada/dataset/gsm8k.jsonl', 'r') as f:
        for line in f:
            problem = json.loads(line)
            problems.append({
                'question': problem['question'],
                'expected_answer': problem['answer']
            })
        
    # If debug flag is active, only evaluate the first 50 problems
    if debug:
        # problems = [problems[12]]
        problems = problems
        print("DEBUG MODE: processing only the first 50 problems.")

    # If resume flag is active, skip already evaluated problems
    output_file = "inference_complete_thinking_cot_gsm8k.jsonl"
    # output_dir = os.path.dirname(output_file)
    # if output_dir:  # 确保路径包含目录
    #     os.makedirs(output_dir, exist_ok=True)

    if resume:
        if os.path.exists(output_file):
            # Deduplicate the results file
            dedup = {}
            with open(output_file, 'r') as res_file:
                for line in res_file:
                    if line.strip():
                        try:
                            rec = json.loads(line)
                            question = rec.get("question")
                            if question is not None:
                                dedup[question] = rec
                        except Exception as e:
                            continue

            # Write deduplicated results back to the file
            with open(output_file, 'w') as res_file:
                for rec in dedup.values():
                    res_file.write(json.dumps(rec) + "\n")

            evaluated_questions = set(dedup.keys())
            original_count = len(problems)
            problems = [p for p in problems if p["question"] not in evaluated_questions]
            skipped = original_count - len(problems)
            print(f"Resuming evaluation: Skipping {skipped} already evaluated problems.")
        else:
            print("No previous evaluation results found. Starting from scratch.")

    # Create a semaphore to limit concurrent tasks
    sem = asyncio.Semaphore(30)  # Adjust the number based on your needs
    
    # Create tasks for each problem
    tasks = [
        asyncio.create_task(evaluate_single_problem(prob, client, model_name, sem))
        for prob in problems
    ]
    
    results = []
    # Use as_completed to update progress with tqdm
    for future in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc='Processing problems'):
        result = await future
        # print('result',result)
        if result is not None:
            print("------------------------------------------------------------")
            results.append(result)
            # Save result immediately
            await save_results_async(output_file, result)

    if results:
        total_pass_at_1 = sum(result["pass@1"] for result in results)
        pass_at_1_rate = total_pass_at_1 / len(results) * 100
        print(f"\nFinal Pass@1 Rate: {pass_at_1_rate:.2f}%")

    print(f"Evaluation complete. Processed {len(results)} problems successfully.")
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", default = True, help="Run in debug mode (only evaluate the first 50 problems)")
    parser.add_argument("--resume", default = False, help="Resume evaluation by skipping already evaluated problems")
    args = parser.parse_args()
    asyncio.run(main(debug=args.debug, resume=args.resume)) 
    

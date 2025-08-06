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

import math as math_old


def calculate_sentence_joint_entropy(tokens_list, logprobs_list, sentence_delimiter='ĊĊ'):

    if len(tokens_list) != len(logprobs_list):
        print("error:token != logprob")
        return []

    sentences_data = []
    current_sentence_tokens = []
    current_sentence_logprobs_sum = 0.0 

    for i in range(len(tokens_list)):
        token = tokens_list[i]
        logprob = logprobs_list[i]

        current_sentence_tokens.append(token)
       
        current_sentence_logprobs_sum += logprob

        
        if sentence_delimiter in token:
            
            sentence_text = "".join(current_sentence_tokens) 
            
            joint_entropy = -current_sentence_logprobs_sum 
            
            sentences_data.append({
                'sentence': sentence_text.strip(),
                'joint_entropy': joint_entropy
            })
            
            
            current_sentence_tokens = []
            current_sentence_logprobs_sum = 0.0

    
    if current_sentence_tokens:
        sentence_text = "".join(current_sentence_tokens).strip()
        joint_entropy = -current_sentence_logprobs_sum
        sentences_data.append({
            'sentence': sentence_text,
            'joint_entropy': joint_entropy
        })
        
    return sentences_data

def calculate_average_token_entropy_per_sentence(tokens: list[str], token_entropies: list[float]) -> dict[str, float]:

    if len(tokens) != len(token_entropies):
        raise ValueError("tokens!= token_entropies")

    sentence_results = [] 
    current_sentence_tokens = []
    current_sentence_entropies = []

    for i, token in enumerate(tokens):
        
        if token == 'ĊĊ':
            
            if current_sentence_tokens:
                
                sentence_content = "".join(current_sentence_tokens) + token 
                
                if not current_sentence_entropies: 
                    avg_entropy = 0.0
                else:
                    avg_entropy = sum(current_sentence_entropies) / len(current_sentence_entropies)
                
                sentence_results.append((sentence_content.strip(), avg_entropy))
                
                current_sentence_tokens = []
                current_sentence_entropies = []
            else:
                
                sentence_results.append((token.strip(), 0.0))
                
        else:
            current_sentence_tokens.append(token)
            current_sentence_entropies.append(token_entropies[i])

  
    if current_sentence_tokens:
        sentence_content = "".join(current_sentence_tokens).strip()
        if not current_sentence_entropies:
            avg_entropy = 0.0
        else:
            avg_entropy = sum(current_sentence_entropies) / len(current_sentence_entropies)
        sentence_results.append((sentence_content, avg_entropy))

    
    result_dict = {}
    for content, entropy in sentence_results:
        result_dict[content] = entropy 
    
    return result_dict


def compute_neg_plogp(log_p_list):
    neg_plogp_list = []
    for log_p in log_p_list:
        p = math_old.exp(log_p)
        neg_plogp = -p * log_p
        neg_plogp_list.append(neg_plogp)
    return neg_plogp_list

tokenizer = AutoTokenizer.from_pretrained("/root/DeepSeek-R1-Distill-Qwen-7B", trust_remote_code=True)
math = load("competition_math") 

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
        
        tokens = []
        logprobs = []
        for token_logprob_data in response.choices[0].logprobs.content:
            token_text = token_logprob_data.token
            tokens.append(token_text)
            logprobs.append(token_logprob_data.logprob)
            # print(f"  Token: '{token_text}', Logprob: {token_logprob_data.logprob:.4f}")
     
    
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
                extracted_ans = str(extracted_ans[-1])  
            else:
                extracted_ans = str(extracted_ans)

            
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
#         
#         os.makedirs(os.path.dirname(output_file), exist_ok=True)
#         async with aiofiles.open(output_file, 'a') as f:
#             await f.write(json.dumps(data) + '\n')
#         print(f"Results saved to {output_file}")
#     except Exception as e:
#         print(f"Error saving results: {str(e)}")

async def main(debug: bool = False, resume: bool = False):
    # Initialize the AsyncOpenAI client
    client = AsyncOpenAI(
        base_url="http://localhost:8019/v1",
        api_key="token-abc123"
    )
    
    model_name = "DeepSeek-R1-Distill-Qwen-7B"
    
    # Load problems from test500.jsonl
    problems = []
    with open('/root/think_ada/dataset/deepscaler.json', 'r') as f:
        problem = json.load(f)
        for i in problem:
            problems.append({
                'question': i['problem'],
                'expected_answer': i['answer']
            })
    # with open('/data/think_ada/dataset/aime2025-I.jsonl', 'r') as f:
    #     for line in f:
    #         problem = json.loads(line)
    #         problems.append({
    #             'question': problem['question'],
    #             'expected_answer': problem['answer']
    #         })
        
    # If debug flag is active, only evaluate the first 50 problems
    if debug:
        # problems = [problems[12]]
        problems = problems[:50]
        print("DEBUG MODE: processing only the first 50 problems.")

    # If resume flag is active, skip already evaluated problems
    output_file = "inference_complete_thinking_cot_deepscaler.jsonl"
    # output_dir = os.path.dirname(output_file)
    # if output_dir: 
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
    

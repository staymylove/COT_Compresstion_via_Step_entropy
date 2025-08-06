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

os.environ["NO_PROXY"] = "localhost,127.0.0.1"
tokenizer = AutoTokenizer.from_pretrained("/root/DeepSeek-R1-Distill-Qwen-7B", trust_remote_code=True)
math = load("competition_math")

async def generate_single_answer(client, question: str, thinking: dict, model_name: str) -> str:
    """Generate a single answer for a question using the language model."""
    
    problem = question
    
    cot_prompt = f"""
    The following is a math problem:

    [Math Problem]

    {problem}

    Your task is to solve it step by step. Please put your final answer in \\boxed{{}}.

    """

    thinking_mode = 0
 
    if thinking_mode == 0:
        try:
            messages=[
                    {"role": "user", "content": cot_prompt}
                ],
            prompt_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            #prompt_text = [prompt_text[0] + '\nOkay, I think I have finished thinking.\n</think>\n\n']
            
            # thinking ="\nOkay, I think I have finished thinking.\n</think>\n\n"
            # dict1 = thinking_step_score("/root/think_ada/data_step_entropy.json")
            dict1 = thinking
            # thinking = score_mask_thinking_steps(dict1, 0.1)
            thinking = random_mask_thinking_steps(dict1, 0.4)
            thinking = thinking + '\n</think>\n\n'
            print(thinking)
            prompt_text = [prompt_text[0] + thinking] # y1 [pad]+y2
        
            response = await client.completions.create(
                model=model_name,
                prompt=prompt_text,
                max_tokens=32768,
                temperature=0.6,
                top_p=0.95,
                n=1
            )
            return response.choices[0].text
        except Exception as e:
            print(f"Error in generate_single_answer: {str(e)}")
            return None

    return response.choices[0].message.content.strip()




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
            answer = await generate_single_answer(client, prob["question"], prob['thinking_steps'],model_name)
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
                "pass@1": pass_at_1
            }
            return result
        except Exception as e:
            print(f"Error in evaluate_single_problem: {str(e)}")
            return None


async def save_results_async(output_file: str, data: dict):
    async with aiofiles.open(output_file, 'a') as f:
        await f.write(json.dumps(data) + '\n')


async def main(debug: bool = False, resume: bool = False):
    # Initialize the AsyncOpenAI client
    client = AsyncOpenAI(
        base_url="http://localhost:8019/v1",
        api_key="token-abc123"
    )
    
    model_name = "DeepSeek-R1-Distill-Qwen-7B"
    
    # Load problems from test500.jsonl
    problems = []
    # with open('/root/think_ada/dataset/deepscaler.json', 'r') as f:
    #     problem = json.load(f)
    #     for i in problem:
    #         problems.append({
    #             'question': i['problem'],
    #             'expected_answer': i['answer']
    #         })
    with open('/root/think_ada/inference_complete_thinking_cot_deepscaler.jsonl', 'r') as f:
        for line in f:
            problem = json.loads(line)
            problems.append({
                'question': problem['question'],
                'expected_answer': problem['expected_answer'],
                'thinking_steps': problem['step_entropy_dict1']
            })
        
    # If debug flag is active, only evaluate the first 50 problems
    if debug:
        # problems = [problems[12]]
        problems = problems[:50]
        print("DEBUG MODE: processing only the first 50 problems.")

    # If resume flag is active, skip already evaluated problems
    output_file = "inference_masking_thinking_cot_deepscaler.jsonl"
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
    

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


async def generate_single_answer(client, question: str, thinking: dict, model_name: str, tokenizer_path: str, mask_ratio: float = 0.8) -> str:
    """Generate a single answer for a question using the language model."""
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    
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
            thinking = score_mask_thinking_steps(dict1, mask_ratio)
            # thinking = random_mask_thinking_steps(dict1, mask_ratio)
            thinking = thinking + '\n</think>\n\n'
            print(f"Using mask_ratio: {mask_ratio}")
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
            return response.choices[0].text, thinking
        except Exception as e:
            print(f"Error in generate_single_answer: {str(e)}")
            return None

    return response.choices[0].message.content.strip()




async def evaluate_single_problem(
    prob: dict,
    client: AsyncOpenAI,
    model_name: str,
    tokenizer_path: str,
    mask_ratio: float,
    sem: asyncio.Semaphore
) -> dict:
    async with sem:
        try:
            # print("Evaluating problem: {}".format(prob["question"]))
            
            math = load("competition_math")
            
            # Generate single answer
            answer, mask_thinking = await generate_single_answer(client, prob["question"], prob['thinking_steps'], model_name, tokenizer_path, mask_ratio)
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
                "mask_ratio": mask_ratio,
                "mask_thinking": mask_thinking
            }
            return result
        except Exception as e:
            print(f"Error in evaluate_single_problem: {str(e)}")
            return None


async def save_results_async(output_file: str, data: dict):
    async with aiofiles.open(output_file, 'a') as f:
        await f.write(json.dumps(data) + '\n')


async def main(
    dataset_path: str,
    model_name: str,
    tokenizer_path: str,
    base_url: str,
    api_key: str,
    output_file: str,
    mask_ratio: float = 0.4,
    debug: bool = False,
    resume: bool = False,
    max_concurrent: int = 30
):
    # Initialize the AsyncOpenAI client
    client = AsyncOpenAI(
        base_url=base_url,
        api_key=api_key
    )
    
    # Load problems from dataset
    problems = []
    
    # Handle different dataset formats
    if dataset_path.endswith('.json'):
        with open(dataset_path, 'r') as f:
            problem_data = json.load(f)
            if isinstance(problem_data, list):
                for i in problem_data:
                    problems.append({
                        'question': i['problem'],
                        'expected_answer': i['answer'],
                        'thinking_steps': i.get('step_entropy_dict1', [])
                    })
            else:
                # Handle single problem in JSON format
                problems.append({
                    'question': problem_data['problem'],
                    'expected_answer': problem_data['answer'],
                    'thinking_steps': problem_data.get('step_entropy_dict1', [])
                })
    elif dataset_path.endswith('.jsonl'):
        with open(dataset_path, 'r') as f:
            for line in f:
                problem = json.loads(line)
                problems.append({
                    'question': problem['question'],
                    'expected_answer': problem['expected_answer'],
                    'thinking_steps': problem.get('step_entropy_dict1', [])
                })
    else:
        raise ValueError("Dataset file must be either .json or .jsonl format")
        
    # If debug flag is active, only evaluate the first 50 problems
    if debug:
        # problems = [problems[12]]
        problems = problems[:50]
        print("DEBUG MODE: processing only the first 50 problems.")

    # If resume flag is active, skip already evaluated problems
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
    sem = asyncio.Semaphore(max_concurrent)
    
    # Create tasks for each problem
    tasks = [
        asyncio.create_task(evaluate_single_problem(prob, client, model_name, tokenizer_path, mask_ratio, sem))
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
    parser = argparse.ArgumentParser(description="Math problem evaluation script with thinking step masking")
    
    # Model configuration
    parser.add_argument("--model_name", type=str, default="DeepSeek-R1-Distill-Qwen-7B", 
                       help="Model name to use for evaluation")
    parser.add_argument("--tokenizer_path", type=str, default="/root/DeepSeek-R1-Distill-Qwen-7B",
                       help="Path to tokenizer")
    parser.add_argument("--base_url", type=str, default="http://localhost:8019/v1",
                       help="Base URL for the API")
    parser.add_argument("--api_key", type=str, default="token-abc123",
                       help="API key for authentication")
    
    # Dataset configuration
    parser.add_argument("--dataset_path", type=str, 
                       default="/root/think_ada/inference_complete_thinking_cot_deepscaler.jsonl",
                       help="Path to the dataset file with thinking steps (.json or .jsonl)")
    parser.add_argument("--output_file", type=str, 
                       default="inference_masking_thinking_cot_deepscaler.jsonl",
                       help="Output file path for results")
    
    # Masking configuration
    parser.add_argument("--mask_ratio", type=float, default=0.8,
                       help="Ratio for random masking of thinking steps")
    
    # Execution options
    parser.add_argument("--debug", action="store_true", default=False,
                       help="Run in debug mode (only evaluate the first 50 problems)")
    parser.add_argument("--resume", action="store_true", default=False,
                       help="Resume evaluation by skipping already evaluated problems")
    parser.add_argument("--max_concurrent", type=int, default=30,
                       help="Maximum number of concurrent requests")
    
    args = parser.parse_args()
    
    asyncio.run(main(
        dataset_path=args.dataset_path,
        model_name=args.model_name,
        tokenizer_path=args.tokenizer_path,
        base_url=args.base_url,
        api_key=args.api_key,
        output_file=args.output_file,
        mask_ratio=args.mask_ratio,
        debug=args.debug,
        resume=args.resume,
        max_concurrent=args.max_concurrent
    ))

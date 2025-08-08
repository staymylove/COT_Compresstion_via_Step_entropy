import os
import argparse
import torch
import json
import glob
import numpy as np
import re
import logging
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from functools import partial
from math_verify import parse
# import deepspeed
from evaluate import load
from datasets import Dataset as HFDataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    HfArgumentParser,
    set_seed,
    TrainerCallback,
    DataCollatorForLanguageModeling
)
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel

from trl import GRPOConfig, GRPOTrainer # Although we'll use a custom reward function, this is for illustration
import os
import os
import torch

# Set GPU device explicitly
local_rank = int(os.environ.get("LOCAL_RANK", 0))
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)

# Initialize process group with explicit device_id
if torch.distributed.is_available() and torch.distributed.is_initialized():
    torch.distributed.init_process_group(
        backend="nccl",
        init_method="env://",
        device_id=local_rank  # Critical addition
    )
# Set the MASTER_PORT environment variable to an available port

math = load("competition_math")
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[
            logging.FileHandler("/mnt/cot_com_1/grpo_train.log")  
        ])
logger = logging.getLogger(__name__)
tokenizer = AutoTokenizer.from_pretrained("/mnt/cot_com_1")

# --- Configuration Arguments ---
@dataclass
class ScriptArguments:
    model_path: str = field(
        default="/mnt/cot_com/sft_model", metadata={"help": "Path to the base SFT model or merged SFT model checkpoint."}
    )
    per_device_train_batch_size: int = field(
        default=14,
        metadata={"help": "Batch size per GPU for training"}
    )
    per_device_eval_batch_size: int = field(
        default=1,
        metadata={"help": "Batch size per GPU for evaluation"}
    )
    data_path: str = field(
        default="/mnt/cot_com/grpo_sampled_10000_data.jsonl",
        metadata={"help": "Path to your JSONL data file for prompts and correctness evaluation."}
    )
    output_dir: str = field(
        default="/mnt/grpo_4_reward_model", metadata={"help": "Output directory for GRPO training checkpoints."}
    )
    new_special_token: str = field(
        default="[SKIP]", metadata={"help": "The new special token added during SFT."}
    )
    mini_batch_size: int = field(
        default=1, metadata={"help": "Mini batch size for PPO updates."}
    )
    gradient_accumulation_steps: int = field(
        default=8, metadata={"help": "Gradient accumulation steps."}
    )
    num_train_epochs: int = field(
        default=1,
        metadata={"help": "Number of training epochs"}
    )
    logging_steps: int = field(
        default=1,
        metadata={"help": "Log every X updates steps"}
    )
    learning_rate: float = field(
        default=1e-5, metadata={"help": "Learning rate for GRPO."}
    )
    save_steps: int = field(
        default=500,
        metadata={"help": "Save checkpoint every X steps"}
    )
    warmup_steps: int = field(
        default=100,
        metadata={"help": "Linear warmup over this many steps"}
    )
    fp16: bool = field(
        default=True, metadata={"help": "Whether to use FP16 (mixed precision) training."}
    )
    use_qlora: bool = field(
        default=True, metadata={"help": "Whether to use QLoRA for GRPO training."}
    )
    lora_r: int = field(
        default=16, metadata={"help": "LoRA R parameter."}
    )
    lora_alpha: int = field(
        default=32, metadata={"help": "LoRA Alpha parameter."}
    )
    lora_dropout: float = field(
        default=0.05, metadata={"help": "LoRA Dropout parameter."}
    )
    seed: int = field(
        default=42, metadata={"help": "Random seed for reproducibility."}
    )
    max_new_tokens: int = field(
        default=256, metadata={"help": "Maximum new tokens to generate during rollouts."}
    )
    kl_penalty: float = field(
        default=0.1, metadata={"help": "KL divergence penalty coefficient."}
    )
    target_kl: float = field(
        default=None, metadata={"help": "Target KL divergence for adaptive KL."}
    )
    # Reward Model Specific Parameters
    skip_token_reward_weight: float = field(
        default=-0.5, metadata={"help": "Weight for penalizing/rewarding [SKIP] token presence. Negative for penalizing."}
    )
    correct_answer_reward_weight: float = field(
        default=10.0, metadata={"help": "Weight for rewarding correct answers."}
    )
    # New: Add field for expected_answer_field
    expected_answer_field: str = field(
        default="expected_answer", metadata={"help": "Field in JSONL that contains the ground truth answer for correctness check."}
    )
    eval_steps: int = field(
        default=100,
        metadata={"help": "Run evaluation every X steps"}
    )

class SaveBestModelCallback(TrainerCallback):
    """Callback to save best model based on average reward."""
    def __init__(self):
        self.best_reward = -float('inf')
    
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        current_reward = metrics.get("eval_reward", 0)
        if current_reward > self.best_reward:
            self.best_reward = current_reward
            # Save the best model
            output_dir = os.path.join(args.output_dir, "best_model")
            os.makedirs(output_dir, exist_ok=True)
            
            # Get the model from kwargs
            trainer = kwargs.get("trainer")
            if trainer:
                trainer.save_model(output_dir)
                logger.info(f"Saved best model with reward {current_reward}")

# --- Data Processing ---
def load_and_prepare_data_for_grpo(data_path: str, tokenizer: AutoTokenizer, args: ScriptArguments):
    """
    Loads JSONL data and prepares it into prompts for GRPO.
    For GRPO, we primarily need the initial prompts from which the model generates.
    The 'response' and 'expected_answer' parts are used by the reward function.
    """
    data_list = []
    a = 0
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            question = item["question"]
            a+=1
            if a> 50:
                pass
            mask_thinking = item["mask_thinking"] 
            response = f"{mask_thinking}{item['generated_answer']}"
            output_tokens = len(tokenizer.encode(response))
            
            EOS_TOKEN = tokenizer.eos_token 
            # prompt_text = (
            #     f"<｜begin▁of▁sentence｜><｜User｜>\n{question}\n\n"
            #     f"<｜Assistant｜><think>\n{response}{EOS_TOKEN}"
            # )
            prompt_text = (
                f"<｜begin▁of▁sentence｜>You are an expert AI assistant specialized in solving complex mathematical problems. Your task is to provide concise and efficient Chain-of-Thought (CoT) reasoning. For any intermediate steps that are redundant, self-evident, or can be safely omitted without losing crucial information for the final solution, you must replace them with the special token [SKIP]. <｜User｜>\n{question}\n\n<｜Assistant｜><think>\n"
            )
            solution = response
            
            data_dict = {
                "prompt": prompt_text,
                "question": question, # Keep original question for reward calculation context
                "mask_thinking": mask_thinking, # For reward calculation, if needed
                "solution": solution, # For reward calculation, if needed
                "expected_answer": item.get(args.expected_answer_field, ""), # Ground truth for correctness
            }
            data_list.append(data_dict)
            
    dataset = HFDataset.from_list(data_list)

    # Tokenize prompts for the GRPO Trainer's generation step
    def tokenize_prompts(examples):
        # We only tokenize the prompt, as the model will generate the response.
        # Trainer will handle padding during generation.
        return tokenizer(examples["prompt"], truncation=True, max_length=8192) # Adjust max_length for prompt

    tokenized_dataset = dataset.map(
        tokenize_prompts,
        batched=True,
        # remove_columns=["prompt", "question", "mask_thinking", "solution", "expected_answer"], # Remove text columns
        desc="Tokenizing prompts"
    )
    return tokenized_dataset


# --- Reward Function ---
def calculate_reward_correctness(
    completions: List[str], # Full generated completions (prompt + response)
    prompts: List[str], # Original prompts
    question: List[str], # Original question from dataset
    expected_answer: List[str], # Ground truth answers from dataset
    **kwargs
) -> List[float]:
    """
    Calculates the reward for each generated sequence.
    Reward components:
    1. Appearance rate of [SKIP] token.
    2. Correctness of the final answer.
    """
    rewards = []
    # skip_token_id = tokenizer.convert_tokens_to_ids(args.new_special_token)

    for i, seq in enumerate(completions):
        response_len = len(seq[i]) 
        # generated_response = seq[prompt_len:].strip()

        think_tag_index = seq.find('</think>')
        solution_of_seq = seq[think_tag_index:]
        # Component 2: Answer Correctness
        extracted_ans = parse(solution_of_seq)
        print('extracted_ans_1', extracted_ans)
        print('ground t', expected_answer[i])
        
        logger.info(f"extracted_ans_1: {extracted_ans} (ground t: {expected_answer[i]})")
        if isinstance(extracted_ans, list):
            if extracted_ans != []:
                extracted_ans = str(extracted_ans[-1])
            else:
                extracted_ans = 'None'

        else:
            extracted_ans = str(extracted_ans)

        # references = str(prob["expected_answer"])
        print('extracted_ans_2', extracted_ans)
        
        pass_at_1 = 1 if math.compute(references=[str(expected_answer[i])], predictions=[extracted_ans])["accuracy"] > 0.99 else 0

        print('pass@1', pass_at_1)
        logger.info(f"extracted_ans_2: {extracted_ans} (pass_at_1: {pass_at_1})")


        if pass_at_1 == 1:
            correctness_reward = 2.0
        else:
            correctness_reward = 0.0
        
        total_reward =  correctness_reward
        
        
        # logger.info(f"--- Reward Calculation for Sequence {i+1} ---")
        # logger.info(f"Prompts: {prompts[i]}")
        # logger.info(f"Generated Response: {seq}")
        logger.info(f"correctness_reward Reward: {correctness_reward:.2f})")
        rewards.append(torch.tensor(total_reward))
    
    # Create a tensor of rewards with correct shape for GRPOTrainer
    rewards_tensor = torch.stack(rewards)
    
    # Log some information about rewards
    if len(rewards) > 0:
        logger.info(f"correctness reward function - Mean reward: {rewards_tensor.mean().item():.4f}")
    
    return rewards_tensor


def calculate_reward_skip_ratio(
    completions: List[str], # Full generated completions (prompt + response)
    prompts: List[str], # Original prompts
    question: List[str], # Original question from dataset
    expected_answer: List[str], # Ground truth answers from dataset
    **kwargs
) -> List[float]:
    """
    Calculates the reward for each generated sequence.
    Reward components:
    1. Appearance rate of [SKIP] token.
    2. Correctness of the final answer.
    """
    rewards = []
    # skip_token_id = tokenizer.convert_tokens_to_ids(args.new_special_token)

    for i, seq in enumerate(completions):
        response_len = len(seq[i]) 
        # generated_response = seq[prompt_len:].strip()

        think_tag_index = seq.find('</think>')
        thinking_content = seq[:think_tag_index]
        # Component 1: [SKIP] token presence
        skip_token_count = thinking_content.count("[SKIP]")
        
        step_num_count = thinking_content.count('\n\n') + skip_token_count
        if step_num_count == 0:
            step_num_count = 1
        skip_ratio = skip_token_count/(step_num_count)

        skip_threshold_low1 = 0.8
        skip_threshold_low2 = 0.5
        response_len = len(tokenizer.encode(seq))


        if skip_ratio >= 0.8:

            skip_reward = 1.0

        elif skip_ratio >=0.5 and skip_ratio < 0.8: 

            skip_reward = 0.5
        else:
            skip_reward= 0.0


        logger.info(f"Skip Token Count: {skip_token_count} (skip ratio Reward: {skip_reward:.2f})")
        rewards.append(torch.tensor(skip_reward))
    
    # Create a tensor of rewards with correct shape for GRPOTrainer
    rewards_tensor = torch.stack(rewards)
    
    # Log some information about rewards
    if len(rewards) > 0:
        logger.info(f"skip ratio reward function - Mean reward: {rewards_tensor.mean().item():.4f}")
    return rewards_tensor


def calculate_reward_skip_num(
    completions: List[str], # Full generated completions (prompt + response)
    prompts: List[str], # Original prompts
    question: List[str], # Original question from dataset
    expected_answer: List[str], # Ground truth answers from dataset
    **kwargs
) -> List[float]:
    """
    Calculates the reward for each generated sequence.
    Reward components:
    1. Appearance rate of [SKIP] token.
    2. Correctness of the final answer.
    """
    rewards = []
    # skip_token_id = tokenizer.convert_tokens_to_ids(args.new_special_token)

    for i, seq in enumerate(completions):
        response_len = len(seq[i]) 
        # generated_response = seq[prompt_len:].strip()

        think_tag_index = seq.find('</think>')
        thinking_content = seq[:think_tag_index]
        # Component 1: [SKIP] token presence
        skip_token_count = thinking_content.count("[SKIP]")

        response_len = len(tokenizer.encode(seq))

        if skip_token_count > 100:
            skip_reward = -2.0 
        else:
            skip_reward= 0.0
        logger.info(f"Skip Token Count: {skip_token_count} (skip num Reward: {skip_reward:.2f})")
        rewards.append(torch.tensor(skip_reward))
    
    # Create a tensor of rewards with correct shape for GRPOTrainer
    rewards_tensor = torch.stack(rewards)
    
    # Log some information about rewards
    if len(rewards) > 0:
        logger.info(f"skip num function - Mean reward: {rewards_tensor.mean().item():.4f}")
    return rewards_tensor





def calculate_reward_repsonse_length(
    completions: List[str], # Full generated completions (prompt + response)
    prompts: List[str], # Original prompts
    question: List[str], # Original question from dataset
    expected_answer: List[str], # Ground truth answers from dataset
    **kwargs
) -> List[float]:
    """
    Calculates the reward for each generated sequence.
    Reward components:
    1. Appearance rate of [SKIP] token.
    2. Correctness of the final answer.
    """
    rewards = []
    # skip_token_id = tokenizer.convert_tokens_to_ids(args.new_special_token)

    for i, seq in enumerate(completions):
        response_len = len(seq[i]) 
        # generated_response = seq[prompt_len:].strip()

        think_tag_index = seq.find('</think>')
        thinking_content = seq[:think_tag_index]
        # Component 1: [SKIP] token presence
        skip_token_count = thinking_content.count("[SKIP]")

        response_len = len(tokenizer.encode(seq))

        if response_len > 3500:
            skip_reward = -1.0 
        else:
            skip_reward= 0.0
        logger.info(f"Skip Token Count: {skip_token_count} (length Reward: {skip_reward:.2f})")
        rewards.append(torch.tensor(skip_reward))
    
    # Create a tensor of rewards with correct shape for GRPOTrainer
    rewards_tensor = torch.stack(rewards)
    
    # Log some information about rewards
    if len(rewards) > 0:
        logger.info(f"repsonse_length - Mean reward: {rewards_tensor.mean().item():.4f}")
    return rewards_tensor



# --- Main Training Script ---
def main():
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]

    set_seed(args.seed)

    # --- 1. Load Tokenizer ---
    logger.info(f"Loading tokenizer: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token 
        logger.info(f"Set tokenizer.pad_token to tokenizer.eos_token: {tokenizer.pad_token}")

    # Add new special tokens to tokenizer if not present
    special_tokens_to_add = []
    if args.new_special_token not in tokenizer.vocab:
        special_tokens_to_add.append(args.new_special_token)
    if "<think>" not in tokenizer.vocab:
        special_tokens_to_add.append("<think>")
    if "</think>" not in tokenizer.vocab:
        special_tokens_to_add.append("</think>")

    if special_tokens_to_add:
        num_added_toks = tokenizer.add_tokens(special_tokens_to_add, special_tokens=True)
        logger.info(f"Added {num_added_toks} new tokens to tokenizer: {special_tokens_to_add}")
        for token in special_tokens_to_add:
            logger.info(f"ID for {token}: {tokenizer.convert_tokens_to_ids(token)}")
    else:
        logger.info("All required special tokens already in tokenizer vocab.")
    logger.info(f"Final tokenizer vocab size: {len(tokenizer)}")

    # --- 2. Load Policy Model (SFT Model) ---
    logger.info(f"Loading policy model from: {args.model_path}")
    
    # Determine the device. Accelerate will handle distribution if `accelerate launch` is used.
    # We remove device_map here to let accelerate manage it.
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        # load_in_4bit=args.use_qlora, # Load in 4-bit if QLoRA is enabled
        # torch_dtype=torch.bfloat16,
        # trust_remote_code=True,
        #device_map="auto", 
        use_cache = False,
        # device_map removed, accelerate will handle it
    )

    # Resize model embeddings if tokenizer size changed
    target_embedding_size = len(tokenizer)
    current_embedding_size = model.get_input_embeddings().weight.shape[0]

    if current_embedding_size != target_embedding_size:
        logger.info(f"Resizing model embeddings from {current_embedding_size} to {target_embedding_size} tokens.")
        model.resize_token_embeddings(target_embedding_size)
    else:
        logger.info(f"Model embedding size ({current_embedding_size}) already matches tokenizer vocab ({target_embedding_size}). No resizing needed.")

    if args.use_qlora:
        model = prepare_model_for_kbit_training(model)
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    else:
        # If not QLoRA, ensure the model is trainable
        for param in model.parameters():
            param.requires_grad = True # All parameters trainable for full fine-tuning

    # --- 4. Load Data for GRPO ---
    logger.info(f"Loading and preparing data from: {args.data_path}")
    grpo_dataset = load_and_prepare_data_for_grpo(args.data_path, tokenizer, args)
    logger.info(f"Loaded {len(grpo_dataset)} samples for GRPO training.")
    train_dataset = grpo_dataset
    eval_dataset = train_dataset
    # --- 5. Initialize GRPO Trainer ---
    training_args = GRPOConfig(
        output_dir=args.output_dir,
        seed=args.seed,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        logging_steps=args.logging_steps,
        eval_strategy="no",  # Use "steps" instead of "no" for proper evaluation strategy
        eval_steps=args.eval_steps,  # Add this to specify when to evaluate
        save_strategy="steps",
        save_steps=args.save_steps,
        warmup_steps=args.warmup_steps,
        save_total_limit=3,
        weight_decay=0.01,
        gradient_checkpointing=True,
        disable_tqdm = False,
        # Let DeepSpeed handle mixed precision (set via config file)
        # bf16=True,  
        # report_to="wandb",
        max_grad_norm=1.0,
        remove_unused_columns=False,
        use_vllm=True,
        #vllm_server_base_url='http://localhost:8000',
        #vllm_mode="server",
        # Generation config
        temperature=0.6,
        num_generations=8,
        # data processings
        max_prompt_length=512,
        max_completion_length=4096,
        log_completions=True,
        # Remove do_eval parameter as it's redundant with eval_strategy
        # Efficiency improvements from TRL v0.16.0
        # scale_rewards=True,  # Enable reward scaling for multiple rewards
        # num_iterations=4,     # Enable multi-step optimization (6x faster)
    )

    base_reward_fn = calculate_reward_correctness

    skip_ratio_fn = calculate_reward_skip_ratio

    skip_num_fn = calculate_reward_skip_num

    repsonse_length_fn = calculate_reward_repsonse_length

    reward_functions = [base_reward_fn, skip_ratio_fn, skip_num_fn, repsonse_length_fn]

    grpo_trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,  # Add evaluation dataset
        reward_funcs=reward_functions,  # Use multiple reward functions with weights
        callbacks=[SaveBestModelCallback()],  # Add callback to save best model based on rewards
    )

    # --- 6. Start GRPO Training ---
    logger.info("Starting GRPO training...")
    grpo_trainer.train()

    # --- 7. Save Final Model ---
    output_dir = os.path.join(args.output_dir, "final_checkpoint")
    os.makedirs(output_dir, exist_ok=True)
    
    grpo_trainer.save_model(os.path.join(args.output_dir, "final_model"))
    logger.info(f"Training completed. Final model saved to {os.path.join(args.output_dir, 'final_model')}")

    logger.info("GRPO training complete!")

if __name__ == "__main__":

    main()
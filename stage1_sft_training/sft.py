import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_dataset, Dataset
from peft import get_peft_model, LoraConfig
from tqdm import tqdm
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model and tokenizer
model_id = "/root/DeepSeek-R1-Distill-Qwen-7B"
# model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token 

# Load dataset
dataset = load_dataset("Jianyuan1/cot-data")
train_dataset = dataset["train"]

def preprocess_function(examples):
    messages = examples["conversations"]
    # Combine question and answer into a single text
    templated_messages = tokenizer.apply_chat_template(messages, tokenize=False)
    
    # Tokenize the texts
    tokenized = tokenizer(
        templated_messages, padding="max_length", truncation=True, max_length=2048, return_tensors="pt"
    )
    
    # Create labels (same as input_ids for causal language modeling)
    tokenized["labels"] = tokenized["input_ids"].clone()
    
    return tokenized

# Process the dataset
train_dataset = train_dataset.map(
    preprocess_function,
    batched=True,
    num_proc=4,
    remove_columns=["conversations"],
    desc="Processing dataset"
)


# Configure LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
)

# Training arguments
training_args = TrainingArguments(
    output_dir="deepseek-r1-cot-math-reasoning-adapters",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    evaluation_strategy="no", 
    save_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_total_limit=2,
    fp16=True,
    dataloader_num_workers=4,
    deepspeed="./ds_config.json",
)

def main():
    # Load and prepare the model
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model = get_peft_model(model, lora_config)

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )

    # Train the model
    trainer.train()

    # Merge and save the model
    model = model.merge_and_unload()
    model.save_pretrained("./deepseek-r1-14b-cot-math-reasoning-full")
    tokenizer.save_pretrained("./deepseek-r1-14b-cot-math-reasoning-full")
    print("Full model and tokenizer saved!")

if __name__ == "__main__":
    main()
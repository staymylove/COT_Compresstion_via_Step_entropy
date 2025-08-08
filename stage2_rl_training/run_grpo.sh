#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6
# MODEL_PATH="model-7b"
OUTPUT_DIR="results/grpo_10k"
GRAD_ACCUM=2
LEARNING_RATE=5e-5
EPOCHS=3
NUM_PROCESSES=7  # Number of GPUs to use
DS_CONFIG="ds_config.json"  # Path to DeepSpeed config file
ACC_CONFIG="/mnt/cot_com_1/accelerate_config.yaml"
# Parse command line arguments
POSITIONAL_ARGS=()
MAX_TRAIN=""
MAX_EVAL=""
#USE_LORA="--use_lora"
# LOAD_4BIT="--load_in_4bit"

# Restore positional arguments
set -- "${POSITIONAL_ARGS[@]}"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# AVAILABLE_PORT=$(get_available_port)

# MASTER_ADDR="127.0.0.1"
# MASTER_PORT=$AVAILABLE_PORT MASTER_ADDR=$MASTER_ADDR accelerate launch \
# MASTER_ADDR="localhost"
# MASTER_PORT="29549" 
# master_addr=$MASTER_ADDR
# echo $master_addr


# export NCCL_DEBUG=INFO
export NCCL_SOCKET_FAMILY=AF_INET
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO          # Enable verbose NCCL logging
export NCCL_DEBUG_SUBSYSTEM=ALL # Get even more detailed NCCL debug info
export NCCL_ASYNC_ERROR_HANDLING=1 # Can sometimes help prevent hangs on errors

    # --machine_rank=0 \
    # --main_process_ip=fdbd:dccd:cdc2:12c8:0:2b5:: \
accelerate launch \
    --config_file=$ACC_CONFIG \
    --main_process_port=34375 \
    ./cot_com_1/grpo_train.py \
    #--model_name_or_path "$MODEL_PATH" \
    #--train_data_dir "$TRAIN_DIR" \
    #--val_data_dir "$VAL_DIR" \
    #--template_path "$TEMPLATE_PATH" \
    #--output_dir "$OUTPUT_DIR" \
    #--per_device_train_batch_size "$BATCH_SIZE" \
    #--gradient_accumulation_steps "$GRAD_ACCUM" \
    #--learning_rate "$LEARNING_RATE" \
    #--num_train_epochs "$EPOCHS" \
    # --deepspeed "$DS_CONFIG" \
    # $USE_LORA \
    # $LOAD_4BIT \
    $MAX_TRAIN \
    $MAX_EVAL 

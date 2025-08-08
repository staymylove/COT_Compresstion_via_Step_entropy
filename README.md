# COT Compression via Step Entropy

Our paper: Compressing Chain-of-Thought in LLMs via Step Entropy


# Environment
To set up the environment, run the following commands:

```bash
conda create -n cot_com python==3.10
conda activate cot_com
bash requirement.sh
```


# COT Compression 
First, we need set up the vllm server, here we can choose DeepSeek-R1 Series models, Qwen3 (reasoning mode), QwQ-32B and S1-32B. 
```bash
cd build_data_cot_compresstion
bash run_vllm_LM.sh
```
And then, run Full Thinking Cot inference.
```bash
python build_complete_thinking_dataset.py \
    --dataset_path /path/to/dataset.jsonl \
    --model_name 'DeepSeek-R1-Distill-Qwen-7B' \
    --tokenizer_path "/root/DeepSeek-R1-Distill-Qwen-7B" \
    --base_url "http://localhost:8019/v1" \
    --api_key "token-abc123" \
    --output_file "/root/think_ada/inference_complete_thinking_cot_deepscaler.jsonl" \
```

Build Compressed Thinking COT inference.

```bash
python build_masking_thinking_dataset.py \
    --dataset_path "/root/think_ada/inference_complete_thinking_cot_deepscaler.jsonl" \
    --model_name 'DeepSeek-R1-Distill-Qwen-7B' \
    --tokenizer_path "/root/DeepSeek-R1-Distill-Qwen-7B" \
    --base_url "http://localhost:8019/v1" \
    --api_key "token-abc123" \
    --output_file "inference_masking_thinking_cot_deepscaler.jsonl" \
    --mask_ratio 0.8 \
```

# Two-Stage Training

```bash
cd stage1_sft_training
deepspeed --num_gpus=8 sft.py 
```
```bash
cd stage2_rl_training
bash run_grpo_train.sh
```

# Citation

```
@misc{li2025compressingchainofthoughtllmsstep,
      title={Compressing Chain-of-Thought in LLMs via Step Entropy}, 
      author={Zeju Li and Jianyuan Zhong and Ziyang Zheng and Xiangyu Wen and Zhijian Xu and Yingying Cheng and Fan Zhang and Qiang Xu},
      year={2025},
      eprint={2508.03346},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2508.03346}, 
}
```

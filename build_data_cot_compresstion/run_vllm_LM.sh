export CUDA_VISIBLE_DEVICES=0
vllm serve \
    "/root/DeepSeek-R1-Distill-Qwen-7B" \
    --served-model-name "DeepSeek-R1-Distill-Qwen-7B" \
    --port 8019 \
    --tensor-parallel-size 1 \
    --dtype auto \
    --api-key "token-abc123" \
    --gpu_memory_utilization 0.8 \
    --enable-prefix-caching

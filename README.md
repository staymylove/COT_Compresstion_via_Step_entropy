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
And then, run Full Cot inference.
```bash
python build_complete_thinking_dataset.py
```
Build

```bash
python build_masking_thinking_dataset.py
```

# Two-Stage Training

```bash

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

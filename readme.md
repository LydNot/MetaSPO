# [NeurIPS 2025] System Prompt Optimization with Meta-Learning
[![Paper](https://img.shields.io/badge/arXiv-2505.09666-b31b1b)](https://arxiv.org/abs/2505.09666)
[![Python](https://img.shields.io/badge/Python-3.10%2B-orange)](https://www.python.org/downloads/release/python-310s0/)
[![GCC](https://img.shields.io/badge/gcc-9.1%2B-blue)](https://gcc.gnu.org/gcc-9/)

🚀 **Welcome to the official repository of** [**System Prompt Optimization with Meta-Learning**](https://arxiv.org/abs/2505.09666)!

## 🔍 Overview
![MetaSPO](asset/main_fig.jpg)
This repository contains the official implementation of Meta-level System Prompt Optimizer (MetaSPO), a meta-learning approach for optimizing system prompts for Large Language Models (LLMs). MetaSPO is designed to optimize system prompts that are robust to diverse user inputs and transferable across a wide range of tasks and domains.

## ✨ Extensions
<!-- Documentation below authored by Claude Opus 4.5 -->

**Note:** Read about the lessons learned from this implementation in [The Bitter Lesson Eats 'System Prompt Optimization with Meta-Learning'](BLOG.md) - particularly how Claude Opus 4.5's raw capability made prompt optimization largely irrelevant on saturated tasks.

This fork extends MetaSPO with:

### 1. Claude Integration
- Full support for Anthropic's Claude models (Opus 4.5, Sonnet)
- Native integration with MetaSPO's optimization pipeline
- Set `ANTHROPIC_API_KEY` in `.env`

### 2. LLM-as-Judge Evaluation
- Uses GPT-4o-mini to evaluate mathematical correctness
- Enables MetaSPO on proof-based tasks where exact matching fails
- Minimal cost overhead (~$0.01-0.02 per training run)

### 3. Math Task Support
- StackMathQA (graduate-level mathematics)
- GSM8K support
- Proper answer extraction from mathematical proofs

### Quick Start: Claude + Math

```bash
# Run MetaSPO with Claude Opus 4.5 on math problems
./run_stackmathqa_with_judge.sh
```

This uses:
- **Base model**: Claude Opus 4.5 (solves math problems)
- **Optimizer**: GPT-4o-mini (optimizes prompts)
- **Judge**: GPT-4o-mini (evaluates correctness)

Estimated cost: ~$12 for 2 tasks, 10 train samples, 50 test samples, 2 iterations.

## 📌 Get Started
### Installation
```bash
git clone https://github.com/Dozi01/MetaSPO.git
cd MetaSPO
conda create -n metaspo python=3.10 -y
conda activate metaspo
pip install -r requirements.txt

# Additional dependencies for extensions
pip install anthropic
```
Ensure your API keys are stored in the .env file:
```bash
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

### MetaSPO: Training and Evaluation
```bash
./main.sh
```
Refer to `main.sh` for detailed instructions.

### Tasks
Modify `configs/$DOMAIN.yaml` to set dataset configurations.
To implement new tasks, include the task name in `srt/tasks/__init__.py` and implement a corresponding task class.

### Using Extensions

#### Training with Custom Models

```bash
python meta_train.py \
  --method metaspo \
  --task_config_path ./configs/math_judged.yaml \
  --base_model_type claude \  # or 'openai', 'vllm'
  --base_model_name claude-opus-4.5 \
  --optim_model_type openai \
  --optim_model_name gpt-4o-mini \
  --train_size 10 \
  --test_size 50 \
  --iteration 2
```

#### LLM-as-Judge Tasks

For tasks requiring semantic evaluation (not exact matching), use the `_judged` suffix:

```yaml
# configs/math_judged.yaml
meta_train_tasks:
  - algebra_small_judged
  - geometry_small_judged
```

The judge evaluates whether answers are mathematically correct, even if phrased differently than the reference answer.

#### Supported Models

- **Claude**: `claude-opus-4.5`, `claude-sonnet-4.5`
- **OpenAI**: `gpt-4o`, `gpt-4o-mini`, `gpt-3.5-turbo`
- **vLLM**: Custom deployed models

#### Cost Estimation

```bash
python estimate_claude_with_judge.py
```

Shows estimated costs for Claude + LLM-as-judge training runs.

<!-- End Claude Opus 4.5 documentation -->

## 📜 Citation
If you find this work useful, please cite our paper:
```
@article{choi2025promptoptimizationmetalearning,
      title={System Prompt Optimization with Meta-Learning}, 
      author={Yumin Choi and Jinheon Baek and Sung Ju Hwang},
      year={2025},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.09666}, 
}
```

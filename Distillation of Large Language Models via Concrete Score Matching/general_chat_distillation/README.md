# General Chat Distillation

## Environment

Requires [Conda](https://docs.conda.io/) and an NVIDIA GPU/driver compatible with CUDA `12.1`. Uses PyTorch `2.4.0`.

```bash
# 1. Create and activate a fresh conda environment
conda create -n env_gen_chat python=3.10 -y
conda activate env_gen_chat

# 2. Install PyTorch (2.4.0, CUDA 12.1)
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# 3. Install the remaining dependencies
pip install -r requirements.txt

# 4. Install FlashAttention-2 (build isolation must be disabled)
pip install flash-attn --no-build-isolation
```

## Generation

Generate teacher and student responses, then combine them into teacher–student pairs. First set the model and output paths. The example below uses **Qwen2.5-7B-Instruct** as the teacher and **Qwen2.5-1.5B-Instruct** as the student—change these to match your setup.

We ran generation on **4 GPUs**, selected via `CUDA_VISIBLE_DEVICES`; adjust this to the devices available on your machine.

```bash
export TEACHER_MODEL=Qwen/Qwen2.5-7B-Instruct
export STUDENT_MODEL=Qwen/Qwen2.5-1.5B-Instruct
export TEACHER_DIR=qwen2.5_output_tch
export STUDENT_DIR=qwen2.5-1.5B-inst_output_stu
export PAIR_DIR=qwen2.5-1.5B-inst_output_com
export SEED=42
export CUDA_VISIBLE_DEVICES=0,1,2,3   # we used 4 GPUs; adjust to your setup
```

Each `generate_vllm.py` run produces one response per prompt and saves it to `<output_dir>/output_<seed>.json`. By default it uses the `HuggingFaceH4/ultrachat_200k` prompt dataset and a sampling temperature of `0.8`.

### 1. Generate teacher responses

```bash
python generate/generate_vllm.py \
    --model $TEACHER_MODEL \
    --output_dir $TEACHER_DIR \
    --seed $SEED
```

### 2. Generate student responses

```bash
python generate/generate_vllm.py \
    --model $STUDENT_MODEL \
    --output_dir $STUDENT_DIR \
    --seed $SEED
```

### 3. Combine into teacher–student pairs

```bash
python generate/reformat.py \
    --teacher_file $TEACHER_DIR/output_$SEED.json \
    --student_file $STUDENT_DIR/output_$SEED.json \
    --output_dir $PAIR_DIR
```

## Training

### 1. Resize the embedding layer

Some LLMs (e.g., Qwen2.5) use different classifier head sizes across model scales. Align the teacher and student before distillation:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python utils/resize_embedding.py \
    --teacher-model Qwen/Qwen2.5-7B-Instruct \
    --student-model Qwen/Qwen2.5-1.5B-Instruct
```

### 2. Run CSD distillation

The training configuration files follow the setups described in the paper and target a 4×A100 GPU setup. You may need to adjust `num_processes` and `per_device_train_batch_size` to match your environment. You can change the student and teacher models by editing `model_name_or_path` and `ref_model_name_or_path` in the config file.

We trained on **4 GPUs** (`--num_processes=4`); set `CUDA_VISIBLE_DEVICES` to a matching number of devices.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --config_file accelerate_configs/deepspeed_zero3.yaml \
    --num_processes=4 \
    src/run_csd.py \
    training_configs/qwen2.5-inst-1.5b-csd_ss_ss.yaml
```

## Evaluation

Following the paper (Section 4.3, Table 4), we evaluate general chat capability on **MT-Bench** and **AlpacaEval** (win rate against `text-davinci-003`).

- **AlpacaEval** — see [tatsu-lab/alpaca_eval](https://github.com/tatsu-lab/alpaca_eval).
- **MT-Bench** — see [LMSYS FastChat](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge) (`fastchat/llm_judge`).

## Acknowledgements

This codebase is built upon the implementations below. We thank the authors for releasing their code.

- HuggingFace alignment-handbook. [Code](https://github.com/huggingface/alignment-handbook)
- *Ko, Jongwoo, et al. "DistiLLM-2: A Contrastive Approach Boosts the Distillation of LLMs." International Conference on Machine Learning, 2025.* [Paper](https://arxiv.org/abs/2503.07067), [Code](https://github.com/jongwooko/distillm-2)

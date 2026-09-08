# Task-specific Distillation

## Environment

We use separate virtual environments for training and evaluation.

### Training Environment

The training environment is used for supervised fine-tuning and student distillation. It uses Python 3.10, PyTorch 2.4.1, CUDA 11.8, and `flash-attn` 2.6.3.

```bash
python3.10 -m venv env_train
source env_train/bin/activate
python -m pip install -U pip setuptools wheel packaging ninja psutil
python -m pip install flash-attn==2.6.3 --no-build-isolation
python -m pip install -r train_requirements.txt
```

Patch the alignment-handbook import path for compatibility with recent versions of huggingface-hub:

```bash
python - <<'PY'
from pathlib import Path

# Make sure this path matches your virtual environment location.
path = Path("./env_train/lib/python3.10/site-packages/alignment/model_utils.py")

text = path.read_text()
old = "from huggingface_hub.utils._errors import RepositoryNotFoundError"
new = "from huggingface_hub.errors import RepositoryNotFoundError"

if old not in text:
    print("Old import line not found.")
    print("Current matching lines:")
    for line in text.splitlines():
        if "RepositoryNotFoundError" in line:
            print(line)
else:
    text = text.replace(old, new)
    path.write_text(text)
    print("Patch applied.")
PY
```

### Evaluation Environment

The evaluation scripts use a separate Python 3.10 virtual environment with PyTorch 2.4.0 and CUDA 12.1.

```bash
python3.10 -m venv env_eval
source env_eval/bin/activate
python -m pip install -r eval_requirements.txt
```

## Datasets

Download the training and evaluation data from [speculative_kd](https://github.com/google-research/google-research/tree/master/speculative_kd) and place them under `./data`.

## Models

Download the student model checkpoint and place it under `./ckpt/student`:

```bash
huggingface-cli download google/gemma-2b-it \
  --repo-type model \
  --local-dir ./ckpt/student
```

## Training

### Teacher Supervised Fine-Tuning

The teacher model can be fine-tuned on each task using the provided YAML configuration files.

| Task | Configuration |
| --- | --- |
| Summarization | `config/sft/sft_config_summ_teacher.yaml` |
| Translation | `config/sft/sft_config_trans_teacher.yaml` |
| GSM8K-Arithmetic reasoning | `config/sft/sft_config_gsm_teacher.yaml` |

Run the following commands to fine-tune the teacher model.

```bash
# Summarization
ACCELERATE_LOG_LEVEL=info accelerate launch \
  --config_file config/deepspeed_zero3.yaml \
  train/train_sft.py ./config/sft/sft_config_summ_teacher.yaml

# Translation
ACCELERATE_LOG_LEVEL=info accelerate launch \
  --config_file config/deepspeed_zero3.yaml \
  train/train_sft.py ./config/sft/sft_config_trans_teacher.yaml

# GSM8K-Arithmetic reasoning
ACCELERATE_LOG_LEVEL=info accelerate launch \
  --config_file config/deepspeed_zero3.yaml \
  train/train_sft.py ./config/sft/sft_config_gsm_teacher.yaml
```

### Knowledge Distillation

After preparing the teacher checkpoints, train the student model with knowledge distillation using the provided configuration files. The following commands reproduce the student distillation runs for each task.

| Task | Configuration |
| --- | --- |
| Summarization | `./config/csd_distill/kd_train_summ.yaml` |
| Translation | `./config/csd_distill/kd_train_mt.yaml` |
| GSM8K-Arithmetic reasoning | `./config/csd_distill/kd_train_gsm.yaml` |

```bash
# Summarization
python train/run_kd_train.py ./config/csd_distill/kd_train_summ.yaml

# Translation
python train/run_kd_train.py ./config/csd_distill/kd_train_mt.yaml

# GSM8K-Arithmetic reasoning
python train/run_kd_train.py ./config/csd_distill/kd_train_gsm.yaml
```

## Evaluation

After training or distillation, evaluate a checkpoint using the task-specific evaluation scripts. Set `CKPT_PATH` to the teacher or student checkpoint you want to evaluate.

| Task | Evaluation Script | Recommended `max_tokens` |
| --- | --- | ---: |
| Summarization | `eval/eval_summ.py` | 128 |
| Translation | `eval/eval_mt.py` | 256 |
| GSM8K-Arithmetic reasoning | `eval/eval_gsm.py` | 512 |

```bash
# Summarization
python eval/eval_summ.py -max_tokens 128 -ckpt ${CKPT_PATH}

# Translation
python eval/eval_mt.py -max_tokens 256 -ckpt ${CKPT_PATH}

# GSM8K-Arithmetic reasoning
python eval/eval_gsm.py -max_tokens 512 -ckpt ${CKPT_PATH}
```

## Acknowledgements

This codebase is built upon the implementation below. We thank the authors for releasing their code.

- *Xu, Wenda, et al. "Speculative Knowledge Distillation: Bridging the Teacher-Student Gap through Interleaved Sampling." International Conference on Learning Representations.* [Paper](https://arxiv.org/abs/2410.11325), [Code](https://github.com/google-research/google-research/tree/master/speculative_kd)

# Task-agnostic Instruction-Following Distillation

## Environment

Set up the project environment by creating a new virtual environment and installing the required packages.

```bash
bash install.sh
```

## Datasets

The raw instruction-response datasets for training and evaluation can be downloaded from [MiniLLM](https://github.com/microsoft/LMOps/tree/main/minillm).

This repository follows the MiniLLM-style data structure: raw datasets should be placed under `data/`, and preprocessed datasets for training and evaluation should be placed under `processed_data/`.

## Models

Model checkpoints should be placed under the `checkpoints/` directory.

For GPT-2 based experiments, the expected directory layout is:

```text
checkpoints/
├── gpt2-base/
├── gpt2-medium/
├── gpt2-large/
└── gpt2-xlarge/
```

Each model directory should contain the corresponding Hugging Face checkpoint files, including model weights, tokenizer files, and configuration files.

## Training

Replace `/PATH/TO/CSD/task_agnostic_distillation` with your local project path.

### Teacher SFT

Train the SFT teacher checkpoints for GPT-2 and OpenLLaMA2.

```bash
bash ./scripts/gpt2/sft/sft_xlarge.sh /PATH/TO/CSD/task_agnostic_distillation ${MASTER_PORT} ${GPU_NUM}
bash ./scripts/openllama2/sft/sft_7B_lora.sh /PATH/TO/CSD/task_agnostic_distillation ${MASTER_PORT} ${GPU_NUM}
```

Before running scripts under `scripts/gpt2/kd_CSD/`, set the teacher checkpoint path in the corresponding bash script.

```bash
TEACHER_CKPT=<PATH_TO_SFT_TEACHER_CHECKPOINT>
```

### Student Initialization

Train student initialization checkpoints. The final checkpoints are selected by validation loss.

```bash
bash ./scripts/gpt2/init/init_base.sh /PATH/TO/CSD/task_agnostic_distillation ${MASTER_PORT} ${GPU_NUM}
bash ./scripts/openllama2/init/sft_3B_lora.sh /PATH/TO/CSD/task_agnostic_distillation ${MASTER_PORT} ${GPU_NUM}
```

### CSD Training

This section covers the KD-only CSD experiments. These scripts do not use student SFT initialization or auxiliary pre-training loss.

#### Main CSD

Use this script to train the main CSD loss used in Table 1.

```bash
bash scripts/gpt2/kd_CSD/csd.sh /PATH/TO/CSD/task_agnostic_distillation ${MASTER_PORT} ${GPUS_PER_NODE} ${LR} ${BATCH_SIZE} ${LOSS_TEMP}
```

Arguments:

- `${LR}`: learning rate.
- `${BATCH_SIZE}`: batch size.
- `${LOSS_TEMP}`: temperature used inside the CSD loss weighting function.

#### CSD Temperature Ablation

Use this script for the temperature-scaling ablation in Figure 5(a).

```bash
bash scripts/gpt2/kd_CSD/csd_temperature_ablation.sh /PATH/TO/CSD/task_agnostic_distillation ${MASTER_PORT} ${GPUS_PER_NODE} ${LR} ${BATCH_SIZE} ${LOSS_TEMP}
```

#### CSD Weighting Function Variants

Use this script for the CSD pair variants in Figure 3.

```bash
bash scripts/gpt2/kd_CSD/csd_weighting_function.sh /PATH/TO/CSD/task_agnostic_distillation ${MASTER_PORT} ${GPUS_PER_NODE} ${LR} ${BATCH_SIZE} ${LOSS_TEMP} ${WEIGHT_TYPE}
```

Arguments:

- `${WEIGHT_TYPE}` selects the CSD pair variant:
  - `US`: CSD `(U,S)`, using uniform and student distributions.
  - `TS`: CSD `(T,S)`, using teacher and student distributions.
- `${LOSS_TEMP}` controls the temperature value used by the loss script.

### ImitKD + CSD Training

This section covers ImitKD + CSD experiments. These scripts use student SFT initialization and auxiliary pre-training loss.

#### GPT-2 Backbone

Use this script for the GPT-2 ImitKD + CSD experiment in Table 2.

```bash
# GPT-2-0.1B
bash scripts/gpt2/imitkd_CSD/csd_imitkd_base.sh /PATH/TO/CSD/task_agnostic_distillation ${MASTER_PORT} ${GPUS_PER_NODE} ${LR} ${BATCH_SIZE}
# GPT-2-0.3B
bash scripts/gpt2/imitkd_CSD/csd_imitkd_medium.sh /PATH/TO/CSD/task_agnostic_distillation ${MASTER_PORT} ${GPUS_PER_NODE} ${LR} ${BATCH_SIZE}
```

#### OpenLLaMA2 Backbone

Use this script for the OpenLLaMA2 ImitKD + CSD experiment in Table 2.

```bash
bash scripts/openllama2/imitkd_CSD/csd_imitkd_3B_7B_teacher_lora.sh /PATH/TO/CSD/task_agnostic_distillation ${MASTER_PORT} ${GPUS_PER_NODE} ${LR} ${BATCH_SIZE} ${GRAD_ACC}
```

Arguments:

- `${GRAD_ACC}` controls gradient accumulation steps.
- This script uses a 3B OpenLLaMA2 student and a 7B OpenLLaMA2 teacher.

### Reproduction Settings

| Result | Category | Student SFT Initialization | Auxiliary Pre-training Loss |
|---|---|---:|---:|
| Table 1: CSD (Ours) | CSD | No | No |
| Figure 5(a): Best CSD Temperature Ablation | CSD | No | No |
| Figure 3: CSD (U,S) | CSD | No | No |
| Figure 3: CSD (T,S) | CSD | No | No |
| Table 2: GPT-2-0.1B ImitKD + CSD | ImitKD + CSD | Yes | Yes |
| Table 2: GPT-2-0.3B ImitKD + CSD | ImitKD + CSD | Yes | Yes |
| Table 2: OpenLLaMA2 ImitKD + CSD | ImitKD + CSD | Yes | Yes |

### Reproduction Commands

#### CSD

```bash
# Table 1: CSD (Ours)
bash scripts/gpt2/kd_CSD/csd.sh /PATH/TO/CSD/task_agnostic_distillation ${MASTER_PORT} 1 0.0001 8 1.0

# Figure 5(a): Best CSD Temperature Ablation
bash scripts/gpt2/kd_CSD/csd_temperature_ablation.sh /PATH/TO/CSD/task_agnostic_distillation ${MASTER_PORT} 1 0.0001 8 1.5

# Figure 3: CSD (U,S)
bash scripts/gpt2/kd_CSD/csd_weighting_function.sh /PATH/TO/CSD/task_agnostic_distillation ${MASTER_PORT} 1 0.0001 8 1.0 US

# Figure 3: CSD (T,S)
bash scripts/gpt2/kd_CSD/csd_weighting_function.sh /PATH/TO/CSD/task_agnostic_distillation ${MASTER_PORT} 1 0.0001 8 1.0 TS
```

#### ImitKD + CSD

```bash
# Table 2: GPT-2-0.1B ImitKD + CSD
bash scripts/gpt2/imitkd_CSD/csd_imitkd_base.sh /PATH/TO/CSD/task_agnostic_distillation ${MASTER_PORT} 1 0.0005 32
# Table 2: GPT-2-0.3B ImitKD + CSD
bash scripts/gpt2/imitkd_CSD/csd_imitkd_medium.sh /PATH/TO/CSD/task_agnostic_distillation ${MASTER_PORT} 1 0.0001 32
# Table 2: OpenLLaMA2 ImitKD + CSD
bash scripts/openllama2/imitkd_CSD/csd_imitkd_3B_7B_teacher_lora.sh /PATH/TO/CSD/task_agnostic_distillation ${MASTER_PORT} 1 0.0001 8 1
```

## Evaluation

```bash
bash ./scripts/gpt2/eval/run_eval.sh ${GPU_NUM} /PATH/TO/CKPT ${MASTER_PORT} ${BATCH_SIZE}
```

## Acknowledgements

This codebase is built upon the implementations below. We thank the authors for releasing their code.

- *Gu, Yuxian, et al. "MiniLLM: Knowledge Distillation of Large Language Models." International Conference on Learning Representations.* [Paper](https://arxiv.org/abs/2306.08543), [Code](https://github.com/microsoft/LMOps/blob/main/minillm)
- *Ko, Jongwoo, et al. "DistiLLM: Towards Streamlined Distillation for Large Language Models." International Conference on Machine Learning.* [Paper](https://arxiv.org/abs/2402.03898), [Code](https://github.com/jongwooko/distillm)

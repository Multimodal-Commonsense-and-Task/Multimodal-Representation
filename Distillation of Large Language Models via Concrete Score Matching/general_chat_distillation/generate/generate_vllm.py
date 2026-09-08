from vllm import SamplingParams, LLM
from transformers import AutoTokenizer
from datasets import load_dataset, load_from_disk

import argparse
import json
import torch, time, json, os
from pathlib import Path
from tqdm import tqdm
from datetime import timedelta

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Decode with vLLM')
parser.add_argument('--data_dir', type=str, default="ultrachat",
                    help='Directory containing the data')
parser.add_argument('--iter', type=int, default='1', help='training iteration')
parser.add_argument('--model', type=str, default='Qwen/Qwen2.5-0.5B-Instruct', 
                    help='Path to the SLM model')
parser.add_argument('--teacher-model', type=str, default=None, 
                    help='Path to the LLM model.')
parser.add_argument('--temperature', type=float, default=0.8,
                    help='Temperature for sampling')
parser.add_argument('--top_p', type=float, default=0.95,
                    help='Top-p probability for sampling')
parser.add_argument('--eps', type=float, default=0.04,
                    help='epsilon for typical acceptance sampler')
parser.add_argument('--max_tokens', type=int, default=1024,
                    help='Maximum number of tokens to generate')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed')
parser.add_argument('--output_dir', type=str, default="datasets/phi3_ultrafeedback",
                    help='output_dir')
parser.add_argument('--split', type=str, default='train_prefs')
parser.add_argument('--frac_idx', type=int, default=0)
parser.add_argument('--frac_size', type=int, default=0)
parser.add_argument('--lora_path', type=str, default=None)

args = parser.parse_args()

data_dir = args.data_dir

# this is recommended for gemma-2 models; otherwise it is not needed
# if 'gemma-2' in args.model:
#     os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"

if args.lora_path is not None:
    llm = LLM(model=args.model, enable_lora=True)
else:
    llm = LLM(model=args.model, dtype="bfloat16", tensor_parallel_size=4,)
tokenizer = llm.get_tokenizer()


if args.data_dir == 'evol-instruct':
    data_dir = "eval/evol-instruct/evol_inst_eval.json"
    train_dataset = load_dataset('json', data_files=data_dir, split='train')
    prompts = train_dataset['prompt']
elif args.data_dir == "alpaca-eval":
    data_dir = "eval/alpacaeval/alpaca_eval.json"
    train_dataset = load_dataset('json', data_files=data_dir, split='train')
    prompts = train_dataset['instruction']
elif args.data_dir == "ultrachat":
    prompts = [
        example[0]['content'] for example in load_dataset(f'UCLA-AGI/SPIN_iter{args.iter}', split='train')['generated']
    ]
else:
    train_dataset= load_dataset(data_dir, split=args.split)
    prompts = sorted(list(set(train_dataset['prompt'])))

if args.frac_size > 0:
    assert args.frac_size > args.frac_idx
    sub_len = len(prompts) // args.frac_size + 1
    if sub_len*(args.frac_idx+1) > len(prompts):
        prompts = prompts[sub_len*args.frac_idx:]
    else:
        prompts = prompts[sub_len*args.frac_idx:sub_len*(args.frac_idx+1)]
else:
    prompts = prompts[:]

conversations = [tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=False, add_generation_prompt=True) for prompt in prompts]

sampling_params = SamplingParams(
    temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens, seed=args.seed
)
if args.lora_path is not None:
    from vllm.lora.request import LoRARequest
    outputs = llm.generate(conversations, sampling_params, lora_request=LoRARequest("lora", 1, args.lora_path))
else:
    outputs = llm.generate(conversations, sampling_params)

# Save the outputs as a JSON file.
output_data = []
for i, output in tqdm(enumerate(outputs)):
    prompt = output.prompt
    generated_text=output.outputs[0].text
    output_data.append({
        'prompt': prompts[i],
        "format_prompt": prompt,
        'generated_text': generated_text,
    })

if args.frac_size > 0:
    output_file = f'output_{args.seed}_{args.frac_idx}_{args.frac_size}.json'
else:
    output_file = f'output_{args.seed}.json'
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

with open(os.path.join(args.output_dir, output_file), 'w') as f:
    json.dump(output_data, f, indent=4)

print(f"Outputs saved to {os.path.join(args.output_dir, output_file)}")
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
parser.add_argument('--data_dir', type=str, default="HuggingFaceH4/ultrafeedback_binarized",
                    help='Directory containing the data')
parser.add_argument('--model', type=str, default='Qwen/Qwen2-0.5B-Instruct', # 'h2oai/h2o-danube2-1.8b-sft'
                    help='Path to the SLM model')
parser.add_argument('--teacher-model', type=str, default=None, # 'mistralai/Mistral-7B-Instruct-v0.2'
                    help='Path to the LLM model.')
parser.add_argument('--temperature', type=float, default=0.8,
                    help='Temperature for sampling')
parser.add_argument('--top_p', type=float, default=0.95,
                    help='Top-p probability for sampling')
parser.add_argument('--eps', type=float, default=0.09,
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
if 'gemma-2' in args.model:
    os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"

llm = LLM(model=args.model)
tokenizer = llm.get_tokenizer()
prompts = load_dataset('openai/openai_humaneval', split='test')['prompt']
task_ids = load_dataset('openai/openai_humaneval', split='test')['task_id']


conversations = [tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=False, add_generation_prompt=True) for prompt in prompts]

sampling_params = SamplingParams(
    temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens, seed=args.seed
)
outputs = llm.generate(conversations, sampling_params)

# Save the outputs as a JSON file.
output_data = []
for i, output in tqdm(enumerate(outputs)):
    prompt = output.prompt
    generated_text=output.outputs[0].text
    output_data.append({
        'task_id': task_ids[i],
        'completion': generated_text,
    })

output_file = f'output_{args.seed}.json'
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

output_file = output_file.replace("json", "jsonl")
with open(os.path.join(args.output_dir, output_file), 'w') as f:
    for entry in output_data:
        json_line = json.dumps(entry); f.write(json_line + '\n')
print(f"Outputs saved to {os.path.join(args.output_dir, output_file)}")
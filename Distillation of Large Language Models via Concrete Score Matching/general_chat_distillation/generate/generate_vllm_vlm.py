from vllm import SamplingParams, LLM
from transformers import AutoTokenizer
from datasets import load_dataset, load_from_disk
from transformers import AutoProcessor, LlavaForConditionalGeneration

import argparse
import json, os
from pathlib import Path
from tqdm import tqdm
import PIL

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Decode with vLLM for VLMs')
parser.add_argument('--data_dir', type=str, default='datasets/vlm/org')
parser.add_argument('--split', type=str, default='dev')
parser.add_argument('--model', type=str, default='llava-hf/llava-1.5-7b-hf')
parser.add_argument('--tokenizer', type=str, default='llava-hf/llava-1.5-7b-hf')
parser.add_argument('--temperature', type=float, default=0.8)
parser.add_argument('--top_p', type=float, default=0.95)
parser.add_argument('--output_dir', type=str, required=True)
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--max_tokens', type=int, default=1024, help='Maximum number of tokens to generate')
args = parser.parse_args()

processor = AutoProcessor.from_pretrained(args.tokenizer)
try: dataset = load_from_disk(args.data_dir)['train']
except: dataset = load_dataset(args.data_dir, split=args.split)
prompts, images = dataset['question'], dataset['image']

# import pdb; pdb.set_trace()
conversations = [
    {
        "prompt": processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}],
            tokenize=False, add_generaion_prompt=True
        ), "multi_modal_data": {"image": image}
    } 
    for prompt, image in zip(prompts, images)
]

sampling_params = SamplingParams(
    temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens, seed=args.seed
)

llm = LLM(model=args.model)
outputs = llm.generate(conversations, sampling_params)

output_data = []
for i, output in tqdm(enumerate(outputs)):
    prompt = output.prompt
    generated_text=output.outputs[0].text
    output_data.append({
        'prompt': prompts[i],
        "format_prompt": prompt,
        'generated_text': generated_text,
    })

output_file = f'output_{args.seed}.json'
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

with open(os.path.join(args.output_dir, output_file), 'w') as f:
    json.dump(output_data, f, indent=4)

print(f"Outputs saved to {os.path.join(args.output_dir, output_file)}")
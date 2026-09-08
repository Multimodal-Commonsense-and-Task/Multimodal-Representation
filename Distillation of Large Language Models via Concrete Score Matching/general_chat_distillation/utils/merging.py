from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForVision2Seq, AutoProcessor
from peft import PeftModel
import os
import argparse

def main(args):
    try: 
        base_model = AutoModelForCausalLM.from_pretrained(args.base_model_name) # LLMs
        tokenizer = AutoTokenizer.from_pretrained(args.base_model_name) # LLMs
    except: 
        base_model = AutoModelForVision2Seq.from_pretrained(args.base_model_name) # VLMs
        tokenizer = AutoProcessor.from_pretrained(args.base_model_name) # VLMs

    lora_model = PeftModel.from_pretrained(base_model, args.lora_model_name)
    merged_model = lora_model.merge_and_unload()
    save_path = f"{args.lora_model_name}/merged"
    merged_model.save_pretrained(save_path); tokenizer.save_pretrained(save_path)

if __name__ == "__main__":
    # For vllm 0.5.4, inference with the LoRA model is less accurate than with the merged model.
    parser = argparse.ArgumentParser(description="Merging the Base and LoRA models for accurate inference")
    parser.add_argument('--base-model-name', type=str, required=True)
    parser.add_argument('--lora-model-name', type=str, required=True)
    args = parser.parse_args()
    main(args)
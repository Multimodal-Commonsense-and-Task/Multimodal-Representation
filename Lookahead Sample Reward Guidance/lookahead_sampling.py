# primary generation script
import os
import json
import numpy as np
from PIL import Image
from tqdm import tqdm

import matplotlib.pyplot as plt

import argparse

from datetime import datetime

import torch
from diffusers import DDIMScheduler, UNet2DConditionModel, DPMSolverMultistepScheduler, StableDiffusionPipeline, StableDiffusionXLPipeline, DiffusionPipeline, LCMScheduler, AutoPipelineForText2Image, EulerDiscreteScheduler, FluxPipeline
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file


import sys
sys.path.append("fkd_diffusers")


from fks_utils import do_eval

# load prompt data
def load_geneval_metadata(prompt_path, max_prompts=None):
    if prompt_path.endswith(".json"):
        with open(prompt_path, "r") as f:
            data = json.load(f)
    else:
        assert prompt_path.endswith(".jsonl")
        with open(prompt_path, "r") as f:
            data = [json.loads(line) for line in f]
    assert isinstance(data, list)
    prompt_key = "prompt"
    if prompt_key not in data[0]:
        assert "text" in data[0], "Prompt data should have 'prompt' or 'text' key"

        for item in data:
            item["prompt"] = item["text"]
    if max_prompts is not None:
        data = data[:max_prompts]
    return data




def main(args):
    # seed everything
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # load prompt data
    prompt_data = load_geneval_metadata(args.prompt_path, max_prompts=args.max_prompt)

    if "runwayml/stable-diffusion-v1-5" in args.model_name: ## SDv1.5-DPM
        pipe = StableDiffusionPipeline.from_pretrained(args.model_name, torch_dtype=torch.float16)
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    elif "mhdang/dpo" in args.model_name and "xl" not in args.model_name: ## SDv1.5-DPO-DPM
        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
        )
        # load finetuned model
        unet_id = "mhdang/dpo-sd1.5-text2image-v1"
        unet = UNet2DConditionModel.from_pretrained(
            unet_id, subfolder="unet", torch_dtype=torch.float16
        )
        pipe.unet = unet
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    elif "stabilityai/stable-diffusion-xl-base-1.0" in args.model_name: ## SDXL-DPM
        pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16,use_safetensors=True, variant="fp16")
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
        pipe.vae.to(dtype=torch.float32)

    elif "dmd2_sdxl_4step_lora_fp16.safetensors" in args.model_name: ## SDXL-DMD-4
        base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
        repo_name = "tianweiy/DMD2"
        ckpt_name = "dmd2_sdxl_4step_lora_fp16.safetensors"
        pipe = DiffusionPipeline.from_pretrained(base_model_id, torch_dtype=torch.float16, variant="fp16").to("cuda")
        pipe.load_lora_weights(hf_hub_download(repo_name, ckpt_name))
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        pipe.fuse_lora(lora_scale=1.0)
        pipe.vae.to(dtype=torch.float32)

    elif "dmd2_sdxl_1step_unet_fp16.bin" in args.model_name: ## SDXL-DMD-1
        base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
        repo_name = "tianweiy/DMD2"
        ckpt_name = "dmd2_sdxl_1step_unet_fp16.bin"
        unet = UNet2DConditionModel.from_config(base_model_id, subfolder="unet").to("cuda", torch.float16)
        unet.load_state_dict(torch.load(hf_hub_download(repo_name, ckpt_name), map_location="cuda"))
        pipe = DiffusionPipeline.from_pretrained(base_model_id, unet=unet, torch_dtype=torch.float16, variant="fp16").to("cuda")
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        pipe.vae.to(dtype=torch.float32)

    elif "latent-consistency/lcm-lora-sdxl" in args.model_name: ## SDXL-LCM-LoRA-4
        model_id = "stabilityai/stable-diffusion-xl-base-1.0"
        adapter_id = "latent-consistency/lcm-lora-sdxl"
        pipe = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=torch.float16, variant="fp16")
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        pipe.to("cuda")
        pipe.load_lora_weights(adapter_id)
        pipe.fuse_lora()
        pipe.vae.to(dtype=torch.float32)

    elif "sdxl_lightning_4step_lora.safetensors" in args.model_name: ## SDXL-Lightning-4
        base = "stabilityai/stable-diffusion-xl-base-1.0"
        repo = "ByteDance/SDXL-Lightning"
        ckpt = "sdxl_lightning_4step_lora.safetensors"
        pipe = StableDiffusionXLPipeline.from_pretrained(base, torch_dtype=torch.float16, variant="fp16").to("cuda")
        pipe.load_lora_weights(hf_hub_download(repo, ckpt))
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
        pipe.fuse_lora()
        pipe.vae.to(dtype=torch.float32)

    elif "Hyper-SDXL-1step-Unet.safetensors" in args.model_name: ## SDXL-Hyper-SD-1
        base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
        repo_name = "ByteDance/Hyper-SD"
        ckpt_name = "Hyper-SDXL-1step-Unet.safetensors"
        unet = UNet2DConditionModel.from_config(base_model_id, subfolder="unet").to("cuda", torch.float16)
        unet.load_state_dict(load_file(hf_hub_download(repo_name, ckpt_name), device="cuda"))
        pipe = DiffusionPipeline.from_pretrained(base_model_id, unet=unet, torch_dtype=torch.float16,variant="fp16").to("cuda")
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        pipe.vae.to(dtype=torch.float32)

    elif "Hyper-FLUX.1-dev-8steps-lora.safetensors" in args.model_name: ## FLUX-Hyper-8
        base_model_id = "black-forest-labs/FLUX.1-dev"
        repo_name = "ByteDance/Hyper-SD"
        ckpt_name = "Hyper-FLUX.1-dev-8steps-lora.safetensors"
        pipe = FluxPipeline.from_pretrained(base_model_id, token=os.environ.get("HF_TOKEN"))
        pipe.load_lora_weights(hf_hub_download(repo_name, ckpt_name))
        pipe.fuse_lora(lora_scale=0.125)
        pipe.to("cuda", dtype=torch.float16)

    # set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipe.to(device)

    # set output directory
    prefix = f"{args.seed}_{args.num_particles}_{args.num_inference_steps}"
    output_dir = os.path.join(args.output_dir, f"{prefix}")

    try:
        os.makedirs(output_dir, exist_ok=False)
    except FileExistsError:
        # make file sleep for a random time
        import time

        print("Sleeping for a random time")
        time.sleep(np.random.randint(1, 10))
        prefix = f"{args.seed}_{args.num_particles}_{args.num_inference_steps}"
        output_dir = os.path.join(args.output_dir, f"{prefix}")
        os.makedirs(output_dir, exist_ok=False)

    arg_path = os.path.join(output_dir, "args.json")
    with open(arg_path, "w") as f:
        json.dump(vars(args), f, indent=4)

    metrics_to_compute = args.metrics_to_compute.split("#")

    # cache metric fns
    do_eval(
        prompt=["test"],
        images=[Image.new("RGB", (224, 224))],
        metrics_to_compute=metrics_to_compute,
    )

    metrics_arr = {
        metric: dict(mean=0, max=0, min=0, std=0) for metric in metrics_to_compute
    }
    n_samples = 0
    average_time = 0



    for prompt_idx, item in enumerate(tqdm(prompt_data)):
        prompt = [item["prompt"]] * args.num_particles

        prompt_path = os.path.join(output_dir, f"{prompt_idx:0>5}")
        os.makedirs(prompt_path, exist_ok=True)

        # dump metadata
        with open(os.path.join(prompt_path, "metadata.jsonl"), "w") as f:
            json.dump(item, f)

        start_time = datetime.now()
        with torch.inference_mode():
            if "dmd2_sdxl_4step_lora_fp16.safetensors" in args.model_name:
                latents = pipe(prompt=prompt, num_inference_steps=4, guidance_scale=0, timesteps=[999, 749, 499, 249], output_type="latent").images
            elif "dmd2_sdxl_1step_unet_fp16.bin" in args.model_name:
                latents = pipe(prompt=prompt, num_inference_steps=1, guidance_scale=0, timesteps=[399], output_type="latent").images
            elif "latent-consistency/lcm-lora-sdxl" in args.model_name:
                latents = pipe(prompt=prompt, num_inference_steps=4, guidance_scale=0, output_type="latent").images
            elif "sdxl_lightning_4step_lora.safetensors" in args.model_name:
                latents = pipe(prompt=prompt, num_inference_steps=4, guidance_scale=0, output_type="latent").images
            elif "Hyper-SDXL-1step-Unet.safetensors" in args.model_name:
                latents = pipe(prompt=prompt, num_inference_steps=1, guidance_scale=0, timesteps=[800], output_type="latent").images
            elif "Hyper-FLUX.1-dev-8steps-lora.safetensors" in args.model_name:
                latents = pipe(prompt=prompt, num_inference_steps=8, guidance_scale=3.5, output_type="latent").images
            else: ## runwayml/stable-diffusion-v1-5 // mhdang/dpo // stabilityai/stable-diffusion-xl-base-1.0
                latents = pipe(prompt=prompt, guidance_scale=7.5, num_inference_steps=args.num_inference_steps, output_type="latent").images

            if "xl" in args.model_name or "XL" in args.model_name:
                latents = latents.to(torch.float32)

            if "FLUX" in args.model_name:
                latents = pipe._unpack_latents(latents, 1024, 1024, pipe.vae_scale_factor)
                latents = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
                images = pipe.vae.decode(latents, return_dict=False)[0]
                images = pipe.image_processor.postprocess(images, output_type="pil")
            else:
                images = pipe.vae.decode( latents / pipe.vae.config.scaling_factor,return_dict=False)[0]
                images = pipe.image_processor.postprocess(images, output_type="pil")

        results = do_eval(prompt=prompt, images=images, metrics_to_compute=metrics_to_compute)
        end_time = datetime.now()
        time_taken = end_time - start_time

        results["time_taken"] = time_taken.total_seconds()
        results["prompt"] = prompt
        results["prompt_index"] = prompt_idx

        n_samples += 1
        average_time += time_taken.total_seconds()
        print(f"Time taken: {average_time / n_samples}")

        # sort images by reward
        guidance_reward = np.array(results[args.guidance_reward_fn]["result"])
        sorted_idx = np.argsort(guidance_reward)[::-1]
        images = [images[i] for i in sorted_idx]
        latents = [latents[i] for i in sorted_idx]
        for metric in metrics_to_compute:
            results[metric]["result"] = [
                results[metric]["result"][i] for i in sorted_idx
            ]

        for metric in metrics_to_compute:
            metrics_arr[metric]["mean"] += results[metric]["mean"]
            metrics_arr[metric]["max"] += results[metric]["max"]
            metrics_arr[metric]["min"] += results[metric]["min"]
            metrics_arr[metric]["std"] += results[metric]["std"]

        for metric in metrics_to_compute:
            print(
                metric,
                metrics_arr[metric]["mean"] / n_samples,
                metrics_arr[metric]["max"] / n_samples,
            )

        sample_path = os.path.join(prompt_path, "samples")
        os.makedirs(sample_path, exist_ok=True)
        torch.save(latents, os.path.join(sample_path, "latent.pt"))

        if args.save_individual_images:
            for image_idx, image in enumerate(images):
                image.save(os.path.join(sample_path, f"{image_idx:05}.png"))

            best_of_n_sample_path = os.path.join(prompt_path, "best_of_n_samples")
            os.makedirs(best_of_n_sample_path, exist_ok=True)
            for image_idx, image in enumerate(images[:1]):
                image.save(os.path.join(best_of_n_sample_path, f"{image_idx:05}.png"))
            _, ax = plt.subplots(1, args.num_particles, figsize=(20, 20))
            if args.num_particles == 1:
                ax = [ax]
            for i, image in enumerate(images):
                ax[i].imshow(image)
                ax[i].axis("off")

            plt.suptitle(prompt[0])
            image_fpath = os.path.join(prompt_path, f"grid.png")
            plt.savefig(image_fpath)
            plt.close()

        with open(os.path.join(prompt_path, "results.json"), "w") as f:
            json.dump(results, f)

    # save final metrics
    for metric in metrics_to_compute:
        metrics_arr[metric]["mean"] /= n_samples
        metrics_arr[metric]["max"] /= n_samples
        metrics_arr[metric]["min"] /= n_samples
        metrics_arr[metric]["std"] /= n_samples

    with open(os.path.join(output_dir, "final_metrics.json"), "w") as f:
        json.dump(metrics_arr, f)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="Lookahead_samples")
    parser.add_argument("--save_individual_images", type=bool, default=False)
    parser.add_argument("--num_particles", type=int, default=100)
    parser.add_argument("--num_inference_steps", type=int, default=100)
    parser.add_argument("--guidance_reward_fn", type=str, default="ImageReward")
    parser.add_argument("--metrics_to_compute",type=str,default="ImageReward#Clip-Score",help="# separated list of metrics")
    parser.add_argument("--prompt_path", type=str, default="prompt_files/geneval_metadata.jsonl")
    parser.add_argument("--model_name", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_prompt", type=int, default=1000)
    args = parser.parse_args()


    return args


if __name__ == "__main__":
    args = get_args()
    main(args)

import argparse, os, json
from datasets import load_dataset

def main(args):

    if args.data == "evol-instruct":
        data_dir = "eval/evol-instruct/WizardLM_testset.jsonl"
        placeholder = "prompt"
        reference_model = "gpt-3.5-turbo"
    elif args.data == "alpacaeval":
        # Reference answers are already included in `alpaca_eval.json`.
        # You don't have to do this.
        data_dir = "eval/alpacaeval/alpaca_eval.json"
        placeholder = "instruction"
        reference_model = "text-davinci-003"
    elif args.data == "ultrafeedback":
        data_dir = "eval/ultrafeedback/ultrafeedback_eval.json"
        placeholder = "prompt"
        reference_model = "gpt-3.5-turbo"
    else:
        raise NotImplementedError
    
    train_dataset = load_dataset('json', data_files=data_dir, split='train')
    prompts = sorted(list(set(train_dataset[placeholder])))

    jobs = []
    for i, query in enumerate(prompts):
        jobs.append(
            {"model": reference_model, "n": 1,
            "messages": [
                {"role": "user", "content": query}
            ]}
        )

    with open(f"eval/evol-instruct/{reference_model}.jsonl", "w") as f:
        for job in jobs:
            json_string = json.dumps(job)
            f.write(json_string + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="reference answer for pairwise comparison")
    parser.add_argument('--data', type=str, default='evol-instruct',
                        help='evaluation benchmark')
    args = parser.parse_args()
    main(args)

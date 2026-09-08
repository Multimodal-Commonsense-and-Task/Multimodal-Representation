import argparse, os, json
from datasets import load_dataset

parser = argparse.ArgumentParser(description="Build Evaluation Datasets")
parser.add_argument('--data-path1', type=str, required=True)
parser.add_argument('--data-path2', type=str, default=None)
parser.add_argument('--judge', type=str, default="gpt-4o-mini")
parser.add_argument('--output-file', type=str, default=None)
parser.add_argument('--pairwise', action="store_true")
args = parser.parse_args()

if args.pairwise:
    system = [
        "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below.",
        "You should choose the assistant that follows the user’s instructions and answers the user’s question better.",
        "Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses.", 
        "Begin your evaluation by comparing the two responses and provide a short explanation.", 
        "Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision.", 
        "Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants.", 
        """Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format: "[[A]]" if assistant A is better, "[[B]]" if assistant B is better, and "[[C]]" for a tie.""",
    ]

    assert args.data_path1 is not None; assert args.data_path2 is not None
    ans1 = load_dataset('json', data_files=args.data_path1, split='train')
    ans2 = load_dataset('json', data_files=args.data_path2, split='train')
    
    if 'alpacaeval' in args.data_path1:
        ans1 = ans1.rename_column("instruction", "prompt")
        ans1 = ans1.rename_column("output", "generated_text")
    if 'alpacaeval' in args.data_path2:
        ans2 = ans2.rename_column("instruction", "prompt")
        ans2 = ans2.rename_column("output", "generated_text")


    samples = []
    for a1, a2 in zip(ans1, ans2):
        if a1['prompt'] == a2['prompt']:
            samples.append([a1['prompt'], a1['generated_text'], a2['generated_text']])

    jobs = []
    for i, sample in enumerate(samples):
        question, answer_a, answer_b = sample
        query = f"""[System]\n{' '.join(system)}\n\n[User Question]\n{question}\n\n[The Start of Assistant A’s Answer]\n{answer_a}\n[The End of Assistant A’s Answer]\n\n[The Start of Assistant B’s Answer]\n{answer_b}\n[The End of Assistant B’s Answer]"""
        jobs.append({"model": args.judge, "n": 1, "messages": [{"role": "user", "content": query}]})


else:
    system = [
        "Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below.",
        "Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response.",
        "Begin your evaluation by providing a short explanation. Be as objective as possible.",
        'After providing your explanation, please rate the response on a scale of 1 to 10 by strictly following this format: "[[rating]]",",',
        'for example: "Rating: [[5]]"."'
    ]

    assert args.data_path1 is not None
    ans1 = load_dataset('json', data_files=args.data_path1, split='train')

    jobs = []
    for i, sample in enumerate(ans1):
        question, answer = sample['prompt'], sample['generated_text']
        query = f"""[System]\n{' '.join(system)}\n\n[Question]\n{question}\n\n[The Start of Assistant’s Answer]\n{answer}\n[The End of Assistant’s Answer]"""
        jobs.append({"model": args.judge, "n": 1, "messages": [{"role": "user", "content": query}]})


os.makedirs('./results/inputs', exist_ok=True)
if args.output_file is None:
    args.output_file = f"{args.data_path1}_{args.data_path1}"

with open(f"./results/inputs/{args.output_file}.jsonl", "w") as f:
    for job in jobs:
        json_string = json.dumps(job)
        f.write(json_string + "\n")
import os
import json
import datasets
from datasets import load_dataset, concatenate_datasets, DatasetDict
from tqdm import tqdm
import argparse


def main(args):
    teacher_data = load_dataset('json', data_files=args.teacher_file, split='train')
    student_data = load_dataset('json', data_files=args.student_file, split='train')

    # make sure the pair
    samples = []
    dict_teacher = {x['prompt']: str(x) for x in teacher_data}
    dict_student = {x['prompt']: str(x) for x in student_data}

    for p in teacher_data['prompt']:
        try:
            chosen, rejected = eval(dict_teacher[p]), eval(dict_student[p])
            chosen = [
                {"content": p, "role": "user"},
                {"content": chosen['generated_text'], "role": "assistant"}
            ]
            rejected = [
                {"content": p, "role": "user"},
                {"content": rejected['generated_text'], "role": "assistant"}
            ]
            samples.append({"prompt": p, "chosen": chosen, "rejected": rejected})

        except:
            continue

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    with open(f'{args.output_dir}/train.json', 'w') as json_file:
        json.dump(samples, json_file)

    dataset = DatasetDict({
        'train': load_dataset('json', data_files=f'{args.output_dir}/train.json', split='train'),
        'test': load_dataset('json', data_files=f'{args.output_dir}/train.json', split='train').select(range(500)),
    })
    dataset.save_to_disk(args.output_dir)
    print (f"Binarized datasets save to {os.path.join(args.output_dir)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_file", type=str, required=True)
    parser.add_argument("--student_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    main(args)
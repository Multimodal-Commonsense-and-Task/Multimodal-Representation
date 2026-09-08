from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import torch

def main(args):
    teacher_model = AutoModelForCausalLM.from_pretrained(args.teacher_model, torch_dtype=torch.bfloat16)
    teacher_tokenizer = AutoTokenizer.from_pretrained(args.teacher_model)

    student_model = AutoModelForCausalLM.from_pretrained(args.student_model, torch_dtype=torch.bfloat16)
    student_tokenizer = AutoTokenizer.from_pretrained(args.student_model)

    assert len(student_tokenizer) == len(teacher_tokenizer)

    teacher_model.resize_token_embeddings(len(teacher_tokenizer))
    student_model.resize_token_embeddings(len(student_tokenizer))

    teacher_directory = 'ckpts' + args.teacher_model.split('/')[-1]
    teacher_model.save_pretrained(teacher_directory)
    teacher_tokenizer.save_pretrained(teacher_directory)

    student_directory = 'ckpts' + args.student_model.split('/')[-1]
    student_model.save_pretrained(student_directory)
    student_tokenizer.save_pretrained(student_directory)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merging the Base and LoRA models for accurate inference")
    parser.add_argument('--teacher-model', type=str, required=True)
    parser.add_argument('--student-model', type=str, required=True)
    args = parser.parse_args()
    main(args)
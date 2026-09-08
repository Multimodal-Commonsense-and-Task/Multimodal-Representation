import json
import argparse

parser = argparse.ArgumentParser(description='Post-processing the judgement')
parser.add_argument('--input1', type=str, required=True)
parser.add_argument('--input2', type=str, default=None)
parser.add_argument('--pairwise', action='store_true')
args = parser.parse_args()

if args.pairwise:
    assert args.input1 is not None
    file1 = open(args.input1, 'r')
    lines1 = file1.readlines()
    samp = []
    for line in lines1:
        try: samp.append(eval(line.split('"choices": ')[1].split('"content": ')[1].split(', "refusal":')[0]))
        except: import pdb; pdb.set_trace(); samp.append(line.split('"choices": ')[1].split('"content": ')[1].split(', "refusal":')[0])

    all, win, lose, tie = 0, 0, 0, 0
    for eval in samp:
        if '[[A]]' in eval: win += 1; all += 1
        elif '[[B]]' in eval: lose += 1; all += 1
        elif '[[C]]' in eval: tie += 1; all += 1
    weighted_wr1 = (win*1+lose*0+tie*0.5)/all
        
    print (f"(M1-M2) Win : {win}/{all}, Lose: {lose}/{all}, Tie: {tie}/{all}")
    print (f"Weighted WR: {weighted_wr1}")
    print ("\n")

    assert args.input2 is not None
    file2 = open(args.input2, 'r')
    lines2 = file2.readlines()
    samp = []
    for line in lines2:
        try: samp.append(eval(line.split('"choices": ')[1].split('"content": ')[1].split(', "refusal":')[0]))
        except: samp.append(line.split('"choices": ')[1].split('"content": ')[1].split(', "refusal":')[0])
    
    all, win, lose, tie = 0, 0, 0, 0
    for eval in samp:
        if '[[A]]' in eval: lose += 1; all += 1
        elif '[[B]]' in eval: win += 1; all += 1
        elif '[[C]]' in eval: tie += 1; all += 1
    weighted_wr2 = (win*1+lose*0+tie*0.5)/all
    print (f"(M2-M1)Win : {win}/{all}, Lose: {lose}/{all}, Tie: {tie}/{all}")
    print (f"Weighted WR: {weighted_wr2}")
    print ("\n")

    print (f"Avg Weighted WR: {(weighted_wr1+weighted_wr2)/2}")

else:
    assert args.input1 is not None
    file = open(args.input1, 'r')
    lines = file.readlines()
    samp = []
    for line in lines:
        try: samp.append(eval(line.split('"choices": ')[1].split('"content": ')[1].split(', "refusal":')[0]))
        except: samp.append(line.split('"choices": ')[1].split('"content": ')[1].split(', "refusal":')[0])

    ratings = []
    for eval in samp:
        ratings.append(int(eval.split('[[')[-1].split(']]')[0]))
    print (f'Avg Single Grading: {(sum(ratings)/len(ratings))}')
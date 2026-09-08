import os
import re
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='parse result')
parser.add_argument('--model', type=str)
parser.add_argument('--method', type=str)
args = parser.parse_args()

model = args.model
method = args.method
method_split = method.split("/")
name = ""
name_log = ""
for idx in range(len(method_split)):
    if idx == len(method_split) - 1:
        name = f"{name[:-1]}/{method_split[-1]}"
        name_log = f"{name_log[:-1]}#{method_split[-1]}"
    else:
        name += f"{method_split[idx]}_"
        name_log += f"{method_split[idx]}_"

metric_list = ["dolly-512", "self_inst-512", "vicuna-512", "sinst_11_-512", "uinst_11_-512"]
seed_list = [10, 20, 30, 40, 50]

output_path = os.path.join("./results", model, "eval_main", "_summary")
if not os.path.exists(output_path):
    os.makedirs(output_path, exist_ok=True)

print (name)

if os.path.exists(os.path.join("./results", model, "eval_main", metric_list[-1], name, str(seed_list[-1]), "log.txt")):
    f = open(os.path.join(output_path, f"{name_log}.txt"), "w", encoding="utf-8")
    for metric in metric_list:
        res = []
        for seed in seed_list:
            log_path = os.path.join("./results", model, "eval_main", metric, name, str(seed), "log.txt")
            with open(log_path, "r", encoding="utf-8") as _f:
                for line in _f:
                    m = re.search(r"'rougeL'\s*:\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)", line)
                    if m:
                        rougeL = float(m.group(1))
            res.append(rougeL)
        log = f"{metric:<15} | {np.mean(res):.2f} ({np.std(res):.2f}) | {res}\n"
        print (log)
        f.write(log)
    f.close()
else:
    print ("Not yet")
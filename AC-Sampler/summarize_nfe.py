"""Print the average number of function evaluations (NFE) per sample.

generate_ac_sampler.py writes <outdir>/nfe_analysis.json. Pass one or more
sample directories to compare their NFE cost.
"""

import json
import os
import sys

if len(sys.argv) < 2:
    print(f'usage: python {sys.argv[0]} <sample_dir> [<sample_dir> ...]')
    sys.exit(1)

print(f'{"directory":<50} {"samples":>10} {"total NFE":>12} {"NFE/sample":>12}')
for d in sys.argv[1:]:
    path = os.path.join(d, 'nfe_analysis.json')
    if not os.path.exists(path):
        print(f'{d:<50} (no nfe_analysis.json)')
        continue
    with open(path) as f:
        info = json.load(f)
    n = info.get('total_samples', 0)
    nfe = info.get('total_nfe', 0)
    print(f'{d:<50} {n:>10d} {nfe:>12d} {nfe / max(n, 1):>12.2f}')

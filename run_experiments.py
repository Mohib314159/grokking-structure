"""Run the locked experimental grid: identical recipe for every group.

Locked settings (chosen so the slowest group, S4, fully groks in budget):
frac=0.7, width=512, lr=2e-3, wd=1.0, AdamW full-batch, max 150k epochs.
"""
import json
import sys
from train import run_one

SETTINGS = dict(frac=0.7, width=512, lr=2e-3, wd=1.0,
                max_epochs=150_000, verbose_every=20_000)

if __name__ == "__main__":
    group = sys.argv[1]
    seeds = [int(s) for s in sys.argv[2:]] or [0, 1, 2]
    for s in seeds:
        r = run_one(group, s, **SETTINGS)
        path = f"results/{group}_seed{s}.json"
        with open(path, "w") as f:
            json.dump(r, f)
        print(f"{group} seed {s}: T_mem={r['T_mem']} T_gen={r['T_gen']} "
              f"delay={r['delay']} final_val={r['final_val_acc']:.3f} "
              f"({r['wall_seconds']}s)", flush=True)

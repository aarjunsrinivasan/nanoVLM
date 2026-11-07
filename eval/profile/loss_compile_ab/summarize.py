"""Summarize the loss-path x compile 2x2 from the per-run JSONs written by run.sh.

    python eval/profile/loss_compile_ab/summarize.py [results_dir]

Reads <dir>/<arm>_r<rep>.json (arm in E_full, E_gather, C_full, C_gather; E = eager, C = compile),
prints one row per arm (mean over reps, with the rep-to-rep range) and the four speedups a 2x2
supports. Also flags any log that hit dynamo's recompile_limit: past that point a compiled arm
silently runs eager for the shapes it has no graph for, which would understate compile.
"""
import glob
import json
import os
import re
import sys
from collections import defaultdict
from statistics import mean

ARMS = ["E_full", "E_gather", "C_full", "C_gather"]


def load(out_dir):
    runs = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(out_dir, "*_r*.json"))):
        m = re.fullmatch(r"(E|C)_(full|gather)_r(\d+)\.json", os.path.basename(path))
        if not m:
            continue
        with open(path) as f:
            data = json.load(f)
        ((impl, v),) = data["variants"].items()
        runs[f"{m[1]}_{m[2]}"].append({"rep": int(m[3]), "log": path[:-5] + ".log", **v["steady_state"]})
    return runs


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
    runs = load(out_dir)
    if not runs:
        sys.exit(f"no <arm>_r<rep>.json under {out_dir}")

    print(f"{'arm':<9}{'reps':>5}{'tok/s':>10}{'range':>19}{'fw+bw s':>9}{'data s':>8}{'alloc GiB':>11}{'rsvd GiB':>10}{'last loss':>11}")
    tps = {}
    for arm in ARMS:
        rs = runs.get(arm)
        if not rs:
            continue
        t = [r["mean_tokens_per_second"] for r in rs]
        tps[arm] = mean(t)
        print(f"{arm:<9}{len(rs):>5}{mean(t):>10,.0f}{f'{min(t):,.0f}-{max(t):,.0f}':>19}"
              f"{mean(r['avg_fw_bw_time_s'] for r in rs):>9.3f}{mean(r['avg_data_load_time_s'] for r in rs):>8.3f}"
              f"{max(r['peak_mem_allocated_gib'] for r in rs):>11.2f}{max(r['peak_mem_reserved_gib'] for r in rs):>10.2f}"
              f"{mean(r['last_batch_loss'] for r in rs):>11.4f}")

    def ratio(label, num, den):
        if num in tps and den in tps:
            print(f"  {label:<38}{tps[num] / tps[den]:.3f}x")

    print("\nSpeedup in tokens/s")
    ratio("gather vs full, eager", "E_gather", "E_full")
    ratio("gather vs full, compile", "C_gather", "C_full")
    ratio("compile vs eager, full", "C_full", "E_full")
    ratio("compile vs eager, gather", "C_gather", "E_gather")
    ratio("gather+compile vs full+eager", "C_gather", "E_full")

    hits = [r["log"] for rs in runs.values() for r in rs
            if os.path.exists(r["log"]) and "recompile_limit" in open(r["log"], errors="replace").read()]
    print("\nrecompile_limit hit in: " + (", ".join(hits) if hits else "no run"))


if __name__ == "__main__":
    main()

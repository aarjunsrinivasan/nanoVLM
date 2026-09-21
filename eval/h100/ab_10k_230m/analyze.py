"""Tables for the 10k-step A/B: per-seed val losses under both masks, the seed spread, and speed.

Usage: python eval/h100/ab_10k_230m/analyze.py <dir with A_s0_try0.log, C_s0_try0.log, A_s1_try0.log, C_s1_try0.log>

The decision rule, fixed before the runs: the fork's improvement counts only if the gap between the arms is larger
than the spread between seeds within an arm.
"""
import os
import re
import statistics as st
import sys

ARMS = {"A": "baseline (full / none / eager)", "C": "fork (gather / flex / compile)"}
SEEDS = (0, 1)


def parse(path):
    txt = open(path).read()
    evals = [(int(m.group(1)), float(m.group(2)), float(m.group(3)))
             for m in re.finditer(r"Step: (\d+), Val Loss: [\d.]+, doc_masked val loss: ([\d.]+), unmasked val loss: ([\d.]+)", txt)]
    final = re.search(r"Final full val pass: doc_masked val loss: ([\d.]+), unmasked val loss: ([\d.]+)", txt)
    wall = re.search(r"Step: \d+/\d+, Train Loss: ([\d.]+) \| Time: ([\d.]+)s", txt)
    tok = [(int(m.group(1)), int(m.group(2)), float(m.group(3)))
           for m in re.finditer(r"Step: (\d+), Loss: [\d.]+, Tokens/s: (\d+), fw_bw: ([\d.]+)", txt)]
    return {
        "evals": evals,
        "final": (float(final.group(1)), float(final.group(2))) if final else None,
        "train_loss": float(wall.group(1)) if wall else None,
        "wall_s": float(wall.group(2)) if wall else None,
        "tok": tok,
    }


def main(d):
    runs = {}
    for arm in ARMS:
        for seed in SEEDS:
            p = os.path.join(d, f"{arm}_s{seed}_try0.log")
            if os.path.exists(p):
                runs[(arm, seed)] = parse(p)
    if not runs:
        sys.exit(f"no run logs found in {d}")

    print("## Final full val pass (whole val set, same evaluator for every arm)\n")
    print("| arm | seed | doc-masked | unmasked | train loss | wall clock |")
    print("|---|---|---|---|---|---|")
    for (arm, seed), r in sorted(runs.items()):
        if r["final"]:
            print(f"| {ARMS[arm]} | {seed} | {r['final'][0]:.4f} | {r['final'][1]:.4f} | "
                  f"{r['train_loss']:.4f} | {r['wall_s'] / 3600:.2f} h |")

    print("\n## Decision rule\n")
    for metric, idx in (("doc-masked", 0), ("unmasked", 1)):
        per_arm = {arm: [runs[(arm, s)]["final"][idx] for s in SEEDS if (arm, s) in runs and runs[(arm, s)]["final"]]
                   for arm in ARMS}
        if not all(len(v) == len(SEEDS) for v in per_arm.values()):
            print(f"{metric}: waiting for all seeds")
            continue
        spread = max(max(v) - min(v) for v in per_arm.values())
        gap = st.mean(per_arm["A"]) - st.mean(per_arm["C"])
        verdict = "larger than the seed spread" if abs(gap) > spread else "WITHIN the seed spread, so not conclusive"
        print(f"- **{metric}**: mean A {st.mean(per_arm['A']):.4f}, mean C {st.mean(per_arm['C']):.4f}, "
              f"gap {gap:+.4f} (fork better when positive); largest within-arm seed spread {spread:.4f} → {verdict}")

    print("\n## Val loss over training (doc-masked, 256 rows)\n")
    steps = sorted({s for r in runs.values() for s, _, _ in r["evals"]})
    cols = [f"{a}_s{s}" for a in ARMS for s in SEEDS if (a, s) in runs]
    print("| step | " + " | ".join(cols) + " |")
    print("|---" * (len(cols) + 1) + "|")
    for step in steps[::4]:
        cells = []
        for a in ARMS:
            for s in SEEDS:
                if (a, s) in runs:
                    v = [d for st_, d, _ in runs[(a, s)]["evals"] if st_ == step]
                    cells.append(f"{v[0]:.4f}" if v else "-")
        print(f"| {step} | " + " | ".join(cells) + " |")

    print("\n## Speed over the run (identical data in both arms)\n")
    for lo, hi in ((0, 2500), (2500, 5000), (5000, 7500), (7500, 10000)):
        line = []
        for seed in SEEDS:
            if ("A", seed) in runs and ("C", seed) in runs:
                a = {s: t for s, t, _ in runs[("A", seed)]["tok"]}
                c = {s: t for s, t, _ in runs[("C", seed)]["tok"]}
                w = [s for s in a if s in c and lo <= s < hi and s > 0]
                if w:
                    line.append(f"seed {seed}: {st.mean(c[s] / a[s] for s in w):.2f}x")
        print(f"- steps {lo}-{hi}: " + ", ".join(line))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__)))

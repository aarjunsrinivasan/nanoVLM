import argparse
import glob
import hashlib
import json
import os

import numpy as np
from scipy import stats

from lmms_eval.api.metrics import paired_ttest

RUNS = ["A_s0", "C_s0", "A_s1", "C_s1"]

# task -> (per-sample metric key, cluster source)
#   ("media",)                              image URL logged in the sample's `input_media`
#   ("col", path, name, split, column)      id column of the HF dataset, joined on doc_id
#   ("image", path, name, split)            md5 of the image bytes, joined on doc_id
TASKS = {
    "docvqa_val": ("anls", ("col", "lmms-lab-encoder/DocVQA", "DocVQA", "validation", "docId")),
    "infovqa_val": ("anls", ("media",)),
    "chartqa": ("relaxed_overall", ("image", "lmms-lab-encoder/ChartQA", None, "test")),
    "textvqa_val": ("exact_match", ("media",)),
    "ocrbench": ("ocrbench_accuracy", ("image", "echo840/OCRBench", None, "test")),
    "ai2d": ("exact_match", ("image", "lmms-lab-encoder/ai2d", None, "test")),
    "scienceqa": ("exact_match", ("image", "lmms-lab-encoder/ScienceQA", "ScienceQA-FULL", "test")),
}
MDE_Z = 1.96 + 0.8416  # two-sided alpha=0.05, 80% power


def latest(root, run, pattern):
    files = sorted(glob.glob(os.path.join(root, run, pattern)))
    if not files:
        raise FileNotFoundError(f"{os.path.join(root, run, pattern)}")
    return files[-1]


def load_samples(root, run, task):
    with open(latest(root, run, f"*_samples_{task}.jsonl")) as f:
        rows = [json.loads(line) for line in f]
    return sorted(rows, key=lambda r: r["doc_id"])


def score_of(row, metric):
    v = row[metric]
    return float(v["score"] if isinstance(v, dict) else v)


def lmms_stderr(root, run, task, metric):
    """lmms-eval's own stderr for this task (naive, else CLT), in score units; nan if absent."""
    res = json.load(open(latest(root, run, "*_results.json")))["results"].get(task, {})
    for key in res:
        if key.startswith(metric + ","):
            flt = key.split(",", 1)[1]
            for suffix in ("_stderr", "_stderr_clt"):
                v = res.get(f"{metric}{suffix},{flt}")
                if isinstance(v, (int, float)):
                    return float(v)
    return float("nan")


def load_clusters(task, spec, rows):
    """One cluster label per row, or None if it can't be built (then question-level only)."""
    if spec[0] == "media":
        return np.array([str(r["input_media"][0]) if r.get("input_media") else f"q{r['doc_id']}" for r in rows])
    import datasets

    _, path, name, split = spec[:4]
    try:
        for kw in ({"token": True}, {}):
            try:
                ds = datasets.load_dataset(path, name, split=split, **kw)
                break
            except Exception as e:  # noqa: BLE001
                err = e
        else:
            raise err
        sub = ds.select([r["doc_id"] for r in rows])
        if "question" in sub.column_names:  # guard against a doc_id -> row misalignment
            hit = np.mean([str(q).strip()[:30] in r["input"] for r, q in zip(rows, sub["question"])])
            if hit < 0.95:
                raise RuntimeError(f"doc_id/row alignment check failed ({hit:.0%} of questions match)")
        if spec[0] == "col":
            return np.array([str(v) for v in sub[spec[4]]])
        imgs = sub.cast_column("image", datasets.Image(decode=False)).select_columns(["image"])
        out = []
        for i, row in enumerate(imgs):
            im = row["image"]
            raw = (im.get("bytes") or (im.get("path") or "").encode()) if im else b""
            out.append(hashlib.md5(raw).hexdigest() if raw else f"noimg{i}")
        return np.array(out)
    except Exception as e:  # noqa: BLE001
        print(f"[warn] {task}: no cluster ids ({e}); question-level only")
        return None


def cluster_sums(d, labels):
    _, inv = np.unique(labels, return_inverse=True)
    return np.bincount(inv, weights=d), np.bincount(inv).astype(float)


def boot_ci(S, n, B, rng, chunk=500):
    G, means = len(S), np.empty(B)
    for s in range(0, B, chunk):
        k = min(chunk, B - s)
        idx = rng.integers(0, G, size=(k, G))
        means[s : s + k] = S[idx].sum(1) / n[idx].sum(1)
    return tuple(np.percentile(means, [2.5, 97.5]))


def compare(x, y, labels, binary, rng, B):
    """Paired comparison of x - y on identical questions; `labels` = cluster id per question or None."""
    d = x - y
    N = len(d)
    out = {"n": N, "delta": float(d.mean()), "mean_x": float(x.mean()), "mean_y": float(y.mean())}

    S, n = cluster_sums(d, labels if labels is not None else np.arange(N))
    G, dbar = len(S), d.mean()
    se = np.sqrt(G / max(G - 1, 1) * np.sum((S - dbar * n) ** 2)) / N
    if se == 0:
        p = 1.0 if dbar == 0 else 0.0
    else:
        p = float(2 * stats.t.sf(abs(dbar / se), max(G - 1, 1)))
    out.update(n_clusters=G, se_cluster=float(se), p_cluster=p, mde=float(MDE_Z * se), ci_cluster=boot_ci(S, n, B, rng))

    Sq, nq = cluster_sums(d, np.arange(N))
    out["ci_question"] = boot_ci(Sq, nq, B, rng)
    out["p_ttest"] = float(paired_ttest(list(x), list(y))["p_value"])
    if binary:
        b, c = int(np.sum((x == 1) & (y == 0))), int(np.sum((x == 0) & (y == 1)))
        out["mcnemar_b_c"] = (b, c)
        out["p_mcnemar"] = float(stats.binomtest(b, b + c, 0.5).pvalue) if b + c else 1.0
    return out


def holm(p):
    order = sorted(p, key=p.get)
    m, run, adj = len(order), 0.0, {}
    for i, k in enumerate(order):
        run = max(run, min(1.0, (m - i) * p[k]))
        adj[k] = run
    return adj


def analyze_task(task, metric, spec, root, B, rng, use_clusters):
    data = {r: load_samples(root, r, task) for r in RUNS}
    ref = data["A_s0"]
    for r in RUNS[1:]:
        same = len(data[r]) == len(ref) and all(
            a["doc_id"] == b["doc_id"] and a["input"] == b["input"] and str(a["target"]) == str(b["target"])
            for a, b in zip(ref, data[r])
        )
        if not same:
            raise ValueError(f"{task}: {r} rows do not match A_s0 (different questions/order)")
    sc = {r: np.array([score_of(x, metric) for x in data[r]]) for r in RUNS}
    binary = all(np.isin(v, (0.0, 1.0)).all() for v in sc.values())
    labels = load_clusters(task, spec, ref) if use_clusters else None

    pairs = {
        "effect_s0": (sc["C_s0"], sc["A_s0"]),
        "effect_s1": (sc["C_s1"], sc["A_s1"]),
        "effect_pooled": ((sc["C_s0"] + sc["C_s1"]) / 2, (sc["A_s0"] + sc["A_s1"]) / 2),
        "ctrl_A": (sc["A_s1"], sc["A_s0"]),
        "ctrl_C": (sc["C_s1"], sc["C_s0"]),
    }
    res = {k: compare(x, y, labels, binary and k != "effect_pooled", rng, B) for k, (x, y) in pairs.items()}
    empty = {r: float(np.mean([not str(x["filtered_resps"]).strip() for x in data[r]])) for r in RUNS}
    return {
        "metric": metric,
        "n": len(ref),
        "binary": binary,
        "clustered": labels is not None,
        "scores": {r: float(sc[r].mean()) for r in RUNS},
        "lmms_stderr": {r: lmms_stderr(root, r, task, metric) for r in RUNS},
        "empty_frac": empty,
        "cmp": res,
    }


def verdict(cmp):
    e0, e1, pooled = cmp["effect_s0"], cmp["effect_s1"], cmp["effect_pooled"]
    excl = lambda e: e["ci_cluster"][0] > 0 or e["ci_cluster"][1] < 0  # noqa: E731
    floor = max(abs(cmp["ctrl_A"]["delta"]), abs(cmp["ctrl_C"]["delta"]))
    sig = e0["p_holm"] < 0.05 and e1["p_holm"] < 0.05
    if e0["delta"] * e1["delta"] > 0 and excl(e0) and excl(e1) and sig and abs(pooled["delta"]) > floor:
        return "credible: C > A" if pooled["delta"] > 0 else "credible: A > C"
    return "inconclusive"


def pts(v):
    return f"{100 * v:+.2f}"


def ci_str(ci):
    return f"[{100 * ci[0]:+.2f}, {100 * ci[1]:+.2f}]"


def write_report(results, path):
    tasks = list(results)
    holm_p = {s: holm({t: results[t]["cmp"][f"effect_{s}"]["p_cluster"] for t in tasks}) for s in ("s0", "s1")}
    for t in tasks:
        for s in ("s0", "s1"):
            results[t]["cmp"][f"effect_{s}"]["p_holm"] = holm_p[s][t]
        results[t]["verdict"] = verdict(results[t]["cmp"])

    L = ["# lmms-eval: doc masking A/B (step_10000)", ""]
    L += [
        "Scores in points (x100). Effect = C (doc-masked) - A (no masking); every metric is higher-is-better.",
        "CIs are 95% cluster-bootstrap (clusters = images/documents when known, else questions).",
        "p_holm = cluster-robust t-test p-value, Holm-corrected across tasks within a seed pair.",
        "",
        "## Scores (mean, lmms-eval stderr)",
        "",
        "| task | N | " + " | ".join(RUNS) + " | max empty resp |",
        "|---|--:|" + "--:|" * len(RUNS) + "--:|",
    ]
    for t in tasks:
        r = results[t]
        se = lambda k: "" if np.isnan(r["lmms_stderr"][k]) else f" ±{100 * r['lmms_stderr'][k]:.2f}"  # noqa: E731
        cells = [f"{100 * r['scores'][k]:.2f}{se(k)}" for k in RUNS]
        flag = " **!**" if max(r["empty_frac"].values()) > 0.01 else ""
        L.append(f"| {t} | {r['n']} | " + " | ".join(cells) + f" | {100 * max(r['empty_frac'].values()):.2f}%{flag} |")

    L += [
        "",
        "## Paired differences",
        "",
        "| task | clusters | Δ seed0 [CI] p_holm | Δ seed1 [CI] p_holm | Δ pooled [CI] | \\|ctrl A\\| | \\|ctrl C\\| | MDE | verdict |",
        "|---|--:|---|---|---|--:|--:|--:|---|",
    ]
    for t in tasks:
        c = results[t]["cmp"]
        cell = lambda e: f"{pts(e['delta'])} {ci_str(e['ci_cluster'])} p={e['p_holm']:.3g}"  # noqa: E731
        L.append(
            f"| {t} | {c['effect_s0']['n_clusters']} | {cell(c['effect_s0'])} | {cell(c['effect_s1'])} | "
            f"{pts(c['effect_pooled']['delta'])} {ci_str(c['effect_pooled']['ci_cluster'])} | "
            f"{100 * abs(c['ctrl_A']['delta']):.2f} | {100 * abs(c['ctrl_C']['delta']):.2f} | "
            f"{100 * c['effect_pooled']['mde']:.2f} | {results[t]['verdict']} |"
        )

    L += [
        "",
        "## Secondary tests (question-level, uncorrected)",
        "",
        "| task | McNemar p s0 (b/c) | McNemar p s1 (b/c) | paired-t p s0 | paired-t p s1 | CI width question vs cluster (s0, pts) |",
        "|---|---|---|--:|--:|---|",
    ]
    for t in tasks:
        c = results[t]["cmp"]
        mc = lambda e: f"{e['p_mcnemar']:.3g} ({e['mcnemar_b_c'][0]}/{e['mcnemar_b_c'][1]})" if "p_mcnemar" in e else "n/a"  # noqa: E731
        e0 = c["effect_s0"]
        w = lambda ci: 100 * (ci[1] - ci[0])  # noqa: E731
        L.append(
            f"| {t} | {mc(e0)} | {mc(c['effect_s1'])} | {e0['p_ttest']:.3g} | {c['effect_s1']['p_ttest']:.3g} | "
            f"{w(e0['ci_question']):.2f} vs {w(e0['ci_cluster']):.2f} |"
        )

    L += [
        "",
        "## How to read this",
        "",
        "- `credible` needs the same sign in both seed pairs, both cluster CIs excluding 0, both p_holm < 0.05, and",
        "  |pooled Δ| larger than both same-arm cross-seed differences (|ctrl A|, |ctrl C|), which show what a change of",
        "  training seed alone does.",
        "- These tests measure eval-set noise for these 4 trained models. Training-seed noise cannot be estimated from",
        "  2 seeds per arm; the controls are the only guide to it. `inconclusive` means undetectable at this MDE, not no effect.",
        "- MDE = smallest pooled Δ detectable at 80% power, alpha 0.05 (2.8 x cluster SE).",
    ]
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")


def selftest():
    rng = np.random.default_rng(0)
    N = 2000
    labels = np.repeat(np.arange(N // 4), 4)
    x = (rng.random(N) < 0.6).astype(float)
    r = compare(x, x, labels, True, rng, 500)
    assert r["delta"] == 0 and r["p_cluster"] == 1.0 and r["p_ttest"] == 1.0 and r["p_mcnemar"] == 1.0, r
    assert r["ci_cluster"][0] <= 0 <= r["ci_cluster"][1], r
    y = x.copy()
    y[(x == 1) & (rng.random(N) < 0.10)] = 0
    r = compare(y, x, labels, True, rng, 500)
    assert r["delta"] < 0 and r["p_cluster"] < 0.01 and r["p_mcnemar"] < 0.01 and r["ci_cluster"][1] < 0, r
    assert holm({"a": 0.01, "b": 0.04, "c": 0.5}) == {"a": 0.03, "b": 0.08, "c": 0.5}
    print("selftest OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="eval_results/lmms_ab")
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--boot", type=int, default=10000, help="bootstrap resamples")
    ap.add_argument("--no_clusters", action="store_true", help="question-level only (skip dataset lookups)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()

    rng = np.random.default_rng(0)
    results = {}
    for task in args.tasks.split(","):
        metric, spec = TASKS[task]
        print(f"[{task}] analysing", flush=True)
        results[task] = analyze_task(task, metric, spec, args.root, args.boot, rng, not args.no_clusters)
    write_report(results, os.path.join(args.root, "summary.md"))
    with open(os.path.join(args.root, "paired_stats.json"), "w") as f:
        json.dump(results, f, indent=1, default=float)
    print(f"wrote {args.root}/summary.md and paired_stats.json")


if __name__ == "__main__":
    main()

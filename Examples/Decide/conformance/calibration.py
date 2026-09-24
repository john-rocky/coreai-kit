#!/usr/bin/env python3
"""Calibration of a decision model's probabilities on SemIf's authored144 fixture, from
`decide-cli oracle` output.

    swift run -c release decide-cli oracle --model minicpm5-2b --fixture authored144.jsonl --out minicpm5-2b.jsonl
    python3 calibration.py authored144.jsonl minicpm5-2b.jsonl [--applied 1.0] [--bins 10]

Prints, per family and overall: n, accuracy, NLL, Brier (multi-class, sum over options of
(p − 1[correct])², as SemIf's evaluate.py), top-label ECE (equal-width bins on the winning
probability), and the temperature that minimises NLL when applied to the reported
probabilities (p ∝ p^(1/T)). `--applied` is the temperature the kit already divided the
logits by (the model card's; 1.0 when none), so the proposed card value is applied × fitted.
Standard library only.
"""
import argparse
import json
import math
from collections import defaultdict


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def scale(p, t):
    logs = [math.log(max(x, 1e-9)) / t for x in p]
    m = max(logs)
    e = [math.exp(x - m) for x in logs]
    z = sum(e)
    return [x / z for x in e]


def metrics(rows, bins):
    n = len(rows)
    correct = sum(r["correct"] for r in rows)
    nll = sum(-math.log(max(r["p_label"], 1e-12)) for r in rows) / n
    brier = sum(r["brier"] for r in rows) / n
    buckets = defaultdict(list)
    for r in rows:
        buckets[min(bins - 1, int(r["conf"] * bins))].append(r)
    ece = sum(len(b) / n * abs(sum(x["correct"] for x in b) / len(b) - sum(x["conf"] for x in b) / len(b))
              for b in buckets.values())
    # balanced accuracy = mean recall over the gold classes (option ids) present, as SemIf's evaluate.py
    recalls = []
    for label in sorted({r["gold"] for r in rows}):
        part = [r for r in rows if r["gold"] == label]
        recalls.append(sum(r["correct"] for r in part) / len(part))
    return dict(n=n, acc=correct / n, bal=sum(recalls) / len(recalls), nll=nll, brier=brier, ece=ece,
                conf=sum(r["conf"] for r in rows) / n)


def score(gold, preds, t):
    rows = []
    for g in gold:
        p = preds[g["id"]]
        ids = [o["id"] for o in g["options"]]
        assert p["option_ids"] == ids, g["id"]
        probs = scale(p["probabilities"], t) if t != 1.0 else list(p["probabilities"])
        best = max(range(len(probs)), key=probs.__getitem__)
        # the gold class is the option id, not its position: the fixture shuffles option order per row
        rows.append(dict(family=g["family"], gold=ids[g["label"]], correct=best == g["label"], conf=probs[best],
                         p_label=probs[g["label"]],
                         brier=sum((q - (k == g["label"])) ** 2 for k, q in enumerate(probs))))
    return rows


def fit_temperature(gold, preds):
    grid = [math.exp(x / 40) for x in range(-60, 100)]  # 0.22 … 12
    best = min(grid, key=lambda t: sum(-math.log(max(r["p_label"], 1e-12)) for r in score(gold, preds, t)))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fixture")
    ap.add_argument("predictions")
    ap.add_argument("--applied", type=float, default=1.0, help="temperature the kit already applied (card value)")
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    gold = read_jsonl(a.fixture)
    preds = {r["id"]: r for r in read_jsonl(a.predictions)}
    missing = [g["id"] for g in gold if g["id"] not in preds]
    assert not missing, f"{len(missing)} fixture rows have no prediction"
    t = fit_temperature(gold, preds)
    out = {"model": next(iter(preds.values())).get("model"), "bins": a.bins, "applied_temperature": a.applied,
           "fitted_temperature": round(t, 3), "proposed_card_temperature": round(a.applied * t, 3), "tables": {}}
    for name, temp in (("as reported", 1.0), (f"scaled by {t:.2f}", t)):
        rows = score(gold, preds, temp)
        fam = defaultdict(list)
        for r in rows:
            fam[r["family"]].append(r)
        table = {f: metrics(v, a.bins) for f, v in sorted(fam.items())}
        overall = metrics(rows, a.bins)
        overall["bal"] = sum(m["bal"] for m in table.values()) / len(table)  # mean family balanced accuracy
        table["all"] = overall
        out["tables"][name] = table
    if a.json:
        print(json.dumps(out, indent=1))
        return
    print(f"model {out['model']}   fitted temperature {t:.2f} on the reported probabilities "
          f"(applied {a.applied} → proposed {a.applied * t:.2f})   ECE bins {a.bins}")
    for name, table in out["tables"].items():
        print(f"\n{name}")
        print("| family | n | accuracy | balanced acc. | NLL | Brier | ECE | mean conf. |")
        print("|---|---:|---:|---:|---:|---:|---:|---:|")
        for f, m in table.items():
            print(f"| {f} | {m['n']} | {m['acc']:.3f} | {m['bal']:.3f} | {m['nll']:.3f} | {m['brier']:.3f} | {m['ece']:.3f} | {m['conf']:.3f} |")


if __name__ == "__main__":
    main()

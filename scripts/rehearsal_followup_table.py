"""Aggregate the four rehearsal follow-up settings into one table (SYNTHETIC data).

Usage: python scripts/rehearsal_followup_table.py results/rehearsal_followup
Reads <setting>/run_bundle.jsonl and rehearsal_summary.json of rate_ratio|poisson_tail x plain|age_matched,
writes followup_runs.csv, followup_table.csv and followup.md next to them.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

root = Path(sys.argv[1])  # results/rehearsal_followup
SETTINGS = [("rate_ratio", "plain"), ("poisson_tail", "plain"), ("rate_ratio", "age_matched"),
            ("poisson_tail", "age_matched")]
rows = []
for stat, age in SETTINGS:
    d = root / f"{stat}__{age}"
    if not (d / "run_bundle.jsonl").is_file():
        continue
    summ = json.loads((d / "rehearsal_summary.json").read_text())
    for line in (d / "run_bundle.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r.get("test") is None:
            continue
        lvl, seed = r["tags"]["level"], r["seed"]
        info = summ["realisations"].get(lvl, {}).get(str(seed), {})
        rec = {"temporal_stat": stat, "injection": age, "level": lvl, "seed": seed, "method": r["method"],
               "auroc": r["test"]["auroc"], "ap": r["test"]["ap"], "n_pos_scored": r["test"]["n_pos"],
               "n_pos_all": r.get("test_n_pos_all"),
               "age_med_pos": (info.get("dst_age_days_median") or {}).get("positives"),
               "age_med_neg": (info.get("dst_age_days_median") or {}).get("negatives")}
        tc = r.get("test_components") or {}
        if r["method"] == "full":
            for c in ("s_embed", "s_geom", "s_temp", "s_curvdyn"):
                rec[f"auroc_{c}"] = (tc.get(c) or {}).get("auroc")
        if r["method"] in ("full", "full_learned", "no_curvature"):
            rec["weights"] = json.dumps(r.get("weights"), sort_keys=True)
        rows.append(rec)
runs = pd.DataFrame(rows).sort_values(["temporal_stat", "injection", "level", "method", "seed"])
runs.to_csv(root / "followup_runs.csv", index=False, float_format="%.6g")


def ms(x):
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], dtype=float)
    if not len(x):
        return "n/a"
    return f"{x.mean():.3f} ± {x.std(ddof=1):.3f}" if len(x) > 1 else f"{x[0]:.3f}"


table = []
for (stat, age), g in runs.groupby(["temporal_stat", "injection"], sort=False):
    for lvl, h in g.groupby("level"):
        piv = h.pivot_table(index="seed", columns="method", values=["auroc", "ap"])
        full = h[h["method"] == "full"].set_index("seed")
        delta_auc = (piv["auroc"]["full"] - piv["auroc"]["no_curvature"]).to_numpy()
        delta_ap = (piv["ap"]["full"] - piv["ap"]["no_curvature"]).to_numpy()
        table.append({
            "s_temp": stat, "injection": age, "level": lvl,
            "pos scored / all": f"{full['n_pos_scored'].mean():.1f} / {full['n_pos_all'].mean():.1f}",
            "cited age med pos / neg (d)": f"{full['age_med_pos'].mean():.0f} / {full['age_med_neg'].mean():.0f}",
            "full AUROC": ms(piv["auroc"]["full"]), "full AP": ms(piv["ap"]["full"]),
            "Δ full−no_curv AUROC": ms(delta_auc), "Δ full−no_curv AP": ms(delta_ap),
            "learned AUROC": ms(piv["auroc"]["full_learned"]), "learned AP": ms(piv["ap"]["full_learned"]),
            "s_temp alone AUROC": ms(full["auroc_s_temp"]),
            "tier-1 avg AUROC": ms(piv["auroc"]["tier1_rank_average"]),
            "XGBoost AUROC": ms(piv["auroc"]["tier2_xgb"]), "XGBoost AP": ms(piv["ap"]["tier2_xgb"]),
            "cited-age AUROC": ms(piv["auroc"]["dst_age"]), "cited-age AP": ms(piv["ap"]["dst_age"]),
            "random AP": ms(piv["ap"]["random"]),
        })
tab = pd.DataFrame(table)
tab.to_csv(root / "followup_table.csv", index=False)
cols = list(tab.columns)
md = ["# Rehearsal follow-up (2026-10-31): Poisson-tail s_temp and age-matched cartels", "",
      "> SYNTHETIC DATA: every number below comes from synthetic streams with planted cartels; none is a "
      "measurement on real citations. n_works 3000, levels moderate and subtle, seeds 0 and 1, mean ± sd "
      "over the two seeds; test period, the full model's scored test events for every method.", "",
      "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
md += ["| " + " | ".join(str(v) for v in r) + " |" for r in tab.itertuples(index=False)]
(root / "followup.md").write_text("\n".join(md) + "\n")
print("\n".join(md))

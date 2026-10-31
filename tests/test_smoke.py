"""S0.3 smoke: every module imports, default config loads (see IMPLEMENTATION_PLAN.md)."""

import importlib

from citecurv.config import load_config

MODULES = [
    "citecurv",
    "citecurv.config",
    "citecurv.data.openalex",
    "citecurv.data.abstracts",
    "citecurv.data.doi",
    "citecurv.data.rwdb",
    "citecurv.data.slice_query",
    "citecurv.data.synthetic",
    "citecurv.embed.specter2",
    "citecurv.curvature.forman_directed",
    "citecurv.curvature.orc",
    "citecurv.curvature.screen",
    "citecurv.curvature.timeslice",
    "citecurv.tgn.convert",
    "citecurv.tgn.train",
    "citecurv.tgn.infer",
    "citecurv.scoring.components",
    "citecurv.scoring.combiner",
    "citecurv.scoring.aggregate",
    "citecurv.eval.metrics",
    "citecurv.eval.split",
    "citecurv.eval.inject",
    "citecurv.eval.ablation",
    "citecurv.eval.baselines.tier1_stats",
    "citecurv.eval.baselines.tier2_xgb",
    "citecurv.eval.baselines.pygod_tier",
    "citecurv.viz",
    "citecurv.viz.explorer",
    "citecurv.viz.precompute",
    "citecurv.viz.graphctx",
]


def test_all_modules_import():
    for name in MODULES:
        importlib.import_module(name)


def test_default_config_loads_with_seed_list():
    cfg = load_config()
    assert cfg["seed_list"] == [0, 1, 2, 3, 4]

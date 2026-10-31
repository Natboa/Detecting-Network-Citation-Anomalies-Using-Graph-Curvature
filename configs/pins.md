# Version pins per stage environment

Copied from IMPLEMENTATION_PLAN.md §9.3. Notebooks and venvs install exactly these;
never `pip install` unpinned in a notebook.

| Stage env | Pins |
|-----------|------|
| data | python 3.12 · pyalex 0.21 · pandas · pyarrow · requests |
| embed | adapters 1.3.0 · transformers ==4.57.6 exactly (Colab ships 5.x; 2026-10 refresh) · torch 2.x CUDA · fallback SciNCL on plain transformers |
| curv | GraphRicciCurvature 0.5.3.2 · POT 0.9.6 · networkit 11.2.1 · scipy ≤1.13.1 · networkx |
| tgn | vendored DyGLib @3aacc36 + third_party/DyGLib/PATCHES.md · torch 2.x CUDA (pure PyTorch — no PyG); CPU smoke passed on torch 2.15 |
| baseline | pygod 1.1.0 (+ 4 runtime fixes in pygod_tier.PYGOD_ADJUSTMENTS) · torch_geometric 2.6.1 · torch-scatter 2.1.2 + torch-sparse 0.6.18 (or pyg-lib) — required by DOMINANT/CoLA samplers · torch 2.x · xgboost (3.3+ on py3.12) · scikit-learn 1.9 |
| curvgad (sealed) | own env, never merged: torch 2.2.1 · numpy 1.26.4 · dgl 2.1.0 · torchdata 0.7.1 · geoopt 0.5.1 · torch_geometric 2.6.1 · GraphRicciCurvature 0.5.3.2 · scipy ≤1.13.1 · pyyaml · pydantic (installs+imports on CPU; upstream requirements.txt is broken — see eval/baselines/curvgad_notes.md) |

Local dev venv (`.venv`, Windows, py3.12) uses the unpinned dev extras from
`pyproject.toml` — it only runs the pure-Python modules and unit tests.

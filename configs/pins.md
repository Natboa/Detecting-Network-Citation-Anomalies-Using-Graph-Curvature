# Version pins per stage environment

Copied from IMPLEMENTATION_PLAN.md §9.3. Notebooks and venvs install exactly these;
never `pip install` unpinned in a notebook.

| Stage env | Pins |
|-----------|------|
| data | python 3.12 · pyalex 0.21 · pandas · pyarrow · requests |
| embed | adapters 1.3.0 (pins transformers ~4.57.x) · torch 2.x CUDA |
| curv | GraphRicciCurvature 0.5.3.2 · POT 0.9.6 · networkit 11.2.1 · scipy ≤1.13.1 · networkx |
| tgn | vendored DyGLib · torch 2.x CUDA (pure PyTorch — no PyG) |
| baseline | pygod 1.1.0 · torch_geometric 2.5–2.6 · torch 2.x · xgboost (3.3+ on py3.12) · scikit-learn 1.9 |
| curvgad (sealed) | whatever karish-grover/curvgad requirements.txt says, in its own env, never merged |

Local dev venv (`.venv`, Windows, py3.12) uses the unpinned dev extras from
`pyproject.toml` — it only runs the pure-Python modules and unit tests.

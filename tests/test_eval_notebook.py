"""S4.3 Task 8: notebooks/50_eval.ipynb drives the full-run matrix.

Every code cell parses once its shell / magic lines are replaced by `pass` at their own indentation, and the notebook
names the pieces the plan requires (TGN folders of the full run, the ORC child job, the strict memo, the memo loader,
the shuffle gate with drift, the split check, the watchdog, the Drive flush, the BLAS pin). The shuffle cell defines
every gate name it writes.
"""

import ast
import json
import re
from pathlib import Path

NB = Path(__file__).resolve().parents[1] / "notebooks" / "50_eval.ipynb"
REQUIRED = ["TGN_full_seed", "orc_job", "require_orc_memo=True", "load_orc_memo", "gate_shuffle_tgn", "split.yaml",
            "MAX_HOURS", "flush_and_unmount", "OPENBLAS_CORETYPE", "start_new_session=True", "killpg",
            "return_degrees=True", "aliases", "shuffle_components_record(", "orc_memo_sha256", "SMOKE_PEAK",
            "redirect_stdout", "gates_repro_0", "drift_tol"]


def code_cells():
    nb = json.loads(NB.read_text(encoding="utf-8"))
    return [("".join(c["source"]) if isinstance(c["source"], list) else c["source"])
            for c in nb["cells"] if c["cell_type"] == "code"]


def runnable(src: str) -> str:
    return "\n".join(re.sub(r"^(\s*)[!%].*$", r"\1pass", line) for line in src.splitlines())


def test_eval_notebook_cells_compile_and_name_the_full_run():
    cells = code_cells()
    assert cells
    for i, src in enumerate(cells):
        ast.parse(runnable(src), filename=f"50_eval cell {i}")
    text = "\n".join(cells)
    for name in REQUIRED:
        assert name in text, name
    shuffle = [c for c in cells if "gate_shuffle_tgn(" in c]
    assert len(shuffle) == 1
    tree = ast.parse(runnable(shuffle[0]))
    assigned = {t.id for node in ast.walk(tree) if isinstance(node, ast.Assign) for t in node.targets
                if isinstance(t, ast.Name)}
    gate_keys = re.findall(r'"gates":\s*\{([^}]*)\}', shuffle[0])
    for block in gate_keys:
        for name in re.findall(r":\s*([A-Za-z_][A-Za-z0-9_]*)", block):
            assert name in assigned, f"shuffle cell writes gate {name!r} it does not define"


def test_install_cell_makes_the_running_kernel_import_the_package(tmp_path, monkeypatch):
    """Colab 2026-11-25 (C3): the kernel started before `pip install -e` wrote its .pth, so `import citecurv` found the
    repo root /content/citecurv as a namespace package and `citecurv.eval` was missing. After the install cell the
    running kernel must import the package from REPO_DIR/src, even with a stale namespace `citecurv` already loaded."""
    import os
    import sys
    import types

    cell = [c for c in code_cells() if "pip install -q -e" in c]
    assert len(cell) == 1
    pkg = tmp_path / "src" / "citecurv"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("MARKER = 'repo src'\n")
    (pkg / "eval").mkdir()
    (pkg / "eval" / "__init__.py").write_text("")
    src = runnable(cell[0]).replace('COMMIT = "SET-COMMIT"', 'COMMIT = "abc1234"')
    src = re.sub(r'^REPO_DIR = .*$', lambda m: f'REPO_DIR = {tmp_path.as_posix()!r}', src, flags=re.M)
    monkeypatch.setenv("CITECURV_GPU_STEP", "1")  # skip the ORC pins check: nothing is installed in this test
    saved_mods = {k: v for k, v in sys.modules.items() if k == "citecurv" or k.startswith("citecurv.")}
    saved_path = list(sys.path)
    stale = types.ModuleType("citecurv")
    stale.__path__ = [str(tmp_path)]  # the namespace package the kernel saw: repo root, no __init__.py
    try:
        for k in saved_mods:
            del sys.modules[k]
        sys.modules["citecurv"] = stale
        exec(compile(src, "50_eval install cell", "exec"), {"os": os, "sys": sys})
        import citecurv
        import citecurv.eval  # noqa: F401
        assert getattr(citecurv, "MARKER", None) == "repo src"
    finally:
        for k in [k for k in sys.modules if k == "citecurv" or k.startswith("citecurv.")]:
            del sys.modules[k]
        sys.modules.update(saved_mods)
        sys.path[:] = saved_path


def test_report_child_starts_after_the_inputs_are_freed():
    c6 = [c for c in code_cells() if "citecurv.eval.report_s43" in c]
    assert len(c6) == 1
    assert c6[0].index("free_inputs()") < c6[0].index("citecurv.eval.report_s43")


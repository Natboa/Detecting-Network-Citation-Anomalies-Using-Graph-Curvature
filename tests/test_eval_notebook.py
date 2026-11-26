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
            "redirect_stdout", "gates_repro_0", "drift_tol",
            # option A (plan 2026-11-26-orc-dir-closed-form, Task 4)
            "reference_overlap(", "real_orcdir", "coupling=coupling()", "citing_uncited"]


def code_cells():
    nb = json.loads(NB.read_text(encoding="utf-8"))
    return [("".join(c["source"]) if isinstance(c["source"], list) else c["source"])
            for c in nb["cells"] if c["cell_type"] == "code"]


def indexed_code_cells():
    """(notebook cell index, source) of every code cell; the index is the DOM `.cell` index Colab shows."""
    nb = json.loads(NB.read_text(encoding="utf-8"))
    return [(i, "".join(c["source"]) if isinstance(c["source"], list) else c["source"])
            for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code"]


# folder paths only: "no_temporal_keep_temp" is also a run-group tag and a report alias, which stay
OLD_ROOT = re.compile(r'matrix"\s*/\s*"(?:real|no_temporal_keep_temp)"|matrix/(?:real|no_temporal_keep_temp)(?!_orcdir)')


def test_only_cell_4_names_the_matrix_roots_and_they_are_the_orcdir_ones():
    """Option A: the rerun writes to real_orcdir / no_temporal_keep_temp_orcdir; no other cell names the C3 roots of
    2026-11-25 in any form (Python paths, shell lines), so old and new records never mix."""
    cells = dict(indexed_code_cells())
    assert 'MATRIX_ROOT = EVAL / "matrix" / "real_orcdir"' in cells[4]
    assert 'KEEP_ROOT = EVAL / "matrix" / "no_temporal_keep_temp_orcdir"' in cells[4]
    for i, src in cells.items():
        if i != 4:
            assert not OLD_ROOT.search(src), f"cell {i} names an old matrix root: {OLD_ROOT.search(src).group(0)}"


def test_matrix_calls_pass_the_coupling():
    cells = dict(indexed_code_cells())
    for i in (13, 14, 20):
        src = cells[i]
        calls = len(re.findall(r"\brun_matrix\(|\brun_pipeline\(", src))
        assert calls and src.count("coupling=") >= calls, f"cell {i}: {calls} calls, {src.count('coupling=')} couplings"
        assert "MEMO = memo(" not in src and "load_orc_memo(" not in src, f"cell {i} still loads an ORC memo"
        assert "orc_memo=" not in src and "require_orc_memo" not in src, f"cell {i} still passes an ORC memo"


def test_no_orc_job_outside_c1_and_c2():
    """C4 is dropped (its cell raises first), C5a no longer runs the shuffled ORC job (option A computes no ORC)."""
    call = re.compile(r"(?<!def )\brun_orc_job\(")
    for i, src in indexed_code_cells():
        if call.search(src) and i not in (9, 11):
            first = next(line for line in src.splitlines() if line.strip() and not line.lstrip().startswith("#"))
            assert first.startswith("raise "), f"cell {i} runs an ORC job"
    assert not call.search(dict(indexed_code_cells())[18])


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
    # the cell sets CITECURV_DYGLIB_ROOT under REPO_DIR (here tmp_path); record it so the monkeypatch undo restores it
    monkeypatch.delenv("CITECURV_DYGLIB_ROOT", raising=False)
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


def test_install_cell_test_leaves_the_environment_alone(tmp_path):
    """Colab C0 2026-11-26: the install-cell test exec'd the cell, which sets CITECURV_DYGLIB_ROOT to a path under the
    test's tmp_path and never restored it, so tests/test_tgn_infer.py run later in the same process failed with
    FileNotFoundError (DyGLib). After the test (and its monkeypatch undo) the variable is what it was before."""
    import os

    import pytest as _pytest

    key = "CITECURV_DYGLIB_ROOT"
    before = os.environ.get(key)
    mp = _pytest.MonkeyPatch()
    try:
        test_install_cell_makes_the_running_kernel_import_the_package(tmp_path, mp)
    finally:
        mp.undo()
    assert os.environ.get(key) == before


def test_report_child_starts_after_the_inputs_are_freed():
    c6 = [c for c in code_cells() if "citecurv.eval.report_s43" in c]
    assert len(c6) == 1
    assert c6[0].index("free_inputs()") < c6[0].index("citecurv.eval.report_s43")



def test_run_tags_record_the_coupling_not_an_unused_orc_memo():
    """Final review of option A: the closed cells read no ORC memo, so tags_for hashes a memo file only when one is
    passed, and records the coupling (rows and a hash of its counts) that produced the scores; the shuffled run of
    C5c records its own coupling."""
    cells = dict(indexed_code_cells())
    assert 'memo_path or EVAL / "orc_matrix.parquet"' not in cells[4]
    assert "coupling_frame" in cells[4] and '"coupling": ' in cells[4]
    assert "coupling_frame=shuf_coupling" in cells[20]
    for i in (13, 14):
        before_tags = cells[i].split("tags_for(")[0]
        assert "coupling()" in before_tags, f"cell {i}: the coupling must exist before the first tags_for call"

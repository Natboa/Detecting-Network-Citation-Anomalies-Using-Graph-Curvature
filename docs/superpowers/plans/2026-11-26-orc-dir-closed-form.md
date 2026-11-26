# Option A: one-sided directed Ollivier-Ricci curvature in s_geom (closed form) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Score the curvature part with the event-time directed Ollivier-Ricci curvature (one-sided: lower curvature = more suspicious) for every citation, computed by its exact closed form, and retarget s_curvdyn to it, so the S4.3 matrix can be rerun (C3) with the formulation that carries the label signal.

**Architecture:** A new component variant `"orc_dir"` and a new curvature mode `"closed"`: the curvature stage computes the reference overlap (`curvature.overlap.reference_overlap`, same graph state and context as the Forman screen) and `orc_dir = variants.orc_directed_uncited(n_shared, n_refs_dst, alpha)`, plus the screen's `in_deg_src` to flag rows where Proposition 2's premise fails. The matrix cells `full`, `no_temporal`, `full_learned` point at component `"orc_dir"` + mode `"closed"`; component `"full"` keeps its code and stays reachable through a new cell `plan_full` outside the matrix, so the old path and its tests survive. `forman_only` is the "old curvature" row of the table (Maor 2026-11-26).

**Tech Stack:** Python 3.12, numpy, pandas, pytest; Colab CPU High-RAM for Task 0 and the rerun.

**Spec:** this plan's Design section; `docs/curvature_formulation_study.md` section 2 (Proposition 2) and section 6; ledger G7 in `PROGRESS.md` (Maor 2026-11-03: "a better formula may come later if this one proves weak"); Maor in session local_674ef4b3 on 2026-11-26: "ok, go with option A" (after the C3 diagnosis recorded in Task 0) and "yes, keep the old row".

## Design (decisions this plan rests on)

1. **Value.** `orc_dir(e) = -(1 - alpha) * (1 - n_shared / n_refs_dst)`, alpha = `curvature.orc.alpha` (0.5), NaN when `n_refs_dst == 0` (unknown reference list, ledger G13) and on self-loop rows (overlap.py already gives NaN there). Exact library value whenever the citing work has no citer at t (Proposition 2; tests/test_curvature_variants.py already checks closed form vs library).
2. **Premise rows.** `citing_cited_at_t = in_deg_src > 0` from `forman_screen(..., return_degrees=True)` (valid on NaN-F rows and under both duplicate rules). On those rows the closed form is an approximation: they keep the value, the stage stats report `n_citing_cited_at_t` and `share_citing_cited_at_t` (denominator: rows with a finite `orc_dir`), and C6 gets a masked view `citing_uncited`. Library ORC for them is a follow-up, only if the share measured in Task 0 is not small.
3. **Direction.** s_geom = `-z(orc_dir)` against the existing reference cell (`_cell_z`: field of src x year, expanding, ts < t), no absolute value. Evidence: pilot S2.2, the 2026-11-19 full-data pass (descriptive, no split), and Task 0's held-out check (global z, validation non-cold rows). Task 0's record must exist before the push and the rerun.
4. **s_curvdyn.** Target `orc_dir` (raw, no asinh), same ridge model and train-only fit; value `(pred - obs) / rmse` (observed curvature lower than the endpoints predict = positive); NaN where `orc_dir` is NaN.
5. **Cells.** `full`, `no_temporal`, `full_learned`: component `"orc_dir"`, curvature `"closed"`. `forman_only`, `no_curvature`: unchanged (rerun into the new root, so one table has one config hash). `ollivier_only`: unchanged, out of the matrix (C4 dropped). New `plan_full`: component `"full"`, curvature `"screen+orc"` (the old `full`), outside `ABLATION_VARIANTS` and the notebook; existing tests that need the ORC path use it.
6. **Run identity.** `resolve_config` adds `scoring.components.formulation = "orc_dir_closed_v1"` (review round 2: NOT under `curvature`, because `orc_settings_hash` hashes the whole `curvature` dict and would then refuse the C2 ORC memo `orc_matrix.parquet` and any later ORC job on the real data). `run_config_hash` covers `scoring`, so every cell's config_hash changes and resume never accepts a C3 folder; `orc_settings_hash` stays equal. The string is bumped whenever the closed-mode or orc_dir component code changes meaning.
7. **Coupling input.** `run_pipeline(..., coupling=None)`: a precomputed `reference_overlap` frame may be passed; the closed mode accepts it only if its event_id, src_id, dst_id and t equal the canonical events' row for row and `n_refs_dst` equals the distinct-neighbour out-degree of the cited work (`forman_screen(..., count_duplicates=False, return_degrees=True)["out_deg_dst"]`, whatever the run's own duplicate setting), else `ValueError` with a clear message (also when a column such as `t` is missing, as in the Drive file `overlap_cache.parquet`). This catches the shuffled stream, a missing or smaller context and another stream. Other modes ignore a passed coupling. When None the closed mode computes it with the run's context.
8. **Closed mode bookkeeping.** `select_escalation` treats `"closed"` like `"screen"` (escalated cleared, the usual selection stats); screen thresholds are still fitted and written to thresholds.json and are unused by this mode (said in the docstring). The C6 control `ctl_coupling_share` ranks like raw `-orc_dir`: the records say it is the component before the cell z, not an independent control.

## Global Constraints

- No labels enter any component; per-event inputs read ts <= t minus the scored row; reference statistics read ts < t (docstring of `scoring/components.py`).
- `score_events` and `build_context` + `score_event` must agree (contract in `tests/test_components.py`).
- Test-first: each new test is seen failing for the expected reason before its code is written.
- Commits on `main`, files staged by name, message from a file with `git commit -F`, ending with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`; the public repo's diff is scanned before each commit; no push without Maor's word.
- Laptop suite baseline: 24 failures, all for missing scipy / sklearn / xgboost; anything else is a regression.

## Review Focus

1. Unknown reference list (11.48% of full-run events): `orc_dir`, s_geom, s_curvdyn NaN, never 0. Tests in Tasks 1 and 2.
2. A coupling frame from another stream (shuffled timestamps, other context) must raise. Test in Task 1.
3. Premise flag on a NaN-F row and with `screen_count_duplicates=True`. Test in Task 1.
4. Resume across formulations refuses a C3 folder. Test in Task 3.
5. Cold-start rows rank by `-orc_dir` through `cold_start_scores`. Test in Task 2.

---

### Task 0: Record the fair test (gate for push and rerun)

**Files:** `PROGRESS.md` (session line), scratchpad scripts `fair_overlap_test.py` / `check_fair_test.py` (already written and checked on synthetic data 2026-11-26).

- [ ] **Step 1:** When Chrome is free (the GDL lab session sends "Chrome free"), run 50_eval cells 1-4 (COMMIT 625c68f) on a CPU High-RAM runtime, then the `%%writefile` cell and the run cell; the run self-releases.
- [ ] **Step 2:** Read `full/eval/fair_overlap_test.json`: the self-check passed, the validation direction, the primary interval (combo - s_embed, AP and AUROC, paper-cluster bootstrap over five seeds), the premise share, the strata controls.
- [ ] **Step 3:** Write a PROGRESS line with those numbers and the C3 component AUROCs read on 2026-11-26 from `full_0/metrics.json` (s_embed 0.5867, s_temp 0.4936, s_geom 0.4517, s_curvdyn 0.4475). Commit locally.
- [ ] **Step 4:** If the validation direction is not "low overlap = suspicious", or the premise share is above 5%, stop and bring it to Maor before Task 1.

### Task 1: Closed mode in the curvature stage

**Files:**
- Modify: `src/citecurv/pipeline.py` (`resolve_config`, `select_escalation`, `curvature_stage`, `run_pipeline` signature and call)
- Modify: `src/citecurv/curvature/screen.py` (`ORC_DIR_COLUMNS = ["n_refs_dst", "n_shared", "orc_dir", "citing_cited_at_t"]` beside `CURVATURE_COLUMNS`, which stays unchanged)
- Test: `tests/test_pipeline_orc_dir.py` (new)

**Interfaces:**
- Consumes: `reference_overlap(events, context=None) -> DataFrame[event_id, n_refs_dst, n_shared, overlap]` (canonical order); `orc_directed_uncited(n_shared, n_refs_v, alpha)`; `forman_screen(..., return_degrees=True)` columns `in_deg_src`, `out_deg_dst`.
- Produces: `curvature_stage(..., coupling=None)` returns, for mode `"closed"`, `CURVATURE_COLUMNS + ORC_DIR_COLUMNS` (`escalated` False, `orc` NaN) and stats including `n_orc_dir_known`, `n_citing_cited_at_t`, `share_citing_cited_at_t`, `n_selected`, `n_eligible`, `orc_subsampled`. `run_pipeline(..., coupling=None)`. `resolve_config(...)["scoring"]["components"]["formulation"] == "orc_dir_closed_v1"`.

- [ ] **Step 1: Write the failing tests** (hand-built frames as in `tests/test_overlap.py`; each test sets `rcfg["variant_spec"]["curvature"] = "closed"` by hand, since no variant uses it before Task 3):
  - `test_closed_mode_orc_dir_equals_closed_form`: `orc_dir == orc_directed_uncited(n_shared, n_refs_dst, 0.5)` where `n_refs_dst > 0`, NaN where 0 and on a self-loop row; `n_refs_dst`, `n_shared` equal `reference_overlap(events, context=ctx)`.
  - `test_premise_flag_from_in_degree`: src u has a same-t citer and a duplicate citer row; one row has an unknown reference list (NaN F); run once with `screen_count_duplicates` False and once True; `citing_cited_at_t` equals `in_deg_src > 0` from `forman_screen(return_degrees=True)` and is True exactly on u's rows in both runs; the share's denominator is the finite-`orc_dir` count.
  - `test_context_rows_enter_the_coupling`: v's references exist only as context rows.
  - `test_coupling_from_other_stream_raises`: a coupling computed on `shuffle_test_timestamps(events, cutoff, seed=1)` (same ids) raises `ValueError` on the unshuffled run; one with a missing id raises; one without a `t` column raises `ValueError` (not KeyError); `reference_overlap(events, context=ctx)` passed in gives the same frame as computing it, and a spy shows `reference_overlap` is not called then.
  - `test_premise_flag_from_in_degree` also includes a cited work with a duplicated reference row, run with `screen_count_duplicates=True`: a correct coupling is accepted.
  - `test_formulation_changes_config_hash`: `resolve_config(cfg)["scoring"]["components"]["formulation"] == "orc_dir_closed_v1"`; `run_config_hash` with the key differs from the same config with the key removed; `orc_settings_hash` is equal with and without it.
  - `test_resume_refuses_old_formulation_folder` (moved here from Task 3, review round 2: it fails before the key exists): run a cell once with `resolve_config` monkeypatched to omit the key, then `run_matrix(..., resume=True)` unpatched raises (resume reads config_hash from metrics.json, ablation.py around line 198). Use the fake ORC function the existing ablation tests pass as `orc_fn`.
- [ ] **Step 2: Run** `.venv/Scripts/python -m pytest tests/test_pipeline_orc_dir.py -q` — Expected: FAIL (unknown curvature mode / no `coupling`).
- [ ] **Step 3: Implement** Design 1, 2, 7, 8: in `curvature_stage`, mode `"closed"` calls the screen with `return_degrees=True` (degrees kept out of `CURVATURE_COLUMNS`), validates or computes the coupling, fills `ORC_DIR_COLUMNS`; `select_escalation` handles `"closed"` as `"screen"`; `resolve_config` adds the formulation key.
- [ ] **Step 4: Run** the new file plus `tests/test_screen.py tests/test_overlap.py tests/test_pipeline_lazy_store.py tests/test_pipeline_orc_precompute.py` — Expected: PASS.
- [ ] **Step 5: Commit** `feat(curvature): closed-form directed ORC mode in the curvature stage`.

### Task 2: Component variant "orc_dir" (additive; component "full" untouched)

**Files:**
- Modify: `src/citecurv/scoring/components.py` (`VARIANTS` gains `"orc_dir"`; `_canonical_events` copies `orc_dir` and `citing_cited_at_t` when present; `_uses` gains the `"orc_dir"` case; `_curv_target` and `fit_curvdyn_model` accept target `"orc_dir"`; `_prepare` picks it for that variant; `score_events`, `build_context`, `score_event` compute Design 3-4 for it, with no `arcsinh` for this target; trace columns and `ScoreBreakdown.inputs` gain `orc_dir`, `z_orc_dir`, `citing_cited_at_t`, while the orc_dir reference statistics reuse the existing `geom_*` reference columns and `_REF_STATS` is NOT changed (it is zipped positionally at two places); module docstring)
- Modify: `src/citecurv/viz/component_text.py` (the s_geom and s_curvdyn entries describe both formulations, plan Forman and orc_dir, since the explorer also shows `forman_only` and old runs)
- Test: `tests/test_components.py` (extend)

**Interfaces:**
- Consumes: Task 1's frame.
- Produces: `score_events(..., variant="orc_dir")`; it raises `ValueError` when the curvature frame has no `orc_dir` column. Component `"full"` output is unchanged.

- [ ] **Step 1: Write the failing tests**:
  - `test_orc_dir_s_geom_is_one_sided`: in one reference cell with enough earlier rows, `orc_dir` -0.45 scores above -0.05, and s_geom equals `-(orc_dir - mean) / sd` of the cell's earlier rows.
  - `test_orc_dir_nan_stays_nan`: NaN `orc_dir` gives NaN s_geom and s_curvdyn.
  - `test_curvdyn_orc_dir_residual_is_signed`: observed below prediction gives a positive s_curvdyn, above gives a negative one.
  - `test_orc_dir_needs_the_column`: `variant="orc_dir"` with a screen+orc frame raises.
  - `test_paths_agree_orc_dir`: the existing agreement test parametrised with `variant="orc_dir"` and an `orc_dir` column.
  - `test_cold_rows_rank_by_minus_orc_dir`: through `pipeline.cold_start_scores`, lower `orc_dir` gives higher `total_cold`.
  - `test_orc_dir_trace_and_breakdown`: `with_inputs=True` columns and `score_event(...).inputs` carry `orc_dir`, `z_orc_dir`, `citing_cited_at_t`.
- [ ] **Step 2: Run** `tests/test_components.py -q` — Expected: the new tests FAIL; every existing test PASSES.
- [ ] **Step 3: Implement** in both paths.
- [ ] **Step 4: Run** `tests/test_components.py tests/test_pipeline_integration.py tests/test_explorer.py` — Expected: PASS (minus the sklearn baseline failures among them).
- [ ] **Step 5: Commit** `feat(scoring): orc_dir component variant, one-sided s_geom and s_curvdyn`.

### Task 3: Cells, the old path under plan_full, and the test migration

**Files:**
- Modify: `src/citecurv/eval/ablation.py` (`VARIANTS`: `full`, `no_temporal`, `full_learned` -> component `"orc_dir"`, curvature `"closed"`; new `plan_full` = component `"full"`, curvature `"screen+orc"`, not in `ABLATION_VARIANTS`; module docstring table)
- Modify (exact migration list, review round 2; rule: tests of the old path's semantics move to `plan_full`, tests of the matrix get a closed-mode version):
  - `tests/test_ablation.py` `test_variant_specs_are_consistent` (lines 28-37): `EXTRA_CELLS` gains `plan_full` (or line 30 is edited), the allowed curvature modes gain `"closed"`;
  - `tests/test_ablation.py` lines 147-170 (166-168 assert that `no_temporal` and `full_learned` reuse `full`'s ORC): rewrite as "closed cells compute no ORC; `plan_full` does". It is one of the laptop's sklearn baseline failures, so it is verified only at Colab C0;
  - `tests/test_pipeline_integration.py` `test_value_ranges_and_cold_start_rule` (110-137, default `variant="full"`): keep it on `plan_full` and add a closed-mode twin (orc all NaN, `orc_dir` in [-(1-alpha), 0], s_geom and s_curvdyn signed);
  - `tests/test_pipeline_integration.py` around 258 and 316-320 (`n_orc_computed > 0`): `plan_full`;
  - `tests/test_pipeline_lazy_store.py`: the `stream` fixture (55-74) resolves `variant="full"`; point it at `plan_full` (tests 77-140, 161-173, 191);
  - `tests/test_pipeline_orc_precompute.py` runs with `variant="full"`: `plan_full`;
  - `tests/test_report_s43.py:227` builds a hand-made `full_0` folder and needs no change.
- Test: `tests/test_ablation.py` (extend)

- [ ] **Step 0:** Before any change, run the laptop suite and save the names of the failing tests to `<workspace>/baseline_failures.txt`; "the known failures only" is compared by name from here on.
- [ ] **Step 1: Write the failing tests**:
  - `test_full_cell_runs_closed_mode_end_to_end`: `run_matrix(["full", "forman_only", "no_curvature"], [0], data, tmp, orc_fn=<the fake ORC the existing ablation tests use>)` on the existing synthetic fixture: `full` metrics.json has `curvature.mode == "closed"`, `n_orc_computed == 0`, `share_citing_cited_at_t` present; `forman_only` has `mode == "screen"`.
  - `test_coupling_passed_once_gives_same_scores`: `full_0` has `curvature.mode == "closed"` and its scores.parquet with `coupling=` equals the one computed without it; a spy shows `reference_overlap` is not called when the coupling is passed.
  - `test_plan_full_keeps_the_old_path`: `plan_full` gives `mode == "screen+orc"` and `n_orc_computed > 0` on the fixture.
- [ ] **Step 2: Run** them — Expected: FAIL.
- [ ] **Step 3: Implement** the VARIANTS change and migrate the tests listed above.
- [ ] **Step 4: Run** the whole laptop suite into a file and compare the failing names with `baseline_failures.txt` — Expected: the same names, nothing new.
- [ ] **Step 5: Commit** `feat(eval): matrix cells use the closed-form directed ORC; old path kept as plan_full`.

### Task 4: Notebook 50_eval for the rerun

**Files:**
- Modify: `notebooks/50_eval.ipynb`:
  - cell 4: `MATRIX_ROOT = EVAL / "matrix" / "real_orcdir"` and `KEEP_ROOT = EVAL / "matrix" / "no_temporal_keep_temp_orcdir"`;
  - cell 5: `coupling()` computes `reference_overlap(INPUTS["events"], context=INPUTS["context"])` once, lazily, under `MEM.set("coupling")`, prints RSS after it and keeps it in `INPUTS`;
  - cells 13 and 14 (C3, C3 repro): every matrix path from `MATRIX_ROOT` / `KEEP_ROOT`, `coupling=coupling()` passed to each `run_matrix`, smoke and repro on `full` seed 0; `MEMO = memo(...)`, `orc_memo=` and `require_orc_memo=` removed (no C3 cell computes ORC); the repro folder `EVAL / "repro_orcdir"`;
  - cell 16 (C4): a first line that raises "C4 dropped (option A, 2026-11-26)";
  - cell 18 (C5a): the `run_orc_job` call removed (not only marked);
  - cell 20 (C5c): base and shuffled runs both computed in the session (as `gate_shuffle_tgn` requires), base with `coupling=coupling()`, shuffled with its own coupling computed on the shuffled stream; `shuf_memo = load_orc_memo(...)` and every `orc_memo` / `require_orc_memo` argument removed; only the `write_run` targets move to `MATRIX_ROOT`; order: inputs, both couplings, base run, drop the base coupling, then the TGN arrays, with VmRSS printed after each step;
  - cell 22 (C6): `MATRIX_ROOT` everywhere including the shell line `--tables` (`{MATRIX_ROOT}`) with `--out {DRIVE}/eval_results_orcdir`, `"orc"` dropped from the spec, `BANNER` reworded for the closed-form directed ORC, and a mask `citing_uncited` (keep = `in_deg_src == 0` from the screen frame it already builds) written and passed to the masked views.
- Test: `tests/test_eval_notebook.py`

- [ ] **Step 1: Write the failing tests**: `REQUIRED` gains `"reference_overlap("`, `"real_orcdir"`, `"coupling=coupling()"`, `"citing_uncited"`; a new test asserts that no code cell other than cell 4 names the old roots in any form (a regex over `matrix" / "real"`, `matrix/real` and `no_temporal_keep_temp` not followed by `_orcdir`, shell lines included), that every `run_matrix(` and `run_pipeline(` call in cells 13, 14 and 20 carries `coupling=`, and that no code cell calls `run_orc_job(` outside the C1/C2 cells.
- [ ] **Step 2: Run** `tests/test_eval_notebook.py` — Expected: FAIL.
- [ ] **Step 3: Edit** with a format-preserving JSON patch script (as `patch_nb50_syspath.py` of 2026-11-25).
- [ ] **Step 4: Run** `tests/test_eval_notebook.py` — Expected: PASS.
- [ ] **Step 5: Commit** `feat(eval): 50_eval reruns the matrix with the closed-form directed ORC`.

### Task 5: Records

**Files:** `src/citecurv/pipeline.py` module docstring (lines 7-13) and the `precompute_matrix_orc` docstring; `scripts/rehearsal.py` (its `kept["full"]` diagnostics point at `plan_full`, or a note that the rehearsal report is the plan-formulation record); `docs/score_components.md` (s_geom, s_curvdyn for the `orc_dir` formulation; a decision entry with Task 0's numbers; the `ctl_coupling_share` note), `docs/superpowers/plans/2026-11-25-s43-ablation-matrix.md` (amendment at the top: matrix root `real_orcdir`, C4 dropped, C5c and C6 read `real_orcdir`, the C3 results under `real` kept as the plan-formulation record), `PROGRESS.md` (G7 row: Maor's 2026-11-26 choices; a session line).

- [ ] **Step 1: Write** the edits, with no number that was not read from a file or a run.
- [ ] **Step 2: Run** the laptop suite — Expected: the 24 known failures only.
- [ ] **Step 3: Commit** `docs: option A (directed ORC, closed form) recorded`.

## After this plan (each on Maor's word)

1. Push. 2. C0 on Colab (the suite there, where GraphRicciCurvature runs). 3. C3 rerun into `real_orcdir`: 30 runs, about 5.9 h at the 2026-11-25 pace of 11.7 min per run, self-releasing. 4. C5 and C6 against `real_orcdir`. 5. Library ORC for premise rows, only if Task 0's share is not small.

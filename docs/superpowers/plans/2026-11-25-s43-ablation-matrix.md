# S4.3 Ablation Matrix (full run) Implementation Plan

> **Amendment 2026-11-26 (option A, Maor's choice; plan `docs/superpowers/plans/2026-11-26-orc-dir-closed-form.md`).** The C3 run of 2026-11-25 under `full/eval/matrix/real` (and `no_temporal_keep_temp`) is kept as the record of the plan formulation. From now on the cells `full`, `no_temporal` and `full_learned` use the closed-form directed ORC (curvature mode "closed", component variant "orc_dir"), C3 reruns into `full/eval/matrix/real_orcdir` (`no_temporal_keep_temp_orcdir`), C4 is dropped (no ORC job, no `ollivier_only` cell), C5a runs no shuffled ORC job, and C5c and C6 read `real_orcdir`; C6 adds the exact-only view `citing_uncited`. The old `full` stays runnable as the record cell `plan_full`.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the five ablation cells, the learned-combiner cell and a split-off `no_temporal` cell for seeds 0 to 4 on the full run, plus the timestamp-shuffle check with the TGN re-scored on the shuffled stream, and publish tables with paper-level uncertainty, the ledger's label views and the G7 counting controls.

**Architecture:** No new training. Every cell is a `citecurv.pipeline.run_pipeline` call that reuses the five trained TGN checkpoints' scores (Drive `full/tgn/infer/TGN_full_seed<s>/`). Code first (laptop, test-first): a curvature stage that never builds an EventStore behind the caller's back, an ORC escalation that records every row it attempts and survives a failing row, a resumable ORC precompute run as a child process with a heartbeat, an eligibility rule for `ollivier_only`, a shuffled DyGLib dataset writer and merge, a shuffle gate with a drift rule, and a reporting module. Then `notebooks/50_eval.ipynb` is pointed at the full-run layout and run in Colab in measured steps; every step starts on Maor's word after the previous step's report.

**Tech Stack:** Python 3.12 (Colab runtime version 2026.07: GraphRicciCurvature 0.6.1, POT 0.9.6, networkit 11.2.1, SciPy 1.13.1 installed there on 2026-11-03 and 11-11), pandas, numpy, networkx, DyGLib (vendored, P1 to P11); torch only for the shuffled re-scoring, on the GPU image that made seed 0's stored scores (Python 3.13.16, torch 2.14.1+cu130, 2026-11-24).

**Spec:** `IMPLEMENTATION_PLAN.md` S4.3 (and S3.4, S4.6, the curvature-cache contract); `docs/score_components.md` section 8; module docstrings of `src/citecurv/pipeline.py`, `src/citecurv/eval/ablation.py`, `src/citecurv/tgn/infer.py`; ledger rows G3, G7, G9 to G13 in `PROGRESS.md`. Review reports of rounds 1 and 2: outside the repo (session notes folder, `plan-review/`).

## Global Constraints

- Cells: `full`, `no_curvature`, `forman_only`, `ollivier_only`, `no_temporal` (`eval.ablation.ABLATION_VARIANTS`), `full_learned` (`EXTRA_CELLS`), and `no_temporal` again with `pipeline.no_temporal_drop_s_temp: false` in its own output folder, group `no_temporal_keep_temp` (splits the `no_temporal` delta into encoder and `s_temp`; D11 unchanged). Seeds `seed_list: [0, 1, 2, 3, 4]`.
- `no_temporal` = the implemented D11 reading (Maor, 2026-11-25, "Alright" to: run the free way now, decide on the DyGLib memory-off run after the results): static SPECTER2 vectors instead of TGN outputs, `s_temp` off.
- `ollivier_only` draws its matched random sample only among rows whose cited work has a known reference list at t, the pool the screen can flag under G13 (Maor, 2026-11-25, "Yes, skip them"). Sample size per split = the screen's escalated count in that split.
- Curvature settings (Maor, 2026-11-25, "Alright" to keeping the defaults, PROVISIONAL): screen percentiles 0.01 / 0.99, `screen_inclusive_ties: false`, ORC radius 1, `orc_max_events: null`, `shortest_path: pairwise`; cold-start k = 3 (S2.4 open). The table banner names these, plus G11 (`total_cold`, departs from L19, advisors not yet told) and G3 (label rule not yet signed off).
- The shuffle check re-scores the shuffled stream with seed 0's TGN checkpoint (Maor, 2026-11-25, "Alright"), never the `static_cosine` fallback.
- ORC values come from GraphRicciCurvature (G7: keep the plan formulation). The closed form of `curvature/orc.py` is a fallback only on Maor's word after C1.
- Split: cutoff from Drive `full/tgn/data/split.yaml` (2023-03-30); the pipeline's own split (`val_frac` 0.10) must reproduce its event counts per part exactly (training part ends 2022-09-19, where DyGLib's training part ends), or the run stops.
- Context rows: the cited works' reference lists, `reference_events` over `full/cited_joint.parquet` read with `columns=["work_id", "pub_date", "ref_ids"]`, in every curvature stage and ORC subgraph (G13 on). Their `context_hash` is recorded wherever an ORC value is stored or read.
- Every matrix cell runs with `require_orc_memo=True`: no EventStore is ever built inside a cell.
- Colab runs, uploads to Drive, pushes and paid compute only on Maor's word in chat. Every Colab job has a `MAX_HOURS` watchdog in the notebook kernel, calls `drive.flush_and_unmount()` and then releases the runtime. Heavy ORC work runs in a child process the kernel watches through a heartbeat file.
- Every run's `TAGS`: repo commit hash, TGN infer folder, checkpoint `best_epoch`, ORC memo file sha256 and `context_hash`, host fingerprint (CPU model, vCPU count, numpy SIMD and BLAS core), and for GPU steps the python, torch and CUDA versions and GPU name.
- Public repository: no account names, e-mail addresses or family details in code, notebooks or commit messages.
- Laptop test runs: every test step first reads DeskFlow's outbox (`~/.claude/main-session/outbox-deskflow.md`) and runs only when no START is open. Known laptop failures: the 24 of 2026-11-23 (missing scipy, sklearn, xgboost), among them `test_run_matrix_on_fixture_writes_runs_and_tables`, `test_temporal_stat_config_reaches_the_components`, `test_calibrated_and_learned_weights_when_val_has_positives`; any other failure is new.

## Review Focus

1. **ORC cost and fragility at full size.** GraphRicciCurvature forks a worker pool on every call (`orc_directed` docstring), fork cost grows with the parent's memory, and one failing row would stop a resumed run on the same row. Expected: the screen set first, the per-seed random sets last; batched calls; an optional lean ORC worker; per-row errors recorded and skipped; a stalled child killed by the kernel. Pinned by `test_batched_escalate_equals_per_event` and `test_escalate_records_a_failing_row_and_moves_on` (Task 1), `test_orc_job_writes_a_heartbeat_per_batch` (Task 3), and by C1's stratified rate measurement in the precompute's memory state.
2. **A memo miss or a stale memo inside a cell.** Expected: the cell raises before building anything; a memo from other settings, data or context rows is refused. Pinned by `test_strict_memo_raises_before_building_a_store` (Task 2) and `test_memo_loader_refuses_another_context` (Task 3).
3. **Stale ORC values on the shuffled stream.** A test event whose shuffled t equals its original t keeps its memo key while the graph at t changed. Expected: only keys with t <= cutoff are reused. Pinned by `test_known_values_after_cutoff_are_recomputed` (Task 3).
4. **A session dies mid-write or mid-cell.** Expected: no truncated checkpoint, no computed ORC value lost, no cell counted as done with files missing or from another configuration. Pinned by `test_checkpoint_survives_a_truncated_temp_file`, `test_rerun_with_fewer_seeds_keeps_computed_rows` (Task 3) and `test_resume_checks_files_and_configuration` (Task 2).
5. **The shuffled DyGLib files drift from the originals in node order or features.** Expected: node files copied byte for byte, ties in `(t, event_id)` order. Pinned by `test_moved_rows_take_new_times_and_keep_the_node_files` (Task 4) on distinct node vectors where a re-run of `to_jodie` gives a different node file.

## Design notes (adjudications of rounds 1 and 2)

- The S3.4 pilot step is dropped: the pilot's newest TGN inference has no embeddings file, its D4 cell is one concept and one year, and its run does not exercise the full-run path. The S3.4 gates run on C3's `full_0` in the session that computes it; reproducibility on a second full-size run of `forman_only` seed 0 in its own folder (group `repro`).
- "ORC reaches fewer than 100 test positives" and "most test positives are cold-start" are reported in C1, not stop rules: every C step starts on Maor's word after the previous step's report.
- Shuffle gate: C5 reruns `full` seed 0 on the original stream in the same CPU session as the shuffled run, so the fitted objects compare on one host; the decision is the existing `gate_shuffle` rule plus a drift rule; matched-row and per-component readings with a paper-clustered interval are reported in C6, and S4.8 says "degrades" only when that interval excludes 0.
- Bootstrap: src-, dst- and batch-clustered intervals per comparison; the widest is the headline. A two-way bootstrap is not used (cost at 1.2 million rows x 1,000 draws).
- Control directions are fixed by a sign rule on validation rows (validation AUROC > 0.5), never by test rows.
- Cells that read ORC from the memo report capped / lossy counts as `None`, never 0; real counts come from the precompute's per-row columns in C6.

---

## Part 1: code (laptop, test-first)

Run tests with `.venv/Scripts/python.exe -m pytest <files> -q` from the repo root, after the DeskFlow outbox check.

### Task 1: ORC escalation that records every attempt, survives a failing row, and batches

**Files:**
- Modify: `src/citecurv/curvature/orc.py` (`escalate`, new `OrcWorker`)
- Test: `tests/test_orc.py`

**Interfaces:**
- Produces:
  - `escalate(events, curv, store=None, radius=1, edge_cap=5000, *, prune=False, orc_fn=None, max_events=None, stage="escalate", rows=None, batch=1, worker=None, progress=None, **orc_knobs)`. `rows`: explicit row positions, processed in the given order (self-loops among them skipped and counted; `max_events` then slices `rows`); default = escalated, not a self-loop, `orc` NaN and not `orc_attempted` (when that column exists), in frame order. Every attempted row (value, NaN or error) gets `radius` and `computed_at_stage` and the per-row columns `orc_subgraph_edges`, `orc_capped`, `orc_lossy`, `orc_src_in_deg` (in-degree of the citing work at t; 0 means radius 1 is exact), `orc_attempted` (True) and `orc_error` (str, empty when fine), added to the returned frame; stats gain `rows` (those columns keyed by `event_id`) and `n_errors`. `progress(event_ids)` is called before every `orc_fn` call with the ids in flight. With `batch > 1`: up to `batch` pruned subgraphs, nodes relabelled `(k, node)` keeping each subgraph's node order, joined into one DiGraph, one `orc_fn` call for all their scored edges; `ValueError` unless `orc_knobs.get("shortest_path") == "pairwise"` (or the worker's knobs say so). An exception raised by `orc_fn` itself is retried row by row; a row that still raises gets NaN, `orc_attempted` True and the error text.
  - `class OrcWorker`: `multiprocessing.get_context("spawn").Process(daemon=False)` with a pipe, never a `Pool` (the library opens its own fork pool inside it, which a daemonic process may not); started before any store is built; `OrcWorker(orc_fn_path: str, **orc_knobs)`, `.knobs`, `.compute(nodes: list, edges: list[tuple], scored: list[tuple]) -> dict` (node order carried, so values equal the in-process path), `.close()`. An exception inside `orc_fn` comes back as text and is a row error; a dead worker (EOF, broken pipe) raises `OrcWorkerDied`, which `escalate` does not turn into row errors: the batch's rows stay pending and the caller restarts the worker once. `escalate` raises when `worker.knobs` without `proc` differ from its own knobs.

- [ ] **Step 1: Write the failing tests**:
  - `test_escalate_records_per_row_attempts` (fake ORC; one record per computed row; `orc_capped` true on a row forced over `edge_cap`).
  - `test_escalate_skips_attempted_rows_by_default` (a row attempted with a NaN result is not recomputed on the next call).
  - `test_escalate_records_a_failing_row_and_moves_on` (a fake that raises on one key: that row NaN with an error text, every other row computed, `n_errors == 1`).
  - `test_batched_escalate_equals_per_event`: fake ORC = for each requested edge, (number of edges in the weakly connected component of u) + (largest edge `t` in that component) / 1e6; `batch=4` equals `batch=1` exactly and makes `ceil(n / 4)` calls; every edge of row k's component has t <= t_k; the fixture holds two batch members citing the same v with u1 cited between t1 and t2.
  - `test_batched_escalate_needs_pairwise_paths`.
  - `test_orc_worker_returns_what_the_function_returns` (a picklable module-level fake, spawn context, laptop-safe).
  - `test_orc_worker_can_host_a_function_that_starts_a_child` (the module-level fake itself starts and joins a spawn child; fails under a daemonic worker, also on Windows).
  - `test_dead_worker_leaves_rows_pending` (a fake that kills its own process: `OrcWorkerDied`, no row marked attempted).
  - `test_explicit_rows_keep_their_order_and_skip_self_loops`.
  - `test_batched_escalate_equals_per_event_with_the_library` and `test_orc_worker_equals_in_process_with_the_library` (`pytest.importorskip("GraphRicciCurvature")`, run in C0): the overlap fixture plus 30 fixture rows, `batch=10` vs `batch=1`, worker vs in-process, equal values.
- [ ] **Step 2: Run and see them fail** (the library test skips on the laptop).
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `tests/test_orc.py`: PASS.
- [ ] **Step 5: Commit** `feat(orc): per-row attempts and errors, a batched path and a lean ORC worker`.

### Task 2: Curvature stage without hidden stores; strict and checked resume

**Files:**
- Modify: `src/citecurv/pipeline.py` (`curvature_stage`, `run_pipeline`, `orc_memo_from_cache`)
- Modify: `src/citecurv/eval/ablation.py` (`run_matrix`)
- Test: `tests/test_pipeline_lazy_store.py` (new)

**Interfaces:**
- Consumes: Task 1's `escalate(..., rows=...)`.
- Produces:
  - `class OrcMemoMiss(ValueError)` in `pipeline.py`.
  - `curvature_stage(..., require_orc_memo: bool = False)`: after the memo fill, misses = escalated non-self-loop rows without a memo key. No misses: no store, no `escalate` call, stats `n_orc_computed: 0` and `orc_s_per_computed`, `mean_subgraph_edges`, `n_capped`, `n_capped_lossy` all `None`. Misses and `require_orc_memo`: raise `OrcMemoMiss` with the count and the first 5 keys, before any store exists. Otherwise `escalate(..., rows=<miss positions>)`. A memo key whose value is NaN is a hit (an attempted row; the cell writes NaN). It returns `curv[CURVATURE_COLUMNS]` (Task 3 adds `ref_known` there), so computing and memo cells give the same columns; the per-row ORC columns live only in `escalate`'s output and the precompute checkpoint.
  - `run_identity(data: dict, variant: str, seed: int, cfg, tags) -> dict`: `config_hash`, `data_hash` (over the per-event `labels_df` exactly as `run_pipeline` builds it, after `resolve_cutoff`) and `commit = (tags or {}).get("commit")`; used by both `run_pipeline` and `run_matrix`. `run_matrix(..., identity_data=None)`: when given (the inputs without TGN outputs), the identity of a cell is computed from it before `data(seed)` loads anything heavy.
  - `run_pipeline(..., require_orc_memo=False)`, `run_matrix(..., require_orc_memo=False)`; not part of `config_hash`.
  - `run_matrix` prebuilds a per-seed `EventStore` only when `orc_memo is None and not require_orc_memo`, from events plus `pipeline_kwargs["context"]` when given.
  - `run_pipeline` writes `escalated.npy` (int64 ids escalated in the cell; empty for `no_curvature`) and writes `metrics.json` last, holding `files: {name: {"bytes", "sha256"}}` of the other six files. `run_matrix(resume=True)` skips a cell only when all seven files exist, the six match `files`, and the stored `config_hash`, `data_hash` and commit equal `run_identity`'s. A missing or mismatching file means rerun the cell; a config, data or commit mismatch raises.
  - `orc_memo_from_cache(curvature, radius, include_attempted: bool = False)`: with True, rows with `orc_attempted` true and NaN `orc` enter as NaN; a frame without the column counts as all False.

- [ ] **Step 1: Write the failing tests** on `load_fixture()` plus one appended escalated self-loop row, `rcfg = resolve_config({"pipeline": {"screen_percentile_low": 0.1, "screen_percentile_high": 0.9}}, variant=...)`, `split = split_events(events, fx["meta"]["t_min"] - 1).to_numpy(dtype=object)`, the closed-form fake ORC of `tests/test_pipeline_integration.py` with a call counter; `Boom` (raising `__init__`) patched as `citecurv.curvature.timeslice.EventStore` and `citecurv.curvature.orc.EventStore`:
  - `test_curvature_stage_builds_no_store_when_the_memo_covers_every_row` (second call equal to the first; `n_orc_computed == 0`; hits == escalated minus self-loops; the four stats `None`).
  - `test_strict_memo_raises_before_building_a_store` (`OrcMemoMiss`, not `Boom`'s error; names 1 key).
  - `test_nan_memo_value_counts_as_a_hit` and `test_one_nan_hit_and_one_miss_compute_only_the_miss` (fake called once).
  - `test_curvature_stage_builds_the_store_once_when_one_row_is_missing` (regression guard; passes today).
  - `test_run_matrix_prebuilds_no_store_when_given_a_memo` (cells `["full", "forman_only"]`; equal `scores.parquet` sha256; equal `metrics.json` after removing `timings`, `curvature` and `files`; second run's `n_orc_computed == 0`).
  - `test_run_matrix_store_holds_the_context_rows`.
  - `test_resume_checks_files_and_configuration` (a deleted `thresholds.json` and a truncated `weights.json` rerun that cell; a changed commit raises; `tests/test_ablation.py::test_run_matrix_resume_skips_finished_cells` still passes).
  - `test_memo_cell_and_computing_cell_have_the_same_columns` (and `test_orc_memo_reuse_leaves_output_unchanged` stays green).
- [ ] **Step 2: Run them and see the new ones fail** (the regression guard passes).
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the new file plus `tests/test_pipeline_integration.py tests/test_ablation.py tests/test_orc.py`: PASS apart from the known failures.
- [ ] **Step 5: Commit** `fix(pipeline): no hidden EventStore; strict ORC memo; checked resume`.

### Task 3: Shared selection, `ollivier_only` eligibility, the ORC precompute and its job runner

**Files:**
- Create: `tests/fixtures/golden_escalation.json` (from the unmodified code, committed first)
- Create: `src/citecurv/eval/orc_job.py` (child-process entry point)
- Modify: `src/citecurv/curvature/screen.py` (`forman_screen`: column `ref_known`, appended to `CURVATURE_COLUMNS`)
- Modify: `src/citecurv/scoring/components.py` (`matched_random_escalation(..., eligible=None)`)
- Modify: `src/citecurv/pipeline.py` (`screen_stage`, `select_escalation`, `precompute_matrix_orc`, `load_orc_memo`, `context_hash`)
- Modify: `docs/score_components.md` section 8 (`ollivier_only`: "reads only whether the cited work's reference list is known", never a Forman value) and the curvature-cache contract text in `IMPLEMENTATION_PLAN.md` (new column)
- Test: `tests/test_pipeline_orc_precompute.py` (new), `tests/test_screen.py`, `tests/test_components.py`

**Interfaces:**
- Consumes: Tasks 1 and 2.
- Produces:
  - `forman_screen` adds `ref_known` (bool) = the out-degree of the cited work at t after the self-loop adjustment is > 0; with G13 on, `ref_known == ~isnan(forman)`.
  - `matched_random_escalation(curvature, *, seed=0, strata=None, eligible=None)`: each stratum draws only among its eligible rows; size per stratum = the screen's escalated count there (capped at the eligible count; the cap recorded).
  - `screen_stage(events, split, rcfg, *, context=None) -> tuple[pd.DataFrame, dict, dict]` (canonical events in, order asserted; values unchanged from today's first half of `curvature_stage`).
  - `select_escalation(curv, split, mode, seed, cap, *, unknown_without_refs) -> pd.DataFrame` ("random+orc": `eligible = curv["ref_known"]` when `unknown_without_refs`, else all rows).
  - `context_hash(context) -> str` (sha256 over the canonical context rows, same form as `data_hash`).
  - `orc_settings_hash(rcfg) -> str`: hash of the resolved `curvature` block without `orc.proc`. Sidecar and loader use `data_hash(events)` without labels.
  - `build_context(works, refs, events) -> tuple[pd.DataFrame, dict]`: `reference_events(refs, exclude_src=set(works.work_id), start_event_id=int(events.event_id.max()) + 1)`; the one place the notebook and the job build context rows.
  - `precompute_matrix_orc(events, cutoff, rcfg, seeds, *, out_path, context=None, known=None, known_max_t=None, orc_fn=None, chunk=5000, batch=1, worker=None, order_seed=0, max_chunks=None, heartbeat=None, freeze_gc=False, retry_errors=False) -> dict`. It builds the EventStore itself, from events plus context (never taken from outside), and with `freeze_gc` calls `gc.freeze()` after the build and `gc.unfreeze()` at the end (only the job passes True). Each chunk's rows go to `escalate` explicitly through `rows=`. `heartbeat` (a local-disk path, never Drive) is rewritten at every setup phase and, through `escalate(progress=...)`, before every ORC call, holding `{"phase", "time", "in_flight": [event ids]}`. On start, a heartbeat left by an abnormal exit (in-flight ids and no clean end) marks those rows attempted with `orc_error="killed: timeout"`; `retry_errors=True` recomputes error rows in a later pass. The 0.1 % error stop is checked before a chunk is written. A `known` checkpoint's sidecar must match on settings hash, `context_hash` and `radius` (its `data_hash` may differ: the shuffled stream). Canonicalizes `events` and builds the split itself (`_canonical_events`, `split_events(ev, cutoff, val_frac=rcfg["split"]["val_frac"])`); raises when `orc_max_events` is set. Targets = `select_escalation(..., "screen+orc", 0, None)` united with `select_escalation(..., "random+orc", s, None)` for every `s` in `seeds`, processed in a seeded random order (`order_seed`). `known` (a checkpoint frame or memo of the same stream) fills target keys with t <= `known_max_t` (required when `known` is given), marked attempted. The checkpoint holds every row ever computed (a superset of the current targets), written to `<out_path>.tmp` then `os.replace`; the sidecar `<out_path>.json` holds the curvature settings hash (ORC knobs without `proc`), `data_hash`, `context_hash`, `cutoff`, `radius`, the union of seeds and `n_targets`. A rerun refuses any mismatch except a different seed list; fewer seeds compute nothing and rewrite no row. It stops with an error when `n_errors` exceeds 0.1 % of the targets. Returns `{"n_targets", "n_known", "n_done", "n_pending", "n_attempted_nan", "n_errors", "n_chunks_run", "seconds", "subgraph_s_per_event", "orc_s_per_event", "split_counts", "n_screen_flagged", "n_eligible"}`.
  - `load_orc_memo(out_path, *, data_hash, context_hash, settings_hash, radius) -> dict`: reads the checkpoint through its sidecar, refuses a mismatch, returns `orc_memo_from_cache(frame, radius, include_attempted=True)`.
  - `python -m citecurv.eval.orc_job <job.json>` (top-level code under `if __name__ == "__main__"`, since a spawn child re-imports the module): `job.json` = `{"events", "works", "cited_joint"` (paths; `cited_joint` read with `columns=["work_id", "pub_date", "ref_ids"]`), `"shuffle": null | {"cutoff", "seed"}` (the job applies `shuffle_test_timestamps` itself; no shuffled file is written), `"config"` (path), `"cutoff"`, `"seeds"`, `"out_path"`, `"known": null | path`, `"known_max_t"`, `"batch"`, `"chunk"`, `"worker": bool`, `"order_seed"`, `"max_chunks"`, `"heartbeat"`, `"result"` (path)`}`. It builds the context with `build_context`, runs `precompute_matrix_orc(..., freeze_gc=True)`, closes its worker in `try/finally`, writes the result JSON and a clean end into the heartbeat.

- [ ] **Step 1: Record the golden file from the unmodified code**: for modes "screen+orc", "random+orc", "screen", seeds {0, 1}, `orc_max_events` {None, 7}, on the fixture with `unknown_without_refs` False: sorted escalated ids and `frame_sha256(curv[V1])`, V1 = today's nine curvature columns written as a literal list in the test. Commit `test: golden escalation sets before the curvature-stage split`.
- [ ] **Step 2: Write the failing tests**:
  - `test_selection_matches_the_golden_file`.
  - `test_ref_known_matches_the_unknown_rule` (G13 on, including a self-loop row with no other reference).
  - `test_ollivier_only_never_escalates_a_row_without_known_references` (per-split sizes equal the screen's counts).
  - `test_ollivier_only_ignores_forman_values` (Forman replaced by F' = 3F - 7, NaN kept: escalated ids unchanged).
  - `test_precompute_covers_every_cell` (`seeds=[0, 1]`; through `load_orc_memo`, `curvature_stage` for `full` seed 0 and `ollivier_only` seeds 0 and 1 with `require_orc_memo=True` and a raising `orc_fn`).
  - `test_precompute_accepts_unsorted_shuffled_events` (split counts equal `run_pipeline`'s).
  - `test_precompute_resumes_without_recomputing` and `test_precompute_with_a_nan_row_finishes` (a fake that returns NaN for one row: `n_pending == 0`, `n_attempted_nan == 1`, calls == targets minus self-loops).
  - `test_precompute_rerun_with_more_seeds_computes_only_new_targets` and `test_rerun_with_fewer_seeds_keeps_computed_rows`.
  - `test_known_values_after_cutoff_are_recomputed` (a shuffled stream where one escalated test event keeps its t while a neighbour moves) and `test_known_nan_value_reaches_the_strict_cell`.
  - `test_checkpoint_survives_a_truncated_temp_file`, `test_precompute_refuses_other_settings`, `test_memo_loader_refuses_another_context`, `test_precompute_refuses_an_orc_cap`.
  - `test_orc_job_writes_a_heartbeat_per_batch` (the job's run function called in-process with a fake ORC: setup phases and every batch written, clean end at the close).
  - `test_rows_in_flight_at_a_kill_are_marked_on_restart` (a heartbeat with in-flight ids and no clean end: those rows become attempted with `killed: timeout`; `retry_errors=True` recomputes them).
  - `test_precompute_error_row_reaches_the_strict_cell` (a fake that raises on one key; a strict `full` seed 0 cell then writes NaN for that row).
  - `test_known_checkpoint_from_other_settings_is_refused`.
- [ ] **Step 3: Run and see them fail** (the golden test passes before Step 4 and must keep passing).
- [ ] **Step 4: Implement**, and update the two documents.
- [ ] **Step 5: Run** the new file, `tests/test_screen.py tests/test_components.py tests/test_orc.py tests/test_pipeline_lazy_store.py tests/test_pipeline_integration.py tests/test_ablation.py`: PASS apart from the known failures.
- [ ] **Step 6: Commit** `feat(pipeline): resumable ORC precompute with a job runner; ollivier_only draws among rows with known references`.

### Task 4: Shuffled DyGLib dataset

**Files:**
- Modify: `src/citecurv/tgn/convert.py` (add `shuffle_dyglib_dataset` after `load_idmap`)
- Test: `tests/test_tgn_shuffle.py` (new)

**Interfaces:**
- Produces: `shuffle_dyglib_dataset(data_dir, name, shuffled_t, out_dir, out_name) -> dict`. `shuffled_t`: frame `event_id, t` covering exactly the dataset's events. Writes `ml_<out_name>.*`: node feature file and idmap `work_id` copied unchanged; events re-sorted by `(t, event_id)` with the new `t`; `u`, `i` kept per event; `idx` renumbered 1..M; edge feature rows re-ordered with their events; idmap `event_id` in the new order; meta `n_events`, `n_moved`, `t_min`, `t_max`, `source_name`. `ValueError` when the event id sets differ.

- [ ] **Step 1: Write the failing tests** on `to_jodie(events10, (ids, vecs), tmp_path, "toy")` with distinct node vectors, same-day ties and a work that first appears in an event the shuffle moves earlier, plus a variant with `reduce_to=2, fit_ids=pca_fit_ids(events10, cutoff)`. Each test first asserts that a re-run of `to_jodie` on the shuffled events gives a different node file (so the test can fail):
  - `test_identity_shuffle_reproduces_the_files`.
  - `test_moved_rows_take_new_times_and_keep_the_node_files` (`ts` non-decreasing, ties in `event_id` order; each event's `ts` equals its `shuffled_t`; node file byte-identical to the original; each `work_id`'s node row equal to the original's; `n_moved` right).
  - `test_rejects_a_different_event_set`.
- [ ] **Step 2: Run and see them fail.**
- [ ] **Step 3: Implement** with pandas and numpy only.
- [ ] **Step 4: Run** `tests/test_tgn_shuffle.py tests/test_convert.py`: PASS.
- [ ] **Step 5: Commit** `feat(tgn): write a DyGLib dataset with shuffled test-period times`.

### Task 5: Merge of shuffled TGN outputs

**Files:**
- Modify: `src/citecurv/tgn/infer.py` (add `merge_shuffled_tgn` after `load_event_scores`)
- Test: `tests/test_tgn_merge.py` (new)

**Interfaces:**
- Produces: `merge_shuffled_tgn(orig_scores, orig_emb, shuf_scores, shuf_emb, events, cutoff) -> tuple[pd.DataFrame, dict, dict]`: events with original `t <= cutoff` keep the original outputs, the others take the shuffled outputs, aligned by `event_id`, in the original order. Consumes `orig_emb` (arrays overwritten in place; the docstring says so). Report `{"n_pre", "n_test", "pre_max_abs_diff_p_link", "pre_max_abs_diff_neg_p_link"}`. `ValueError` when the event id sets differ.

- [ ] **Step 1: Write the failing tests**: `test_merge_takes_pre_cutoff_rows_from_the_original` (shuffled outputs in another row order, pre-cutoff `p_link` shifted by 1e-7: merged pre-cutoff rows exact originals, test rows the shuffled values by id, drift about 1e-7, embeddings by the same rule) and `test_merge_rejects_different_event_sets`.
- [ ] **Step 2: Run and see them fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `tests/test_tgn_merge.py tests/test_tgn_infer.py`: PASS (torch tests skip).
- [ ] **Step 5: Commit** `feat(tgn): merge TGN outputs of a shuffled stream with the original`.

### Task 6: Shuffle gate with a drift rule; top-20 gate under G13

**Files:**
- Modify: `src/citecurv/pipeline.py` (`gate_shuffle_tgn`, `gate_top_breakdown`)
- Test: `tests/test_pipeline_integration.py` (new tests)

**Interfaces:**
- Produces:
  - `gate_shuffle_tgn(base: RunResult, shuffled: RunResult, merge_report: dict, *, drift_tol: float) -> dict`: PASS when `gate_shuffle(base, shuffled)` passes and both drift values of the merge report are <= `drift_tol`; its value adds the matched-row readings (test rows scored, not cold-start, in both runs, by `event_id`): total and per-component AUROC / AP on both sides. `drift_tol` has no default: C5 passes the measured value.
  - `gate_top_breakdown`: a NaN `z_s_geom` or `z_s_curvdyn` counts as complete when the row's `ref_known` is false in `result.curvature` (G13 unknown); `result.curvature is None` (no curvature) is handled explicitly.
- [ ] **Step 1: Write the failing tests** on the fixture: `test_gate_shuffle_tgn_follows_gate_shuffle_and_the_drift_rule` (both outcomes built by editing a copy of the shuffled result: test AUROC set to 0.5 gives PASS at drift 0 and FAIL above the tolerance; AUROC equal to the base's gives FAIL), `test_gate_shuffle_tgn_matched_rows_match_a_hand_computation`, `test_top_breakdown_accepts_g13_unknown_rows`, `test_top_breakdown_without_curvature`.
- [ ] **Step 2: Run and see them fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `tests/test_pipeline_integration.py`: PASS apart from the known failures.
- [ ] **Step 5: Commit** `feat(eval): shuffle gate with a drift rule; top-20 gate follows G13`.

### Task 7: Reporting: paper-level intervals, label views, controls, ORC coverage

**Files:**
- Create: `src/citecurv/eval/report_s43.py` (with `python -m citecurv.eval.report_s43 <runs> ...` regenerating every derived record)
- Modify: `src/citecurv/eval/ablation.py` (`write_run` accepts kind "report"; `tables()`: tables `ablation_intervals`, `ablation_views`, `orc_coverage`, `shuffle_components`, `controls`, a "seed varies" column and an "interval excludes 0" verdict next to the seed-noise verdict)
- Test: `tests/test_report_s43.py` (new)

**Interfaces:**
- Consumes: Task 2's run folders, labels with `via_src` / `via_dst`, `evaluate_baseline` / `baseline_views` / `baseline_record`.
- Produces (every record: kind "report", integer `seed` (-1 when pooled), `group`, `inputs: {path: sha256}` of every file it read; `tables()` marks a record whose inputs changed as stale):
  - `cluster_keys(events, labels, batches) -> dict[str, np.ndarray]`: per test row, keys "src", "dst" and "batch" (a labelled paper in a `mass_events` batch maps to the batch id `(publisher, ret_date)`, other rows to their citing paper).
  - `paired_paper_bootstrap(pairs: list[tuple[pd.DataFrame, pd.DataFrame]], labels, keys, *, metric="ap", n_draws=1000, seed=0) -> dict`: one cluster resample per draw applied to every seed's pair; the metric computed per seed on that seed's rows scored in both runs; deltas averaged over seeds; for each key in ("src", "dst", "batch") the 2.5 and 97.5 percentiles; headline = the widest; `{"delta", "per_key": {...}, "lo", "hi", "method", "n_clusters", "n_pos"}`. One pair = per-seed interval. Scope: scored rows, cold-start rows ranked by `total_cold` (G11), and each masked view.
  - Comparisons: `full` against every other cell; `no_temporal_keep_temp` against `full` (the encoder part); `no_temporal` against `no_temporal_keep_temp` (the `s_temp` part; order-dependent, said in the note).
  - `masked_view_records(run_dir, labels, masks) -> list[dict]`: masks `without_recordless` (G12: events citing an id without an OpenAlex record), `without_top_batches` at three levels (G9: the largest 1, the largest 5, every `mass_events` batch with `min_size=50`; an event is dropped when any labelled endpoint is a batch paper; batches and dropped positives listed in the record), `unseen_cited` (G9: the cited work appears in no train-part event; the note records the share of test rows whose citing work appears in train-part events).
  - `control_records(result_dir, controls, labels) -> list[dict]`: kind "control" rows for `coupling_share`, `cited_ref_count`, `citing_ref_count`, `dst_age`, `ref_known` (one bit); direction per control fixed by validation AUROC > 0.5, recorded; each reported twice: NaN rows ranked last, and on rows with a known reference list only, next to `full` on the same rows; also on cold-start test rows against `total_cold`.
  - `orc_coverage(run_dir, orc_rows, labels) -> dict`: escalated test rows, test positives with an ORC value (total, citing side, cited side), lossy share, share with `orc_src_in_deg > 0`, errors.
- [ ] **Step 1: Write the failing tests** on small synthetic frames: the widest interval is at least as wide as each one-way interval under positives concentrated in one cited paper and one batch; the pooled point equals the mean of per-seed deltas; rows of different seeds are never ranked together; masks drop exactly their rows; `unseen_cited` differs from the total on a fixture where every citing work is new; control direction comes from validation rows; `tables()` output for main, ablation and label_views is unchanged when report records are added, and the new tables render.
- [ ] **Step 2: Run and see them fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `tests/test_report_s43.py tests/test_ablation.py`: PASS apart from the known failures.
- [ ] **Step 5: Commit** `feat(eval): paper-level intervals, ledger views, counting controls and ORC coverage for S4.3`.

### Task 8: `50_eval` for the full run, runbook, ledger

**Files:**
- Modify: `notebooks/50_eval.ipynb`, `docs/colab_runbook.md`, `PROGRESS.md` (ledger and S4.3 note), `docs/score_components.md` (D11 row: decided reading, deferred alternative; section 8 `no_temporal` with `s_temp` off), `configs/pins.md` (curv row GraphRicciCurvature 0.6.1, G9)
- Test: `tests/test_eval_notebook.py` (new)

**Interfaces:**
- Consumes: Tasks 1 to 7.
- Produces, in the notebook:
  - Install cell: environment variables `OPENBLAS_CORETYPE=Haswell`, `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` set before numpy is imported; clone + `git checkout <hash>` + `pip install -e`; curvature pins of G9; an assertion of Python 3.12 and of the installed versions (`importlib.metadata`); the host fingerprint into `TAGS`.
  - Inputs: `DRIVE = .../citecurv/full`; `events.parquet`, `works.parquet`, `event_labels.parquet`; static vectors (`embeddings.npz` plus `embeddings_cited.npz` joined into one `(ids, vectors)` pair, parts freed); `CONTEXT, ctx_stats = build_context(works, refs, events)`; the split check against `split.yaml` (`n_train`, `n_val`, `n_test`, raise on a difference); `tgn_inputs(seed)` from `tgn/infer/TGN_full_seed<seed>/`.
  - One cell per C step, each with `MAX_HOURS`, a memory log by a background thread reading `VmRSS` from `/proc/self/status` every second (per-phase peaks; `clear_refs` optional inside a try), `drive.flush_and_unmount()` and release. ORC job cells run before the Inputs cell of their session (or after its frames are freed) and check `VmRSS` first; they start `python -m citecurv.eval.orc_job` with `subprocess.Popen(..., start_new_session=True)` and kill its whole process group (`os.killpg(pgid, SIGKILL)`, which takes the ORC worker and the library's pool too) when the heartbeat is older than its phase's limit: setup 2 x C1's measured setup time, ORC 5 x C1's slowest batch time. A killed job is restarted at most once before a report to Maor. Matrix cells pass `require_orc_memo=True`, `context=CONTEXT`, `resume=True`, `TAGS`, the memo from `load_orc_memo`; before loading a cell's inputs they log `VmRSS` and stop above the C3 smoke baseline plus 2 GB. `full_0`'s gates (cold-start, top-20, D4) run in the session that computes it. The shuffle cell defines every gate name it writes; pilot and full-run gates never share a record.
- [ ] **Step 1: Write the failing test** `test_eval_notebook_cells_compile_and_name_the_full_run`: each code cell parses with `ast.parse` after every `!` / `%` line is replaced by `pass` at its own indentation; the text names `TGN_full_seed`, `orc_job`, `require_orc_memo=True`, `load_orc_memo`, `gate_shuffle_tgn`, `split.yaml`, `MAX_HOURS`, `flush_and_unmount`, `OPENBLAS_CORETYPE`.
- [ ] **Step 2: Run and see it fail.**
- [ ] **Step 3: Edit the notebook and the documents**; the ledger rows quote Maor's words of 2026-11-25 ("Alright" three times, "Yes, skip them"); the runbook lists, before C5 runs, the expected shuffle readings: `s_temp` toward chance; the Forman part of `s_geom` close to unchanged by construction (context rows are not shuffled) except for cited works published in the test period; the ORC part partly toward chance; `s_embed` and `s_curvdyn` partly kept (timeless text features).
- [ ] **Step 4: Run** `tests/test_eval_notebook.py`, then the whole laptop suite: only the known failures.
- [ ] **Step 5: Commit** `feat(eval): 50_eval runs the full-run matrix; S4.3 decisions recorded`.
- [ ] **Step 6: Push** on Maor's word.

## Part 2: Colab runs (each step on Maor's word, after the previous step's report)

Results and logs go to Drive `full/eval/`; numbers into `PROGRESS.md` and the unit-burn log; CPU High-RAM about 0.26 units an hour, L4 about 1.94 (rates measured 2026-11-19 and 11-24).

- [ ] **C0. Suite on Colab** (CPU High-RAM, runtime 2026.07): `pytest -q -rA` from the editable install. Verify: 0 failed, or only `test_training_steps_match_upstream[1]` failing exactly as at c57b6df; `test_batched_escalate_equals_per_event_with_the_library` among the PASSED; the memory-log thread and the `clear_refs` probe tried and reported.
- [ ] **C1. Measurements in the precompute's memory state** (events, context, EventStore; no vectors, no TGN): split check; `n_screen_flagged` and `n_eligible` per split; G13 unknown count; test positives escalated by the screen; cold-start counts (`test_n_pos_all`, `test_n_pos_cold_start`, by label source); for D4, the field `gate_fairness` will use (`works.field` if present, else the first concept) and its largest cell's share of scored test rows (the test period lies in 2023, so year adds nothing). ORC rate on a seeded sample stratified by year x {low tail, high tail} for the screen set and by year from seed 0's random set, weighted by the real counts: in-process `batch` 1, 10 and 50 and the lean worker at `batch` 1 and 10, subgraph time and ORC time apart, p95 and max per stratum. One report comparison timed for C6's projection. Report to Maor: projected hours for the screen set and for each seed's random set; named fallbacks when a projection exceeds 20 hours (one random set for all seeds; the closed form for rows with `orc_src_in_deg == 0`, checked against the library; building the tier <= 2 subgraph directly, exact); the D4 rule if the largest cell exceeds half; which G9 batch level to headline.
- [ ] **C2. ORC precompute, screen set** (child process, resumable, may run overnight): `orc_job` with `seeds=[]` and C1's batch choice into `full/eval/orc_matrix.parquet`. Verify: `n_pending == 0`; `n_errors` below the 0.1 % stop; `n_attempted_nan` reported.
- [ ] **C3. Matrix without `ollivier_only`**: first `forman_only` seed 0 alone (integration smoke; per-phase memory), then `full`, `no_curvature`, `forman_only`, `no_temporal`, `full_learned` and the `no_temporal_keep_temp` group for seeds 0 to 4, one `run_matrix` call per seed; `full_0`'s gates in its session. Reproducibility: `forman_only` seed 0 again in a fresh session into a `repro` folder, `gate_reproducibility` on the two `scores.parquet` with both host fingerprints printed; on different hosts, rerun on a matching host before ruling on NFR4. Verify: 30 run folders passing the resume check; `n_orc_computed == 0` where present; gates recorded with group `full_run_gates`.
- [ ] **C4. Random sets and `ollivier_only`**, one seed at a time: `orc_job` with `seeds=[0..s]` (only the new seed's rows are computed; the child's memory returns to the system when it exits), then that seed's `ollivier_only` cell after the `VmRSS` check. Verify: 5 more run folders; `n_orc_computed == 0`.
- [ ] **C5. Shuffle check**: CPU: `shuffle_test_timestamps(events, CUTOFF, seed=777)`, `shuffle_dyglib_dataset(full/tgn/data, "full", ..., "full_shuf777")`, `orc_job` with `"shuffle": {"cutoff": CUTOFF, "seed": 777}`, `seeds=[]`, `known` = the C2 checkpoint, `known_max_t=CUTOFF`, into `full/eval/orc_shuffled777.parquet`. GPU, on the image that made seed 0's stored scores (versions asserted and recorded): `infer_tgn(TGN_full_seed0, ..., "full_shuf777", out_dir=full/tgn/infer/TGN_full_seed0_shuffled777, device="cuda", seed=0, embeddings_dtype="float32")` (about 20 minutes, about 1 unit); `drift_tol` = 1e-12 on the same stack; if that image is gone, re-score the original dataset in the same GPU session first (about 20 more minutes), measure the cross-stack maximum and set `drift_tol` to 10 times it, stated in the record. CPU, one session: `full` seed 0 on the original stream (base), then `merge_shuffled_tgn`, the shuffled `full` run with the shuffled memo, `gate_shuffle_tgn`. Verify: PASS expected; a FAIL or a drift above the tolerance stops the plan and goes to Maor; no tables until he rules.
- [ ] **C6. Reporting and tables** (CPU High-RAM, checkpointed per comparison): `python -m citecurv.eval.report_s43` over `full/eval/matrix/real` (intervals per seed and pooled for every listed comparison, masked views, controls against each `full_<s>`, ORC coverage per cell from `orc_matrix.parquet`, shuffle per-component readings with a paper-clustered interval), then `python -m citecurv.eval.ablation --tables full/eval/matrix/real --out full/eval_results --banner "<provisional settings, G11, G3>"`; copy `tables.md` and the CSV files to `results/s43_2026-11/`. Verify: 5 seeds per cell; every listed comparison has a paper-level interval; no stale record; records pushed on Maor's word.

## Afterwards (not in this plan)

The DyGLib memory-off `no_temporal` run (about 27 units of training), the L12 label-sensitivity run (S4.6; the validation calibration used Ref/Attr labels), S2.3 and S2.4 calibration, S4.4 baselines beyond the counting controls, S4.5 injection on the real stream, the advisors' update.

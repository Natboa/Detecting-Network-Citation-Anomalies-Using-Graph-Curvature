# Colab runbook

Operator's guide for Maor and Gilboa: which notebook to run when, on which runtime, with which pins,
what survives a dead session, and what to write down afterwards. Written 2026-10-31 for the
notebooks in this repository at the tag that contains this file. Anything that depends on an open
decision is marked PROVISIONAL. Plan references (S0.4, 9.1, ...) are to `IMPLEMENTATION_PLAN.md`.

The notebooks are thin drivers: all logic is in the `citecurv` package, installed from a release tag
of this repository. Every notebook has a `TAG` (or `REPO_TAG`) variable near the top; set it to the
tag you want to run, e.g. `v0.2.0`. The repository is public, so no GitHub token is needed. A tag is
created by whoever merges the work (`git tag v0.2.0 && git push origin v0.2.0`).

## 1. Order of work

| Step | Notebook | Plan tasks | Runtime type | Python | Status |
|---|---|---|---|---|---|
| 0 | (local, no notebook) | S0.4 key and `check_key()` | any | 3.12 | ready |
| 1 | `00_smoke.ipynb` | S0.5 (four go/no-go gates) | GPU (T4) | 3.12 required (opt-in 3.13 path) | ready |
| 2 | `10_data.ipynb` | S1.2 pilot pull and label join, S1.3 slice choice, S4.0 full pull | CPU (standard) | 3.12 recommended; the notebook accepts 3.12 or newer | ready |
| 3 | `20_embed.ipynb` | S1.6 pilot encoding, S4.0 embedding | GPU (T4) | 3.12 recommended; no version check in the notebook yet | ready |
| 3b | `25_cited_text.ipynb` | S1.7 follow-up test (2026-11-12): title and abstract of cited works outside the pulled set, their SPECTER2 vectors (`embeddings_cited.npz`, read by `40_tgn` through `EXTRA_EMB_PATH`), and the cold-start count | GPU (T4) | 3.12 required | pilot test, not part of the planned pipeline yet |
| 4 | `30_curvature.ipynb` | S2.2 Forman screen, S2.3 threshold/radius/budget grid and ORC escalation | CPU, high-RAM | 3.12 required (opt-in 3.13 path) | ready |
| 5 | `40_tgn.ipynb` | S1.7 pilot training, S4.2 main runs | GPU (T4 for the pilot) | 3.12 (warns otherwise) | ready |
| 6 | `50_eval.ipynb` | S3.4, S4.3 ablations, S4.4 baselines, S4.6 tables | GPU | 3.12 | not written yet |

S1.1-live (the first real pull) and S1.2 run from `10_data` or locally; see the checklist in
section 9. The directed ORC values of S1.5 come from Block A of `00_smoke` and from
`scripts/run_orc_validation.py` (results in `results/orc_validation_*.json`).

`10_data` writes `events.parquet` (the event table with `event_id` that `30`, `40` and `50` read)
through `citecurv.data.events.build_events`, and every notebook reads and writes the same
`RUN_DIR` (section 5). Do not improvise event ids by hand: every artifact is joined on `event_id`,
which is defined once, in code (sorted by t, src_id, dst_id).

## 2. Runtime type and Python version

Colab's default runtime has been Python 3.13 since 2026-08-25 (Google Colaboratory, 2026). The
curvature stack needs Python 3.12, because every GraphRicciCurvature release up to 0.6.1 declares
`scipy<=1.13.1` and SciPy 1.13.1 has no Python 3.13 wheel (`docs/orc_version_comparison.md`).

At the start of every session:

1. Runtime > Change runtime type.
2. Runtime version: **26.07** (or an older version; these run Python 3.12). The choice does not
   persist between sessions, so check it every time.
3. Hardware accelerator: CPU for `10` and `30`, T4 GPU for `00`, `20` and `40`. For `30` also turn on
   High-RAM if the plan allows it.
4. Save, then Runtime > Run all.

The first code cell of `00_smoke`, `20_embed`, `30_curvature` and `40_tgn` checks the Python version
(`00`, `20` and `30` stop with a message when it is not 3.12).
Both have an opt-in flag for Python 3.13 (`ALLOW_PY313` in `00`, `ALLOW_PY313_NODEPS` in `30`): it
installs GraphRicciCurvature with `--no-deps` next to the runtime's SciPy. That path gave values
identical to Python 3.12 in a Linux container, not on Colab. PROVISIONAL: Maor decides whether it
becomes the documented fallback; if you use it, write it in `PROGRESS.md`.

Compute cost (plan 9.1 estimates, not measured here): a T4 uses about 1.8 compute units per hour,
an A100 about 15. Use CPU runtimes wherever the table above says CPU.

## 3. Exact pins per notebook

Source: `configs/pins.md`, with the changes from `docs/literature_refresh_2026-10.md` and wave 1.
torch is never installed by a notebook: the GPU notebooks use Colab's preinstalled CUDA build of
torch 2.x.

| Notebook | Installs |
|---|---|
| `00_smoke` | `citecurv @ git+https://github.com/Natboa/Detecting-Network-Citation-Anomalies-Using-Graph-Curvature@<TAG>`; Block A: `GraphRicciCurvature==0.6.1 POT==0.9.6 networkit==11.2.1 "scipy<=1.13.1" networkx`; Block C: `adapters==1.3.0 transformers==4.57.6`; Block D: `pygod==1.1.0 torch_geometric==2.6.1`, then `--no-deps torch_scatter==2.1.2 torch_sparse==0.6.18 -f https://data.pyg.org/whl/torch-<runtime torch version>.html`. It also clones the tag for the vendored DyGLib. |
| `10_data` | citecurv from a clone of the tag, `pyalex==0.21`, and Colab's own pandas, pyarrow, requests, pyyaml |
| `20_embed` | citecurv from the tag, `adapters==1.3.0`, `transformers==4.57.6` (exactly; adapters 1.3.0 requires `transformers~=4.57.6`) |
| `30_curvature` | citecurv from the tag; on 3.12: `GraphRicciCurvature==0.5.3.2 "scipy<=1.13.1" POT==0.9.6 networkit==11.2.1 networkx`; on 3.13 with the flag: `POT==0.9.6 networkit==11.2.1 networkx python-louvain cython packaging`, then `--no-deps GraphRicciCurvature==0.5.3.2` |
| `40_tgn` | a clone of the tag installed with pip, `pyyaml`; DyGLib is the vendored copy in `third_party/DyGLib` (pure PyTorch, no PyG) |
| `50_eval` | planned: `pygod==1.1.0`, `torch_geometric` 2.6.x, `xgboost` (3.3 or newer on Python 3.12), `scikit-learn` 1.9; CurvGAD only in its own sealed environment |

PROVISIONAL (decision L5, Maor): `30_curvature` still installs GraphRicciCurvature 0.5.3.2 as in
`configs/pins.md`; `docs/orc_version_comparison.md` recommends 0.6.1 with the same POT and networkit
pins, and `00_smoke` already tests 0.6.1. Both give the same values on the unweighted graphs we use.

Side effects to expect: `scipy<=1.13.1` resolves to SciPy 1.13.1, which requires `numpy<2.3`, so pip
may downgrade NumPy; `transformers==4.57.6` requires `huggingface-hub<1.0`. pip prints dependency
warnings about other preinstalled Colab packages; they do not affect our stages.

## 4. Colab Secrets

| Secret | Used by | Notes |
|---|---|---|
| `OPENALEX_API_KEY` | `10_data`; `check_key()` | Left sidebar, key icon > Add new secret; name exactly `OPENALEX_API_KEY`; switch on "Notebook access" for the notebook. The notebook copies it into an environment variable and never prints it. |

Nothing else is needed: SPECTER2 is public on the Hugging Face hub (a warning about a missing
`HF_TOKEN` can be ignored), and no notebook calls Semantic Scholar. `OPENALEX_KEY_MODE` (`bearer`,
the default, or `param`) is an ordinary environment variable, not a secret; set it only if
`check_key()` says calls are not counted against the key (section 9).

Never paste the key into a cell, a commit, a screenshot or `PROGRESS.md`.

## 5. Drive layout and the few-large-files rule

Every notebook uses one layout: `<DRIVE_ROOT>/citecurv/<RUN>/`, with `DRIVE_ROOT = /content/drive/MyDrive`
and `RUN` = `pilot` (the S1.2 pilot slice) or `full` (the S4.0 slice). `RUN` is set in one cell of each
notebook (the parameters cell of `10`, the Drive cell of `20`, `30` and `40`) and must have the same value in
every notebook of a run; the notebooks call that folder `RUN_DIR`. Inside it, files carry the contract names
(IMPLEMENTATION_PLAN.md section 3, CONTRACTS.md). Files shared by both runs stay one level up.

```text
MyDrive/citecurv/
  smoke_report.json                     00, S0.5
  labels/rwdb_2026-07-06.csv            10, frozen label snapshot (SHA256 checked on every run)
  labeled_works_openalex.parquet        10, S1.3 DOI lookup cache (also adds labelled cited works to event_labels)
  slice_measurements.json, slice.yaml   10, S1.3 (copy slice.yaml into configs/ and commit)
  pilot/  or  full/                     RUN_DIR, one per run
    year=YYYY/works.parquet, edges.parquet               10, one pull per publication year
    year=YYYY/pull_state.json, pull_state_parts/         10, resume state of that year's pull (parts deleted after the merge)
    works.parquet                       10, all years merged; input of 20, 30, 50
    edges.parquet                       10, S1.1 contract (src_id, dst_id, t, doi_pair_key)
    events.parquet                      10, contract events (event_id, src_id, dst_id, t); input of 30, 40, 50
    labels.parquet                      10, doi_norm, reason_family, retraction_date, rwdb_snapshot_date
    event_labels.parquet                10, event_id, label (L12, PROVISIONAL(G3)); input of 50
    join_stats.json                     10, counts (copy into results/)
    dataset_manifest.json               10, S4.0 only: counts and SHA256 of the five parquet files (commit it)
    embeddings.npz                      20, ids + vectors (SPECTER2, 768 dimensions)
    embed_chunks/part_00000.npz ...     20, resume parts (about 15 for 300,000 works)
    split.parquet                       S4.1 (event_id, split); no notebook writes it yet; optional input of 30
    curvature_cache.parquet             30
    screen_report_by_year.csv, screen_report_by_field.csv, escalation_budget.csv, screen_thresholds.json   30
    tgn/data/                           40, DyGLib files from to_jodie: ml_<RUN>.csv, .npy, _node.npy, _idmap.npz, _meta.json
    tgn/runs/TGN_<RUN>_seed<s>/         40: checkpoint/, metrics.json, resolved_args.json,
                                            infer/ (tgn_event_scores.parquet, tgn_event_embeddings.npz)
```

`event_id` is assigned in one place, `citecurv.data.events.build_events`, which `10_data` calls: rows of
`edges.parquet` are sorted by (t, src_id, dst_id) and numbered from 0, so the ids do not depend on the order
in which OpenAlex returned pages; repeated (src_id, dst_id) pairs keep their earliest row. Never write an
`events.parquet` by hand. `30` and `40` read it with `read_events`, which checks the contract.

`50_eval` must read the same `RUN_DIR` (`events.parquet`, `event_labels.parquet`, `works.parquet`,
`embeddings.npz`, `curvature_cache.parquet`, `tgn/runs/...`) and write the S4.1 `split.parquet` and its run
folders there, so every artifact of a run is joined on the same `event_id`.

Rules (plan 9.1):

- Store results as a few large files (parquet, npz, json). Never let a Drive folder hold more than
  about 10,000 items; the mount fails past that.
- Chunk folders are the exception and stay small: `embed_chunks/` holds one file per 20,000 works;
  a resumable OpenAlex pull keeps one pair of part files per page (by default) in
  `pull_state_parts/` next to its state file, and deletes them after the merge. A year bucket of
  50,000 works is 500 pages, so 1,000 transient files; `pages_per_part=10` (argument of `pull_works`
  or `data.pages_per_part` in the config) cuts that tenfold, at the price of fetching up to 9 pages
  again after a crash.
- Training work folders (`/content/tgn/...`) stay on the runtime's local disk and are copied to
  Drive once a step finishes.
- Small result summaries (< 200 KB: counts, metrics, reports) are copied into `results/` in the
  repository and committed. Raw data, embeddings and checkpoints never go into git.

## 6. Checkpoint and resume, stage by stage

A Colab session can die at any time (idle timeout, usage limits, units running out). Plan 9.1 asks
for checkpoint-resume on anything longer than 30 minutes. What survives, per stage:

**10_data (OpenAlex pulls).** Pulls run one publication year at a time; a year whose
`edges.parquet` exists is skipped on rerun. Inside a year, `pull_works(..., state_path=...)` writes
the cursor and counts to a JSON state file after every page (atomically) and the page rows to part
files, so a rerun with the same filters and state file resumes at the saved cursor; parts missing on
Drive are fetched again. `10_data` passes one state file per year
(`RUN_DIR/year=Y/pull_state.json`) and, on `BudgetExhausted`, prints the resume instructions and stops.

Daily budget: a free key has $1 a day, reset at midnight UTC, which is about 10,000 list pages or
1,000,000 works at 100 per page (OpenAlex, 2026c). A 300,000-work slice is about 3,000 pages. If the
budget runs out mid-pull, `pull_works` stops with `BudgetExhausted`; its message gives the reset
time. Stop the notebook and rerun it after the reset with the same state file. A 429 that only
means "too fast" is retried automatically with exponential backoff (`data.max_retries` = 5,
`data.backoff_base_s` = 2 s: waits of 2, 4, 8, 16 and 32 s).

**20_embed (SPECTER2).** `encode_resumable` saves one npz part per 20,000 works in `embed_chunks/`.
After a disconnect, Run all again: finished parts are reused and encoding continues with the next
one. A part whose ids differ from the current works table raises `ValueError`; that means the works
table changed, so use a new chunk folder.

**30_curvature.** The Forman screen is written to `curvature_cache.parquet` once; a rerun loads it.
Escalation (ORC on flagged events) runs in chunks of `curvature.orc_escalate_chunk` events (5,000)
through `escalate(..., max_events=CHUNK)`, and the cache is written after every chunk; a rerun keeps
rows already computed and continues with the rest. The S2.3 budget grid is a sample and is simply
rerun.

**40_tgn (DyGLib TGN).** DyGLib has no mid-run resume. `train_tgn` runs DyGLib in a child process
with its work folder on local disk; the best-on-validation checkpoint is moved into the run folder
when training ends, and the notebook copies the run folder to Drive after that. A session that dies
during training loses that training run. What to do:

1. Train one seed per session (PROVISIONAL(G1): with Colab Pro, runs are chunked by seed; Pro+
   background execution would allow longer runs).
2. Estimate the run length from the pilot first: `metrics.json` lists `wall_clock_s` per epoch. Set
   the number of epochs so that one run fits a session with margin, and keep the browser tab open.
3. If the session dies during training, rerun the same seed from the start. The seed fixes the run,
   so the result is reproducible within tolerance on the same hardware.
4. If it dies after training but before inference: in a new session run cells 1 to 4 (conversion
   is deterministic), copy `DRIVE_RUN` back to `LOCAL_RUN` (`shutil.copytree`), then run cells 6 and 7.
   The checkpoint paths in `metrics.json` are relative to the run folder, so a copied folder works.
5. Write each death in the unit-burn log (section 7); repeated deaths are evidence for gate G1.

**50_eval.** Not written yet. The plan requires per-run folders under `runs/` with `metrics.json`,
so a dead session costs at most the run in progress.

## 7. What to record in PROGRESS.md

After every heavy run, add a row to the unit-burn log (columns: Date, Notebook / run, Runtime,
Units) and the Verify output to the task row. Record:

- date (UTC) and the TAG that ran;
- notebook, stage or task, and the run's identity (slice year, seed, variant);
- runtime: CPU / High-RAM / T4 / A100, and the runtime version (26.07 = Python 3.12, or the 3.13
  opt-in path);
- wall-clock time, and compute units before and after (Colab's resources panel, opened from the
  RAM/disk indicator at the top right, shows the units left);
- outcome: finished, or died at which point, and whether the rerun resumed;
- for OpenAlex pulls: pages, works, edges and `cost_usd_reported` from `PullStats` (or the
  `/rate-limit` usage before and after);
- versions printed by the notebook when they differ from the pins.

Row format: `<date> | 00_smoke, TAG <tag>, all four blocks, gate <PASS/FAIL> | T4, runtime 26.07 | <units> units, <minutes> min`.

## 8. Known failure modes and fixes

| Symptom | Cause | Fix |
|---|---|---|
| First cell stops: "This notebook needs Python 3.12, found 3.13..." | Colab's default runtime is Python 3.13 since 2026-08-25 | Runtime > Change runtime type > Runtime version 26.07, Save, Run all. Check again in every new session. |
| pip: no version satisfies `scipy<=1.13.1`, or a SciPy source build fails (meson or Fortran errors) | Python 3.13 runtime: SciPy 1.13.1 has no 3.13 wheel | As above. Off Colab, the 3.13 opt-in path (section 2) is the PROVISIONAL alternative. |
| "Restart needed: numpy ... loaded but ... installed", or import errors such as `numpy.dtype size changed` | pip replaced a package that was already imported (SciPy 1.13.1 requires NumPy < 2.3; adapters downgrades transformers) | Runtime > Restart session, then Run all. The installs are already done and finish quickly. |
| `20_embed` assertion "embed pins not active", or adapters will not install | Colab ships transformers 5.x (5.22.0 on the Python 3.13 image); adapters 1.3.0 requires `transformers~=4.57.6` | Let the install cell pin `transformers==4.57.6`, then restart the session if transformers was imported before. If the pair cannot be installed at all, the documented fallback is SciNCL (`citecurv.embed.specter2` docstring); record the switch, because its vectors are not interchangeable with SPECTER2. |
| DyGLib: `TypeError: Population must be a sequence`, `UnpicklingError: Weights only load failed`, an assertion on feature width 172, or argparse "invalid choice" for the dataset name | Unpatched upstream DyGLib (Python 3.11+, torch 2.6+, feature width, dataset names) | Never clone upstream DyGLib. Use the vendored copy from the tag (`third_party/DyGLib`, patches P1-P7 in its `PATCHES.md`); `40_tgn` and `00_smoke` set `CITECURV_DYGLIB_ROOT` to it. Do not use DyGLib's `preprocess_data.py` or `evaluate_link_prediction.py` (left unpatched). |
| `ZeroDivisionError: float division by zero` from GraphRicciCurvature | `method="ATD"` divides by the number of predecessors of u times successors of v, which is zero for almost every citation at its own time | Never use ATD. The config uses `OTDSinkhornMix` (equal to OTD at our sizes). |
| `DeprecationWarning: ... multi-threaded, use of fork() may lead to deadlocks` | networkit starts threads, GraphRicciCurvature forks a pool | Harmless so far (no hang in any run). If a cell ever hangs, keep `curvature.orc.proc: 1` (the default) and restart. |
| `BudgetExhausted: OpenAlex daily budget is spent ... resets at ... UTC` | The $1 daily budget is used up (confirmed by `/rate-limit` or the 429 headers) | Stop. Rerun the same cell after the reset time with the same `state_path`; the pull resumes at the saved cursor. Check that no other pull ran the same day. |
| Repeated `OpenAlex HTTP 429` errors without `BudgetExhausted` | Requests too fast, or the budget check gave no answer | Lower `data.max_requests_per_s`; run `check_key()` and look at `rate_limit_after.remaining`. |
| `OpenAlex HTTP 401` | No key, or an invalid key (cursor paging past the first page always needs a key; OpenAlex, 2026b) | Check the secret's name and notebook access. |
| `OpenAlex HTTP 403 ... key_not_activated` | The key was requested but its activation link was not clicked | Click the link in the activation email, then rerun. |
| `ApiKeyMissing` | `OPENALEX_API_KEY` is not set | Add the Colab secret and enable notebook access (section 4). |
| `check_key()` reports "not counted against the key" | The key is not reaching OpenAlex in the current key mode, or usage is booked late | Rerun after a minute. If it persists, set `OPENALEX_KEY_MODE=param` (or `bearer`) and rerun; note the working mode in `PROGRESS.md` so it can go into `data.api_key_mode`. |
| `00_smoke` Block C reports SKIP | No GPU in the runtime | Switch to a T4 runtime and rerun. If no GPU is available, run again later; the gate needs all four PASS. |
| `00_smoke` Block D: `ImportError: 'NeighborSampler' requires either 'pyg-lib' or 'torch-sparse'` from `DOMINANT.fit` | PyG's neighbour sampler has no backend: `pip install pygod torch_geometric` alone does not bring torch-sparse or pyg-lib (Colab run of 2026-11-03, tag v0.2.0) | The install cell adds `torch_scatter==2.1.2` and `torch_sparse==0.6.18` from `https://data.pyg.org/whl/torch-<torch version>.html` (for example `torch-2.11.0+cu128`). If they were installed after `torch_geometric` was imported, restart the session and run again. |
| "There are adapters available but none are activated for the forward pass" in Block C or `20_embed` | Logged by `adapters` while `load_specter2` loads the model, before the adapter is set active | Harmless. Checked 2026-11-03: after loading, the active setup is `Stack[specter2]`, and the embeddings change when the adapter is switched off. |
| `drive.mount` fails with "mount failed" | The Drive consent dialog in the tab was not answered within the 120 s timeout | Run the cell again and approve the dialog with the Colab account straight away. |
| Drive errors when listing or writing a folder | A folder with too many items | Keep the layout of section 5; never write per-event or per-work files. |

## 9. First-week checklist: S0.4, S0.5, S1.1-live, S1.2

**S0.4: OpenAlex key and cost meter (Maor).**

- [ ] Create the free key: openalex.org > Settings > API key (https://openalex.org/settings/api-key;
      OpenAlex, 2026a). The key works only after the account's email address is verified.
- [ ] Store it as the local environment variable `OPENALEX_API_KEY` and as the Colab secret of the
      same name (section 4). Do not set `mailto`: OpenAlex ignores it since February 2026.
- [ ] Run the key check from the repository root, in the local dev venv (`pip install -e .[dev]`
      from S0.3):
      `python -c "import json; from citecurv.data.openalex import check_key; print(json.dumps(check_key(), indent=2))"`
      It makes one list call (about $0.0001) between three `/rate-limit` calls; the first two
      measure what a `/rate-limit` call itself adds, if anything.
- [ ] Expected: `"ok": true`, `"counted_against_key": true`, `probe_status` 200 and a `probe_meta`
      block. Copy `verdict`, `key_mode`, `probe_meta` and `rate_limit_after.fields` into the S0.4 row
      of `PROGRESS.md` (they show the live field names, which the documentation does not list). If it
      says "not counted", follow the matching row in section 8.

**S0.5: smoke notebook (one Colab session, Maor).**

- [ ] Make sure a tag exists that contains `notebooks/00_smoke.ipynb`; set `TAG` in its parameters cell.
- [ ] Runtime: T4 GPU, runtime version 26.07. Run all; authorise the Drive mount when asked.
- [ ] Expected: Block A, B, C, D all PASS, then "S0.5 gate: PASS". Block B trains on synthetic data;
      in a CPU run here it reached validation AP 0.79 (threshold 0.6).
- [ ] Copy `MyDrive/citecurv/smoke_report.json` to `results/smoke_report_<date>.json`, fill in the
      S0.5 row and the unit-burn log in `PROGRESS.md`, commit.
- [ ] Any FAIL: read the block's `reason` or `error` in the report, find the row in section 8, and
      open the matching risk in plan section 10. A Block A FAIL also triggers the WSL2 check (plan 9.2).

**S1.1-live: first real pull (locally, or in a scratch cell of `10_data`).**

- [ ] Run a two-page pull with a state file, for example:

      from citecurv.config import load_config
      from citecurv.data.openalex import pull_works
      cfg = load_config("configs/default.yaml")
      s = pull_works({"publication_year": "2022", "primary_topic": {"subfield": {"id": "1702"}}},
                     "data/s11_live", max_works=200, config=cfg, state_path="data/s11_live/pull_state.json")
      print(s)

- [ ] Expected: 2 pages, 200 works, edges > 0, `est_cost_usd` 0.0002, `cost_usd_reported` close to it
      (None means the response meta has no `cost_usd` field: look at `first_meta` in the state file and
      write down the real field name). Both parquet files open with the contracted columns.
- [ ] Resume test: delete `data/s11_live`, start the same pull with `max_works=1000`, interrupt the
      kernel after a few pages, run it again: `pages_resumed` must be greater than 0 and the totals the
      same as an uninterrupted run.
- [ ] Record the numbers in the S1.1 row of `PROGRESS.md` and mark S1.1 done.

**S1.2: pilot pull and label join (`10_data`, CPU runtime).**

- [ ] In `10_data`: set `TAG`, `STAGE = "S1.2"`; keep `PILOT_CANDIDATE = "computing"`,
      `PILOT_YEAR = 2022`, `PILOT_MAX_WORKS = 10_000` (about 100 pages, $0.01). Run all.
- [ ] Expected: `pilot/works.parquet`, `edges.parquet`, `labels.parquet` and `join_stats.json` on Drive;
      the strict and the loose join counts are printed, and more than 0 labeled papers. If there are
      none, the plan says to switch the pilot to another shortlist candidate
      (`docs/slice_shortlist.md`).
- [ ] Copy `join_stats.json` (counts only) to `results/`, write both join counts and the join-rule
      decision into `PROGRESS.md` (S1.2), add the unit-burn row, commit.
- [ ] Then S1.3 in the same notebook with `STAGE = "S1.3"` (`docs/slice_shortlist.md`, last section).

## 10. Open items for the integrator (not operator tasks)

Closed on 2026-10-31: `10_data` passes `state_path` and catches `BudgetExhausted`; `events.parquet`
is built in code (`citecurv.data.events`); one `RUN_DIR` layout for all notebooks; `20_embed` has the
Python check; `slice_query` fetches through `citecurv.data.openalex` (key modes + budget check).

Still open:
- `30_curvature` and `configs/pins.md`: the GraphRicciCurvature version (decision L5, Maor;
  recommendation 0.6.1 in `docs/orc_version_comparison.md`).
- No code writes `split.parquet` yet; `30` reads it from `RUN_DIR` (write it from `citecurv.eval.split`
  at S4.1, or from `50_eval`'s pilot cells).
- `embeddings.npz` covers `works.parquet` only; cited works outside the slice get zero rows in
  `to_jodie` and NaN in `static_cosine` mode. Decide at S1.3 whether to embed out-of-slice cited works.
  Not decided at S1.3; being tested on the pilot since 2026-11-12 with `25_cited_text` (PROGRESS.md, S1.7).

## References

Google Colaboratory. (2026, August 18). *Upgrade to Python 3.13* (Issue #6081) [GitHub issue].
https://github.com/googlecolab/colabtools/issues/6081

OpenAlex. (2026a, October 15). *Authentication*. https://help.openalex.org/api/authentication/
(source: https://github.com/ourresearch/openalex-help, `content/api/authentication.md`)

OpenAlex. (2026b, October 14). *Page through results*. https://help.openalex.org/api/paging/
(source: https://github.com/ourresearch/openalex-help, `content/api/paging.md`)

OpenAlex. (2026c, October 23). *Example costs*. https://help.openalex.org/access/example-costs/
(source: https://github.com/ourresearch/openalex-help, `content/access/example-costs.md`)

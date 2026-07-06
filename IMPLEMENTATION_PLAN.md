# Phase B implementation plan — Detecting Network Citation Anomalies Using Graph Curvature

> **For agentic workers:** this plan is executed by Claude Code sessions on Maor's machine, task by task, using superpowers:executing-plans discipline (subagent-driven execution is allowed for isolated code tasks). Steps use checkbox (`- [ ]`) syntax. Read `PROGRESS.md` before doing anything; update it before ending any session.

**Goal:** build, calibrate, and evaluate the system committed in the Phase A book: a directed temporal citation-anomaly detector combining a DyGLib TGN encoder (SPECTER2-initialized memory, bidirectional messaging) with hybrid directed curvature (Forman screen, Ollivier escalation) and a four-component interpretable anomaly score.

**Architecture:** four replaceable pipeline components (data, curvature, temporal encoder, scoring head) as a `src/citecurv` Python package; five thin per-stage Colab notebooks drive the heavy runs; all artifacts persist to Drive as few large files; evaluation follows the book's protocol (time split, seeds with spread, five-variant ablation, three baseline tiers, synthetic injection, top-50 manual review).

**Tech stack:** Python 3.12 · pyalex 0.21 · SPECTER2 (adapters 1.3.0 / transformers 4.57.x) · DyGLib (vendored, MIT) · GraphRicciCurvature 0.5.3.2 · custom directed Forman module · PyGOD 1.1.0 + PyG 2.5–2.6 · CurvGAD (sealed env) · XGBoost / scikit-learn · Colab (primary) / WSL2 (fallback).

**Status:** plan approved 2026-07-06. Sources: Phase A book (23/6 version), Component research findings 2026-07-02, Book audit findings 2026-07-02, decision rounds with Maor 2026-07-06.

**Document rule:** this markdown file is the single source of truth and the executable copy. The `.docx` in "Summeries from claude" is a generated snapshot — regenerate it from this file, never edit it by hand.

## Global constraints

Every task implicitly includes these. Violating one is a bug even if the task text doesn't repeat it.

- Python 3.12 everywhere (Colab default; inside the curvature stack's 3.10–3.12 window; xgboost 3.3.0 floor).
- Versions come from `configs/pins.md` (section 9.3). Never `pip install` unpinned in a notebook.
- No information leakage: any feature used to score an event at time t is computed from graph state with timestamp ≤ t (NFR3). Every new feature gets a leak argument written down.
- Determinism: seeds fixed to `[0, 1, 2, 3, 4]`; same data + same seed → same scores within tolerance 1e-6 (NFR4).
- All runtime parameters live in `configs/*.yaml`, never hardcoded (NFR9).
- Raw data, caches, embeddings, checkpoints stay OUT of git (`.gitignore`); they live on Drive (Colab) or `data/` locally. Code, configs, small result JSONs, and this plan live IN git.
- Secrets (OpenAlex key, S2 key) live in Colab Secrets / local env vars — never in code, notebooks, or commits.
- License rules: never copy code from GeneralDyG (no license — pattern reference only). TGTOD (Apache-2.0) may be adapted with attribution. CGNN's NCTM/CNM are reimplemented from the math, not copied (its PyG 1.5 pin is unusable anyway). DyGLib (MIT) is vendored with its LICENSE preserved. Retraction Watch data is described as "publicly available via Crossref" — never claim CC0.
- Anything written for the book or advisors: English only, APA author–year citations, no AI-tell phrasing, and no factual claim that wasn't verified against a live source or one of the three grounding documents.
- Human-gate policy (Maor, 2026-07-06): when a pending human decision (section 2) blocks nothing technically, proceed with the recorded default, mark affected outputs `PROVISIONAL(Gn)` in `PROGRESS.md`, and continue.

## Task tags

- *(untagged)* — Claude does it end to end on this machine (Windows-safe or WSL2).
- `[COLAB-RUN]` — Claude prepares the notebook/script completely; Maor presses run on Colab; outputs land on Drive/repo; Claude analyzes them next session.
- `[HUMAN]` — only Maor/Gilboa can do it (accounts, money, advisors, manual review).

---

## 1. Standing decisions (locked)

| # | Decision | Choice | Source |
|---|----------|--------|--------|
| L1 | Temporal encoder | Plain TGN in DyGLib; transformer appears only in the anomaly-scoring head | Book 2.2.3/2.2.4; transformer decision 2026-06 |
| L2 | Encoder extensions | Text-initialized memory + bidirectional messaging (Shen et al. ideas, SPECTER2 instead of SciBERT) | Book 2.1.2/2.2.3 |
| L3 | Curvature strategy | Directed Forman-Ricci as global linear-time screen; directed Ollivier-Ricci only on flagged subgraphs | Book 1.4/2.2.2 |
| L4 | Directed Forman source | Custom module implemented from Saucan et al. (2019) closed form — no package exists; GeneralisedFormanRicci is out of the stack | Research §2.1; audit fix 2 |
| L5 | Directed ORC source | GraphRicciCurvature pinned 0.5.3.2 (directed support confirmed in source; dormant upstream) | Research §4.4 |
| L6 | Text embeddings | SPECTER2: `allenai/specter2_base` + `allenai/specter2` proximity adapter (naming trap: the latter is the adapter) | Research §4.2 |
| L7 | Data source | OpenAlex REST API with free key ($1/day credit); snapshot path dropped (dead on Colab) | Research §4.1, §8 |
| L8 | Client | pyalex 0.21 + custom 429 backoff (no built-in handling, issue #100) | Research §4.1 |
| L9 | Labels source | Retraction Watch CSV from GitLab (the only correct source), frozen dated snapshot | Research §4.1 |
| L10 | Compute venue | Colab primary (college-funded), WSL2 fallback; five per-stage notebooks + pip-installable `src/` from repo | Maor 2026-07-06; research §8 |
| L11 | Data slice | Retraction-dense slice: fields/years where positive labels concentrate; exact filters picked by the S1.3 label-density query; target ≈300K works / 2–3M edges (Shen-scale) | Maor 2026-07-06 (D11) |
| L12 | Positive-label rule | Referencing/Attributions reason family primary (14,769 records as of 2026-06-17); broadened family set as one sensitivity run | Maor 2026-07-06 (D13); audit item |
| L13 | Score combination | Four named components (embedding surprise, geometric anomaly, temporal anomaly, curvature dynamics); fixed calibrated weights first, small learned combiner as one extra ablation cell | Maor 2026-07-06 (D1); book 1.4 |
| L14 | Temporal framing | Continuous-time event stream; curvature computed on graph state at time t (slicing module is first-class work) | Book 2.1.2; research D2 |
| L15 | Explainability scope | The named score breakdown only (FR7/NFR7); PyG explainers are future work | Research D3 |
| L16 | Extra score components | A_meta / A_sem from the April plan stay out (future work) | Research D1 |
| L17 | Baseline set | CurvGAD (sealed env) + de-curvatured own TGN + PyGOD GNN tier (DOMINANT, CoLA, GADNR) + classical ML tier (XGBoost) + statistical tier | Book 2.2.5/4.1.2 (audit-corrected); research D7 |
| L18 | Synthetic injection | Custom injector on the CIDRE donor/recipient reciprocity pattern; PyGOD `gen_structural_outlier(directed=True)` clique injector as crude variant | Research D14 |
| L19 | Cold start | Papers enter scoring after k events; k set in calibration (S2.4); provisional default k=3 | Research D5; Maor 2026-07-06 |
| L20 | Repo | `github.com/Natboa/Detecting-Network-Citation-Anomalies-Using-Graph-Curvature` (must be created/made public — currently 404) | Research §2.4; book title page |

## 2. Pending human decisions (defaults active)

Tracked live in `PROGRESS.md`. Policy: proceed with the default, mark outputs `PROVISIONAL(Gn)`.

| Gate | Question | Owner | Default in force | What is provisional under it |
|------|----------|-------|------------------|------------------------------|
| G1 | Colab Pro or Pro+ (background execution up to 24 h matters for the ablation matrix) | Maor → college | Pro | Long-run training schedule in S4.2/S4.3 (chunked with checkpoint-resume instead of background runs) |
| G2 | Are notebooks the required deliverable format, or only the compute venue? | Maor → advisors | Venue only; `src/` package is the deliverable, notebooks stay thin drivers | Notebook polish level |
| G3 | Sign-off on the positive-label rule (L12) | Maor → advisors | Referencing/Attributions family | Slice choice rationale (S1.3), all evaluation numbers (cheap to re-join: scores are stored, labels join afterwards) |
| G4 | Audit Q1: is a solution-description section (pseudocode/flowcharts) required in the book? | Maor → advisors | Assume yes for the Phase B book; does not block implementation | Book-writing hooks in S4.9 |
| G5 | Audit Q2: page-limit accounting for the Phase A book | Maor → advisors | Book track only | Nothing in this plan |
| G6 | Repo visibility timing (public now vs. before first deliverable) | Maor + Gilboa | Public at creation | Nothing — private works for execution, link resolution waits |

## 3. Repository layout and conventions

Locked at S0.3. The working folder `C:\Users\maor4\OneDrive\Desktop\Detecting-Network-Citation-Anomalies-Using-Graph-Curvature` becomes the clone of L20.

```
Detecting-Network-Citation-Anomalies-Using-Graph-Curvature/
├── IMPLEMENTATION_PLAN.md          # this file
├── PROGRESS.md                     # live status, ledgers, session log
├── pyproject.toml                  # makes src/citecurv pip-installable from GitHub
├── configs/
│   ├── default.yaml                # every runtime knob (NFR9)
│   ├── slice.yaml                  # written by S1.3 (OpenAlex filters)
│   └── pins.md                     # version pin tables (section 9.3)
├── src/citecurv/
│   ├── config.py                   # yaml loader, seed fixing
│   ├── data/
│   │   ├── openalex.py             # pull harness: pull_works(filters, out_dir, max_works=None) -> PullStats
│   │   ├── abstracts.py            # invert_abstract(inv_idx) -> str | None
│   │   ├── doi.py                  # normalize_doi(raw) -> str | None
│   │   ├── rwdb.py                 # load_labels(csv_path, families) -> DataFrame
│   │   └── slice_query.py          # S1.3 label-density analysis
│   ├── embed/specter2.py           # encode(papers, batch_size) -> np.ndarray (N, 768)
│   ├── curvature/
│   │   ├── forman_directed.py      # forman_directed(G, weight=None) -> dict[(u,v), float]
│   │   ├── orc.py                  # orc_directed(G, edges, **knobs) -> dict[(u,v), float]
│   │   └── timeslice.py            # EventStore: subgraph_at(u,v,t,radius), degrees_at(t)
│   ├── tgn/
│   │   ├── convert.py              # to_jodie(edges_df, node_emb, out_dir, name)
│   │   └── train.py                # thin wrapper around vendored DyGLib entry points
│   ├── scoring/
│   │   ├── components.py           # score_event(...) -> ScoreBreakdown (4 parts + total)
│   │   ├── combiner.py             # fixed weights + optional learned logistic head
│   │   └── aggregate.py            # per-paper / per-author rollups (FR8), export (FR9)
│   └── eval/
│       ├── metrics.py              # ranking_metrics(scores, labels) -> dict
│       ├── split.py                # time cutoff, train/val/test freeze
│       ├── inject.py               # inject_cartel(...) -> (events, truth_ids)
│       ├── ablation.py             # variant/seed run matrix driver
│       └── baselines/              # tier1_stats.py, tier2_xgb.py, pygod_tier.py, curvgad_notes.md
├── third_party/DyGLib/             # vendored MIT clone + PATCHES.md (every diff documented)
├── notebooks/
│   ├── 00_smoke.ipynb              # S0.5 environment gates
│   ├── 10_data.ipynb               # pulls (CPU runtime)
│   ├── 20_embed.ipynb              # SPECTER2 (GPU)
│   ├── 30_curvature.ipynb          # Forman sweep + ORC escalation (CPU/high-RAM)
│   ├── 40_tgn.ipynb                # training (GPU)
│   └── 50_eval.ipynb               # ablations, baselines, metrics (GPU)
├── tests/                          # pytest; mirrors src layout
└── .gitignore                      # data/, *.npz, *.parquet, checkpoints/, .env
```

**Artifact contracts** (parquet unless stated; all under `data/` locally, `Drive/citecurv/` on Colab — few large files, never >10K items per folder):

- `works.parquet`: `work_id, doi_norm, title, abstract, pub_date, concepts, venue_id, is_retracted_oa`
- `edges.parquet`: `src_id, dst_id, t (days since epoch), doi_pair_key` — one row per citation event, src cites dst
- `labels.parquet`: `doi_norm, reason_family, retraction_date, rwdb_snapshot_date`
- `embeddings.npz`: `ids (N,), vectors (N, 768) float32`
- `curvature_cache.parquet`: `src_id, dst_id, t, forman, orc (nullable), radius, computed_at_stage`
- `scores.parquet`: `src_id, dst_id, t, s_embed, s_geom, s_temp, s_curvdyn, total, model_variant, seed, config_hash` — NFR10: enough columns to trace any flag back to inputs
- `runs/<variant>_<seed>/`: checkpoint + `metrics.json` + resolved config copy

**Definition-of-done discipline:** a task is done only when its Verify line has been executed and its output recorded in `PROGRESS.md`. After the repo exists, every task ends with a commit.

---

## 4. Stage 0 — pre-flight

Purpose: repo, keys, environment go/no-go, frozen labels. Exit criteria at the end of the section.

### Task S0.1 `[HUMAN]` Create the GitHub repository

**Steps:**
- [ ] Create repo `Natboa/Detecting-Network-Citation-Anomalies-Using-Graph-Curvature` (exact name — the book's title-page link must resolve), MIT license, README stub. Public per G6 default (private acceptable until first deliverable).
- [ ] Add Maor's account as collaborator if it lives under Gilboa's.

**Verify:** `https://github.com/Natboa/Detecting-Network-Citation-Anomalies-Using-Graph-Curvature` returns 200 while logged out (or collaborators confirmed, if private for now).

### Task S0.2 Turn the working folder into the clone

**Files:** the whole working folder.

**Steps:**
- [ ] `git init` in the working folder, add remote, pull the README/LICENSE from S0.1, resolve.
- [ ] Write `.gitignore` first commit: `data/`, `*.parquet`, `*.npz`, `checkpoints/`, `runs/`, `.env`, `__pycache__/`, `.ipynb_checkpoints/`.
- [ ] Commit `IMPLEMENTATION_PLAN.md` + `PROGRESS.md`.

**Note:** the folder is OneDrive-synced. Acceptable because git tracks only small text (data is ignored); if OneDrive/git contention appears (lock errors, sync storms), move the clone to `C:\dev\` and leave a pointer file — record in `PROGRESS.md`.

**Verify:** `git log --oneline` shows the initial commits; `git status` clean; push succeeds.

### Task S0.3 Package skeleton

**Files:** create `pyproject.toml`, `src/citecurv/` tree per section 3, `configs/default.yaml`, `tests/test_smoke.py`.

**Steps:**
- [ ] `pyproject.toml`: name `citecurv`, version `0.0.1`, requires-python `>=3.12`, no dependencies pinned here (per-stage pins live in notebooks/pins.md; the package itself imports lazily).
- [ ] Empty modules with docstrings + the public function signatures from section 3 raising `NotImplementedError`.
- [ ] `configs/default.yaml` skeleton:

```yaml
seed_list: [0, 1, 2, 3, 4]
data:
  page_size: 100          # design for 100 per docs; live accepts 200 today
  mailto: <project email>
  max_retries: 5
  backoff_base_s: 2.0
slice: {}                  # filled by S1.3 -> configs/slice.yaml
curvature:
  screen_percentile_low: null    # set in S2.3
  screen_percentile_high: null   # set in S2.3
  orc_radius: null               # set in S2.3
  orc_subgraph_edge_cap: 5000    # April-plan knob, revisit in S2.3
  orc: {alpha: 0.5, method: OTDSinkhornMix, nbr_topk: 3000}
tgn:
  node_feat_dim: 768
  memory_projection_dim: null    # fallback knob if 768-dim memory too heavy
scoring:
  cold_start_k: 3                # provisional (L19), finalized in S2.4
  weights: {s_embed: 1.0, s_geom: 1.0, s_temp: 1.0, s_curvdyn: 1.0}  # calibrated in S3.2
eval:
  cutoff_date: null              # set in S4.1
  top_k: [100, 1000]
```

- [ ] `tests/test_smoke.py`: imports every module, loads the yaml, asserts seed list. Run `pytest -q` → passes.
- [ ] Commit.

**Verify:** `pip install -e .` succeeds in a fresh venv; `pytest -q` green; `python -c "import citecurv"` works.

### Task S0.4 OpenAlex key and cost meter

**Steps:**
- [ ] `[HUMAN]` Register the free key at `openalex.org/settings/api` under the project email; put it in local env `OPENALEX_API_KEY` and (later) Colab Secrets under the same name.
- [ ] Wire `citecurv.data.openalex` to read the key from env and always send `mailto`.
- [ ] Make one keyed list call (any works filter, one page) and print the response `meta` block.

**Verify:** response meta shows the cost field (~$0.0001 per list page per the 2 Jul verification) and no 403; key absent → code raises a clear error instead of falling back silently to keyless.

### Task S0.5 `[COLAB-RUN]` Smoke-test notebook — four go/no-go gates

**Files:** `notebooks/00_smoke.ipynb`.

Each block is independent, prints `PASS`/`FAIL` + versions, and writes a combined `smoke_report.json` to Drive. Claude writes the notebook completely; Maor runs it on a fresh Colab (default CPU runtime is fine except block C).

- [ ] **Block A — directed ORC.** `pip install GraphRicciCurvature==0.5.3.2 "scipy<=1.13.1"`. Build the toy directed graphs from S1.4's test table as `nx.DiGraph`, run `OllivierRicci` with directed input, `alpha=0.5, method="OTD"`. PASS = runs without hanging (fork multiprocessing works on Colab) and curvature signs match the expectations table. Record exact output values in the report for S1.5.
- [ ] **Block B — DyGLib stock TGN.** Clone DyGLib, download its bundled Wikipedia dataset per `DG_data/DATASETS_README.md`, run 1 training epoch of TGN with current torch 2.x. PASS = epoch completes, loss finite. (This has never been tested upstream against torch 2.x — this block is the compatibility gate.)
- [ ] **Block C — SPECTER2 (GPU runtime).** `pip install adapters==1.3.0` (brings transformers 4.57.x), load `allenai/specter2_base`, attach `allenai/specter2` proximity adapter, encode 3 sample title+abstract strings. PASS = output shape `(3, 768)`.
- [ ] **Block D — PyGOD + injectors.** `pip install pygod==1.1.0 torch_geometric==2.6.*`. Import DOMINANT, CoLA, GADNR; call `pygod.generator.gen_structural_outlier` on a small directed graph with `directed=True`. PASS = returns modified graph + outlier mask.

**Verify:** `smoke_report.json` on Drive shows 4× PASS with pinned versions. Any FAIL → open the matching risk row (section 10) before proceeding; Block A FAIL additionally triggers the WSL2 fallback check (section 9.2).

### Task S0.6 Freeze the Retraction Watch snapshot and count label families

**Files:** `src/citecurv/data/rwdb.py`, `tests/test_rwdb.py`; artifact `data/labels/rwdb_<date>.csv` + `data/labels/rwdb_counts.json` (counts json IS committed).

**Steps:**
- [ ] Download the current CSV from the Retraction Watch GitLab repository (the only correct source — the old Crossref Labs endpoint is dead), save with the snapshot date in the filename, record its SHA256 in `rwdb_counts.json`.
- [ ] Implement `load_labels(csv_path, families)`: parse the 19-column CSV (`RetractionDOI`, `OriginalPaperDOI`, `Reason`, dates), normalize DOIs with `doi.normalize_doi`, explode the semicolon-joined `Reason` field, filter to requested families.
- [ ] Count and store: total records; records whose Reason set intersects the Referencing/Attributions family; the candidate broadened set (list every distinct Reason value containing `Paper Mill`, `Fake Peer Review`, `Reference`, `Attribution`, `Citation` — the exact broadened list is chosen from this vocabulary and recorded in `rwdb_counts.json`); non-null `OriginalPaperDOI` rate.
- [ ] `tests/test_rwdb.py`: a 5-row fixture CSV exercising DOI normalization, multi-reason explosion, family filtering.
- [ ] Commit code + counts json (not the CSV).

**Verify:** `pytest tests/test_rwdb.py -q` green; counts json exists; Referencing/Attributions count is in the expected order of magnitude (~14–16K; it was 14,769 on 2026-06-17 — a large deviation means the parse is wrong, investigate before proceeding).

### Task S0.7 Book carry-over fixes (parallel track, never blocks)

The Phase A book edits from the 2 July audit. Separate track: any session may pick it up; nothing in stages 1–4 depends on it.

- [ ] Apply the 3 must-fixes: Ekle & Eberle reference → arXiv:2406.00134 / ACM TKDD 18(8), art. 192, 2024, and "(2023)"→"(2024)" in 2.1.1; section 2.2.2 + Table 2 → "directed Forman-Ricci is implemented directly from the Saucan et al. (2019) formulas", GeneralisedFormanRicci removed from Table 2 and links; 4.1.2 → "a temporal model without curvature (the TGN backbone of Shen et al., 2024, adapted for event scoring)".
- [ ] Apply the small fixes: CurvGAD branch description (edge-curvature reconstruction / structure+attribute reconstruction), RWDB category naming per L12, "incredibly dense"→"remains very large even after filtering", the three listed typos.
- [ ] Follow the python-docx rules learned in Phase A: never match-first-occurrence blindly (TOC contains section titles too), never reversed-iteration+addprevious, always extract-and-diff after saving.

**Verify:** re-run the extraction script on the saved copy; diff shows exactly the intended changes and the reference list stays alphabetical; a fresh citation-check of the three edited entries against arXiv/ACM records.

**Stage 0 exit criteria:** repo exists and the working folder pushes to it; `pip install -e .` + smoke pytest green locally; `smoke_report.json` = 4× PASS on Colab; OpenAlex key live with cost meter observed; RWDB snapshot frozen with family counts committed. (S0.7 is exempt — parallel track.)

---

## 5. Stage 1 — feasibility (components in isolation)

Book research stage 1. Each component proves itself on pilot data before integration.

### Task S1.1 Data pull harness

**Files:** `src/citecurv/data/openalex.py`, `abstracts.py`, `doi.py`; `tests/test_doi.py`, `tests/test_abstracts.py`, `tests/test_openalex.py`.

**Interfaces produced:**
- `normalize_doi(raw: str | None) -> str | None` — lowercase; strip `https://doi.org/`, `http://dx.doi.org/`, `doi:` prefixes; strip trailing whitespace/punctuation; return None for empty/malformed (no `10.` prefix).
- `invert_abstract(inv_idx: dict[str, list[int]] | None) -> str | None` — rebuild text from OpenAlex `abstract_inverted_index`; None in → None out.
- `pull_works(filters: dict, out_dir: Path, max_works: int | None) -> PullStats` — cursor pagination, page size from config, retry with exponential backoff on 429/5xx (pyalex has none built in — issue #100), per-call cost ledger accumulated into `PullStats(works, edges, pages, est_cost_usd)`; writes `works.parquet` + `edges.parquet` per the section 3 contracts (edges come from each work's `referenced_works` — direction: the pulled work cites, so `src=work, dst=referenced`, `t=pub_date` of the citing work).

**Steps:**
- [ ] Write failing tests first. `test_doi.py` cases: `"HTTPS://DOI.ORG/10.1234/AbC.5"` → `"10.1234/abc.5"`; `"doi:10.1/x"` → `"10.1/x"`; `""` → None; `"not-a-doi"` → None; version-suffix case `"10.1/x.v2"` survives as-is (version handling = exact keep; the loose-join variant is a labels-side concern, S1.2).
- [ ] `test_abstracts.py`: `{"Deep": [0], "learning": [1], "wins": [2]}` → `"Deep learning wins"`; None → None; empty dict → None.
- [ ] Implement; `pytest -q` green.
- [ ] Implement `pull_works` with the backoff wrapper; `test_openalex.py` mocks the client (no network in tests) and asserts: cursor loop termination, 429 → sleeps and retries, cost accumulation, parquet schema columns exactly as contracted.
- [ ] Commit.

**Verify:** `pytest tests/ -q` green; a real 2-page pull (~200 works, any filter) writes valid parquet, `PullStats.est_cost_usd` ≈ pages × 0.0001.

### Task S1.2 Pilot pull + label join

**Files:** `notebooks/10_data.ipynb` (pilot cells) or local run; `src/citecurv/data/rwdb.py` (join helper).

**Steps:**
- [ ] Pull a pilot slice: one mid-size concept + recent decade, ~5–10K works with references (runs locally on Windows — pure API work, no Linux need).
- [ ] Join works ↔ labels two ways and report both counts: strict (`doi_norm` equality) and loose (strip version suffixes `.v\d+$` before comparing). Record the delta — this is the evidence for the join rule decision.
- [ ] Fix the join rule (expected: strict on normalized DOIs, loose only if the delta is material — decide from data, record in `PROGRESS.md`).
- [ ] Commit stats json (counts only).

**Verify:** join produces >0 labeled papers on the pilot (if 0, the pilot concept is label-poor — switch pilot concept to one of the S1.3 candidates and rerun); both join counts recorded.

### Task S1.3 Slice selection — turn L11 into concrete filters

**Files:** `src/citecurv/data/slice_query.py`; output `configs/slice.yaml` + a one-page rationale note `docs/slice_rationale.md` (book-ready).

**Steps:**
- [ ] Using grouped/count OpenAlex queries (cheap list pages), measure per candidate slice (concept × year-range, optionally source/venue): works count, works-with-references share, RWDB-labeled paper count inside the slice (join via S1.2 rule), labeled count in the *late* period (candidate test years).
- [ ] Candidates to measure first (retraction-dense per the RWDB analysis): medicine/oncology-adjacent concepts and CS-adjacent concepts over ~2005–2025 (the Hindawi paper-mill era), then adjust from the data.
- [ ] Selection criteria (all must hold): ≈300K works (±50%); ≥2–3M citation edges (estimate: works × mean refs in slice); ≥300 labeled papers with ≥100 in the candidate test period (enough positives to rank); pull fits inside the 1M-results/day API cap and ≈$0.50 budget (≈300K works ≈ 3,000 list pages).
- [ ] Write `configs/slice.yaml` (exact OpenAlex filter string, year bounds, expected counts) and `docs/slice_rationale.md` (why this slice, the measured numbers, `PROVISIONAL(G3)` banner).
- [ ] Commit both.

**Verify:** slice.yaml filters re-run against the API reproduce the recorded counts within 5%; rationale note has no unverified numbers.

### Task S1.4 Directed Forman-Ricci module

**Files:** `src/citecurv/curvature/forman_directed.py`; `tests/test_forman_directed.py`. Pure NetworkX/NumPy — runs anywhere including Windows.

**Interfaces produced:** `forman_edge(G, u, v, weight=None) -> float`; `forman_directed(G, weight=None) -> dict[tuple, float]` (all edges).

Closed form (Saucan et al., 2019), edge e = v1→v2 with edge weight w_e, node weights w_v:

```
F(e) = w_e * [ (w_v1/w_e − Σ_{e_l ∈ in(v1)} w_v1/sqrt(w_e·w_{e_l}))
             + (w_v2/w_e − Σ_{e_l ∈ out(v2)} w_v2/sqrt(w_e·w_{e_l})) ]
```

Unweighted (all weights 1): `F(e) = 2 − in_deg(v1) − out_deg(v2)`.

- [ ] **Step 1 — failing tests.** Hand-computed table (unweighted):

```python
import networkx as nx
CASES = [
    # (edges, tested_edge, expected F = 2 - in_deg(u) - out_deg(v))
    ([("u","v")],                                   ("u","v"),  2.0),   # isolated edge
    ([("a","b"), ("b","c")],                        ("a","b"),  1.0),   # chain head
    ([("a","b"), ("b","c")],                        ("b","c"),  1.0),   # chain tail
    ([("x1","u"),("x2","u"),("x3","u"),("u","v")],  ("u","v"), -1.0),   # 3 in-edges at source
    ([("a","b"), ("b","c"), ("c","a")],             ("a","b"),  0.0),   # directed 3-cycle
    ([("u","v"),("v","r1"),("v","r2"),("v","r3"),
      ("v","r4"),("v","r5")],                       ("u","v"), -3.0),   # target cites 5
]

def test_unweighted_cases():
    for edges, e, expected in CASES:
        G = nx.DiGraph(edges)
        assert forman_edge(G, *e) == expected

def test_weighted_reduces_to_unweighted():
    # all weights exactly 1.0 -> weighted formula == unweighted, random graphs, 3 seeds
    for seed in (0, 1, 2):
        G = nx.gnp_random_graph(30, 0.1, seed=seed, directed=True)
        nx.set_edge_attributes(G, 1.0, "w")
        nx.set_node_attributes(G, 1.0, "node_w")
        for u, v in G.edges:
            assert abs(forman_edge(G, u, v, weight="w") - forman_edge(G, u, v)) < 1e-12
```

- [ ] **Step 2:** run → fails (`NotImplementedError`).
- [ ] **Step 3:** implement both functions exactly from the formula; `in(v1)` excludes e itself if present as v2→v1? No — `in(v1)` is the in-edge set of v1, and e=v1→v2 is not an in-edge of v1, so no exclusion arises; document this in the docstring. Cost must be O(in_deg(v1)+out_deg(v2)) per edge.
- [ ] **Step 4:** run → green. Also add the Saucan-paper worked examples if their toy graphs are recoverable from the paper's figures at execution time (fetch the paper; if figures give explicit values, encode them as cases; record which in the test docstring).
- [ ] **Step 5:** micro-benchmark on `gnp_random_graph(10_000, 3e-4, directed=True)`: full sweep < 5 s single-threaded. Commit.

**Verify:** `pytest tests/test_forman_directed.py -q` green; benchmark number recorded in `PROGRESS.md`.

### Task S1.5 `[COLAB-RUN]` Directed ORC validation

**Files:** `notebooks/30_curvature.ipynb` (validation cells); `src/citecurv/curvature/orc.py`.

**Steps:**
- [ ] `orc.py`: thin wrapper around `GraphRicciCurvature.OllivierRicci` for directed graphs, exposing the knobs from config (`alpha`, `method` OTD/ATD/Sinkhorn/OTDSinkhornMix, `nbr_topk`, `proc`), taking an edge list to restrict computation.
- [ ] Notebook cells: run on the S1.4 toy graphs + a 200-node random DAG; assert directed behavior (mass on in-neighbours at source, out-neighbours at target — flipping an edge's direction changes its curvature where the neighborhood is asymmetric); compare OTD vs OTDSinkhornMix values on the same edges (mix should approximate OTD).
- [ ] Persist `orc_validation.json` to Drive with all values.

**Verify:** direction-sensitivity assertion holds; OTD vs mix max abs deviation recorded and < 0.05 on the toy set (if larger, note it and prefer OTD for small subgraphs in config).

### Task S1.6 `[COLAB-RUN]` SPECTER2 encoding pilot

**Files:** `src/citecurv/embed/specter2.py`; `notebooks/20_embed.ipynb`.

**Steps:**
- [ ] `encode(papers, batch_size=32)`: input `(id, title, abstract|None)`; text = `title + tokenizer.sep_token + abstract` when abstract present, title alone otherwise (FR3 fallback); truncation at 512 tokens; model in eval mode, no dropout; returns float32 `(N, 768)`; writes `embeddings.npz` per contract.
- [ ] Unit test (CPU, tiny): fallback path chooses title-only when abstract is None; output dims; determinism (two calls → identical vectors).
- [ ] Notebook: encode the S1.2 pilot works on GPU; record throughput (docs/min) and unit burn; sanity: cosine similarity of two same-field papers > two cross-field papers (3 hand-picked triples).
- [ ] Commit code + tests; embeddings stay on Drive.

**Verify:** pytest green locally (with model download or mocked tokenizer — mock if download too heavy locally); pilot npz on Drive with N = pilot works count; throughput recorded (extrapolate: full slice ≈300K docs must be ≤ a few GPU-hours; if not, batch/precision tuning before S4.0).

### Task S1.7 `[COLAB-RUN]` DyGLib integration on the pilot slice

**Files:** `third_party/DyGLib/` (vendored at this task), `PATCHES.md`, `src/citecurv/tgn/convert.py`, `tests/test_convert.py`; `notebooks/40_tgn.ipynb`.

**Steps:**
- [ ] Vendor DyGLib (MIT, LICENSE preserved) at its last upstream commit; record commit hash in `PATCHES.md`.
- [ ] `to_jodie(edges_df, node_emb, out_dir, name)`: emit `ml_{name}.csv` (`u, i, ts, label, idx` — JODIE format, indices shifted +1 per DyGLib convention), `ml_{name}.npy` (edge features: at minimum the citation-direction indicator the book requires — bidirectional messaging must not erase direction), `ml_{name}_node.npy` (row 0 = padding zeros, then SPECTER2 vectors aligned to the +1 index shift). Unit test with a 5-event fixture checks shapes, the +1 shift, padding row, direction feature values.
- [ ] Patch 1 in `PATCHES.md`: `utils/DataLoader.py` `NODE_FEAT_DIM = 172` → read from data (or 768); keep the assert but derive it. (Upstream writes zero node features and hardcodes 172; both facts verified 2 Jul.)
- [ ] Notebook: convert pilot slice, train TGN a few epochs, transductive link prediction. Checks: loss decreases; eval AP above random; memory footprint printed (768-dim memory ≈ 3 GB per 1M nodes — pilot is far below); wall-clock per epoch recorded.
- [ ] Leak check: assert the loader never serves an event with `ts` greater than the current batch time to the memory module (add a debug assertion in the wrapper, run one epoch with it on).
- [ ] Commit code, patches, notebook.

**Verify:** `pytest tests/test_convert.py -q` green; training run artifacts on Drive; leak assertion never fired; `PATCHES.md` lists every changed line of vendored code.

### Task S1.8 (optional, small) Text-initialized memory patch

Only if S1.7 shows static-feature flow is not enough (default: skip — DyGLib already concatenates static node features into every temporal embedding; init patch is a few lines per the 2 Jul source read).

- [ ] Patch memory init to copy the SPECTER2 vector (or its projection) into node memory at first appearance; document as Patch 2; A/B one pilot training run (same seed) init-on vs init-off; keep whichever wins on pilot AP, record the delta.

**Verify:** decision + delta in `PROGRESS.md`; `PATCHES.md` updated if adopted.

**Stage 1 exit criteria:** all unit tests green; pilot slice pulled, embedded, converted, and trained end-to-end in isolation; directed Forman module validated by hand-cases; directed ORC validated on Colab; `configs/slice.yaml` locked with rationale note; join rule decided and recorded.

---

## 6. Stage 2 — calibration

Book research stage 2. Sets every knob the integrated system needs. Everything here runs on pilot-scale data first; the expensive sweeps go to Colab CPU/high-RAM runtimes.

### Task S2.1 Curvature-at-time-t: the EventStore module

The biggest unlisted engineering item (research §4.4): no library computes curvature on "the graph as it existed at time t".

**Files:** `src/citecurv/curvature/timeslice.py`; `tests/test_timeslice.py`.

**Interfaces produced:**
- `EventStore(edges_df)` — holds events sorted by t once.
- `.subgraph_at(u, v, t, radius) -> nx.DiGraph` — the union of `radius`-hop directed neighborhoods of u and v using only edges with `ts ≤ t` (strictly `< t` for the event being scored itself — the scored edge is excluded from its own context; docstring states this).
- `.degrees_at(t) -> (in_deg_view, out_deg_view)` — incremental counters advanced monotonically, O(Δevents) per advance, for the Forman screen sweep.

**Steps:**
- [ ] Failing tests: a 10-event hand-built stream where the correct subgraph at each t is enumerated by hand; brute-force cross-check helper (filter edges ≤ t, take nx ego graphs, compare node/edge sets) run over every event of a 200-event random stream; monotonic-advance equality: `degrees_at(t)` equals brute-force degree count at 5 probe times.
- [ ] Implement with: edges array sorted by t + binary search (`np.searchsorted`) for time cuts; adjacency built incrementally for `degrees_at`; per-call BFS bounded by radius for `subgraph_at`.
- [ ] Benchmark on the pilot slice: mean `subgraph_at` time at radius 1 and 2; record. Commit.

**Verify:** pytest green including the brute-force sweep; benchmark recorded (this number × expected escalation count drives the S2.3 budget).

### Task S2.2 Forman screen sweep at event times

**Files:** `src/citecurv/curvature/screen.py` (new, small); cells in `notebooks/30_curvature.ipynb`.

**Steps:**
- [ ] For each event (u→v, t) in the pilot slice: `F = 2 − in_deg_at_t(u) − out_deg_at_t(v)` using `EventStore.degrees_at` (unweighted screen — weighted variant only if a weighting scheme enters later; YAGNI).
- [ ] Write `curvature_cache.parquet` rows (`forman` column, `orc` null).
- [ ] Distribution report by year and by concept: percentiles, tails. This is the raw material for threshold setting.

**Verify:** sweep covers 100% of pilot events; spot-check 20 random events against a brute-force recomputation; distribution report saved to Drive.

### Task S2.3 `[COLAB-RUN]` Screen threshold, radius, and ORC budget

**Steps:**
- [ ] Grid: screen threshold ∈ {extreme 1%, 2%, 5% two-sided percentiles from S2.2} × radius ∈ {1, 2}. For each cell: escalation rate (share of events flagged), ORC cost per flagged event (measured on a sample with S2.1 subgraphs + S1.5 wrapper), subgraph sizes vs the `orc_subgraph_edge_cap` (5,000-edge April knob — events whose subgraph exceeds the cap get the cap enforced by neighbor truncation via `nbr_topk`; count how often).
- [ ] Pick the cheapest cell whose escalation rate keeps full-slice ORC inside a single CPU/high-RAM Colab session per run (estimate with measured cost × expected full-slice event count; document arithmetic).
- [ ] Write chosen values into `configs/default.yaml` (`screen_percentile_low/high`, `orc_radius`); record the sweep table on Drive + summary in `PROGRESS.md`.

**Verify:** config values non-null; the arithmetic note exists and uses only measured numbers.

### Task S2.4 Cold-start k

**Steps:**
- [ ] On pilot scores-in-progress (use the S3 components once available — this task closes during stage 3): plot score stability vs number of prior events of the cited paper; pick the smallest k where component variance stabilizes; default stays k=3 until then (L19).
- [ ] Set `scoring.cold_start_k` in config; events below k are excluded from scoring and marked `cold_start` in outputs (they still update memory — exclusion is from *scoring*, not from *training*).

**Verify:** config updated; exclusion logic unit-tested (events 1..k-1 of a synthetic paper carry the flag, event k does not).

### Task S2.5 Per-event scoring latency (NFR2)

**Steps:**
- [ ] Measure end-to-end single-event scoring time (memory read + curvature-at-t + components) on pilot; record p50/p95.
- [ ] Write the measured budget into the book-hooks note (`docs/results_hooks.md`, created S4.9 — park the numbers in `PROGRESS.md` until then).

**Verify:** numbers recorded; no code change required (this task defines the budget NFR2 refers to).

**Stage 2 exit criteria:** EventStore tested and benchmarked; screen knobs + radius + cap locked in config from measured sweeps; cold-start rule implemented (k possibly still provisional); latency budget measured.

---

## 7. Stage 3 — integration pilot

Book research stage 3: all four components on one small slice, scoring event by event.

### Task S3.1 Component definitions note, then implementations

**Files:** `docs/score_components.md` (definition note); `src/citecurv/scoring/components.py`; `tests/test_components.py`.

The book names the four components (1.4) but does not fix formulas. The note proposes exact definitions, each with its leak argument; Maor reads it (one short review — this is the only in-stage human checkpoint); then implementation. Proposed defaults (to be confirmed in the note):

- `s_embed` — embedding surprise: negative log-probability of the observed event under the TGN link decoder at time t (the decoder DyGLib trains for link prediction, applied to (u, v, t) before the event updates memory).
- `s_geom` — geometric anomaly: |z-score| of the event's screen Forman value (and ORC where escalated) against the reference distribution for its (concept, year) cell from S2.2 — reference cells use only data with ts ≤ t (rolling or expanding window; note decides).
- `s_temp` — temporal anomaly: deviation of the citing gap pattern (time since v's previous citation, burst count in a trailing window) from the (concept, year) norm; exact statistic fixed in the note.
- `s_curvdyn` — curvature dynamics: residual between predicted and observed edge curvature, CurvGAD-style: a small MLP trained to predict the screen curvature of an event from the endpoint embeddings at t; large |predicted − observed| = geometry inconsistent with representation.

**Steps:**
- [ ] Write `docs/score_components.md` with formulas, windows, leak arguments, and the normalization scheme (per-component z against training-period distributions). `[HUMAN]` Maor reads and approves or amends (gate is cheap — one read).
- [ ] `ScoreBreakdown` dataclass: `s_embed, s_geom, s_temp, s_curvdyn, total, cold_start: bool, inputs: dict` (ids + values used — NFR10 traceability).
- [ ] Implement each component behind one function `score_event(...)`; unit tests per component on hand-built micro-fixtures (e.g., a burst of 10 citations in one day to one paper must raise `s_temp` above the same paper cited at natural spacing; an event whose Forman is at its cell median must have `s_geom ≈ 0`).
- [ ] Commit.

**Verify:** pytest green; note approved and referenced from the code docstrings.

### Task S3.2 Combiner — fixed weights, calibrated

**Files:** `src/citecurv/scoring/combiner.py`; `tests/test_combiner.py`.

**Steps:**
- [ ] `total = Σ w_c · z_c` with weights from config; z-normalization statistics computed on the training period only and frozen into the run artifact (never recomputed on test — leak).
- [ ] Calibration procedure (small grid over weight simplex, objective = average precision on the pilot validation slice labels); write chosen weights to the run config copy, not to `default.yaml` (weights are per-run artifacts).
- [ ] The learned-combiner variant (logistic head over the four z-scores) is implemented here too but only *used* in S4.6's extra ablation cell (L13).
- [ ] Unit test: with weights (1,0,0,0) total == z(s_embed); grid returns weights on the simplex.

**Verify:** pytest green; pilot calibration produces weights + AP number recorded.

### Task S3.3 Breakdown output, aggregation, export

**Files:** `src/citecurv/scoring/aggregate.py`; `tests/test_aggregate.py`.

**Steps:**
- [ ] `scores.parquet` writer per the section 3 contract (FR7: every score row carries its four named parts; NFR7).
- [ ] Per-paper and per-author rollups (FR8): paper score = top-quantile aggregate of its edge scores (exact aggregate fixed here: mean of top-5 edge scores, plus count above threshold — simple, defensible, documented); author rollup only if author ids survive the slice pull (OpenAlex authorships — check at S4.0; if dropped for size, mark FR8-author as descoped in `PROGRESS.md`).
- [ ] Export/filter helper (FR9, minimal by design): one function filtering `scores.parquet` by time window / concept / venue / score threshold → CSV. No UI — the book promises capability, not product.
- [ ] Unit tests on fixtures.

**Verify:** pytest green; a pilot export CSV opens in Excel with sane columns.

### Task S3.4 `[COLAB-RUN]` Integrated pilot run + sanity gates

**Steps:**
- [ ] Wire the full path in `notebooks/50_eval.ipynb` (pilot cells): trained pilot TGN → EventStore → screen/ORC cache → components → combiner → `scores.parquet`.
- [ ] Sanity gates, each PASS/FAIL in a `pilot_report.json`: (a) flag-rate by concept × year is not pathologically concentrated (no single cell holds >50% of top-1% flags — the D4 fairness sanity check); (b) reproducibility: full rerun same seed → identical `scores.parquet` (hash compare, NFR4); (c) cold-start exclusions match S2.4 rule; (d) top-20 flagged events eyeballed — each has a coherent breakdown (no component NaN/degenerate).
- [ ] Close S2.4 (finalize k) and S2.5 (latency) with this run's data if still open.

**Verify:** `pilot_report.json` all PASS; stage exit review appended to `PROGRESS.md`.

**Stage 3 exit criteria:** integrated scores exist for the pilot slice with named breakdowns; reproducibility gate passed; component definitions note approved; calibration weights recorded.

---

## 8. Stage 4 — full evaluation

Book research stage 4: the ablation table, baselines, injection track, manual review. Heavy compute; everything checkpoint-resumable; runs tagged `[COLAB-RUN]` unless noted.

### Task S4.0 `[COLAB-RUN]` Full-slice pull and preprocessing

- [ ] Run the S1 pipeline at `configs/slice.yaml` scale on a CPU runtime: pull (≈300K works, ≈3,000 pages, ≈$0.30–0.50, under the 1M/day cap — if the slice needs more results than the cap, split the pull across days by year-bucket; the harness already supports resumable cursor state), embed (GPU session), convert, screen sweep.
- [ ] Freeze the dataset: `dataset_manifest.json` (pull dates, filters, counts, SHA256 of parquet files) committed.

**Verify:** manifest committed; counts within 20% of S1.3 estimates (if not, investigate filter drift before any training).

### Task S4.1 Time split

**Files:** `src/citecurv/eval/split.py`; `tests/test_split.py`.

- [ ] Choose the cutoff date from data: latest date such that the test period (cutoff → end) contains ≥100 labeled positive papers under L12 *and* ≥15% of all events. Deterministic function of the frozen dataset + labels — unit-tested on a fixture.
- [ ] Freeze `configs/split.yaml` (cutoff, counts). Train/val split inside the pre-cutoff period (val = last 10% of train time — for calibration and early stopping).
- [ ] Label rule reminder: a positive test citation = citation to or from a paper retracted under L12 (book 4.1.2); retraction *date* may postdate the cutoff — allowed, labels are outcome data, not features (leak argument documented in split.py docstring).

**Verify:** pytest green; split counts recorded; no event appears in two splits.

### Task S4.2 `[COLAB-RUN]` Main training runs

- [ ] Full model (curvature features + all four components), seeds 0–4, on the frozen split. Checkpoint every epoch to Drive; resume logic tested by killing one run mid-epoch and resuming (do this once deliberately).
- [ ] Each run writes `runs/full_<seed>/metrics.json` + resolved config + git commit hash.

**Verify:** 5/5 runs complete; seed spread of val AP recorded; resume test passed.

### Task S4.3 `[COLAB-RUN]` Ablation matrix

Variants (book 4.2.1): `full`, `no_curvature` (components s_geom, s_curvdyn off AND curvature features absent from head), `forman_only` (no ORC escalation), `ollivier_only` (screen replaced by random escalation at matched rate — the screen must not leak Forman signal into the "no Forman" cell; this design choice is the fair reading of "only Ollivier" and is documented in `docs/score_components.md`), `no_temporal` (static GNN encoding: DyGLib run with memory disabled / time encoding ablated — use DyGLib's own static-ish config nearest to this and document exactly what "temporal off" means).

- [ ] 5 variants × 5 seeds = 25 runs, driven by `eval/ablation.py` generating per-run configs; plus the timestamp-shuffle sanity run (test set with shuffled timestamps through the full model — scores must degrade toward chance; book 4.2.1).
- [ ] Chunk to fit session limits under G1 default (Pro): estimated from S4.2 wall-clock; if a single variant-seed run exceeds a session, escalate G1 (ask for Pro+) in `PROGRESS.md`.

**Verify:** 25 `metrics.json` + shuffle run present; ablation table renders (S4.6).

### Task S4.4 `[COLAB-RUN]` Baselines

**Files:** `src/citecurv/eval/baselines/`; sealed-env notes in `baselines/curvgad_notes.md`.

Fair-comparison protocol (book 3.3): static detectors see the graph snapshot at the cutoff date; they score the same test edges; node-scoring detectors map to edges by `edge_score = max(score(u), score(v))` (documented; the simple defensible rule).

- [ ] Tier 1 — statistical: self-citation rate, citing-journal concentration (Gini), citation burst z-score, reference-list overlap — CATCHSYNC-inspired, ~4 features, each a scoring rule alone and a combined rank-average. Pure pandas; unit-tested.
- [ ] Tier 2 — classical ML: XGBoost on the tier-1 features + degrees + SPECTER2 cosine of endpoints (leak-checked: all features computable at t). Python 3.12 → current xgboost fine.
- [ ] Tier 3 — PyGOD: DOMINANT, CoLA, GADNR on the static snapshot (PyG 2.6 pin per pins.md).
- [ ] CurvGAD: clone `karish-grover/curvgad` into its own Colab env (its requirements mix torch+PyG+DGL+geoopt+GraphRicciCurvature+pygod+catboost — never share an env), run on the same snapshot, export scores; treat as sealed artifact (MIT, cite ICML 2025 / PMLR v267).
- [ ] De-curvatured own TGN: this is the `no_curvature` ablation cell reused as the "temporal detector without curvature" baseline (audit fix 3 wording: the TGN backbone adapted for event scoring).

**Verify:** every baseline produces a `scores.parquet`-compatible file on the same test events; a joint table lists them all with the same metric code path (no per-baseline metric reimplementation).

### Task S4.5 Synthetic cartel injection track

**Files:** `src/citecurv/eval/inject.py`; `tests/test_inject.py`.

- [ ] `inject_cartel(events, n_cartels, size_range=(8, 25), reciprocity, internal_density, window_days, seed)`: pick member papers from a held-out region (no RWDB label, one concept), add timestamped citation events among members following the CIDRE donor/recipient reciprocity pattern (a share of members mostly *receive*, a share mostly *donate*, edges concentrated inside the group), timestamps drawn inside each cartel's window. Returns modified events + ground-truth event ids.
- [ ] Obviousness ladder (book 3.3): three parameter levels (blatant / moderate / subtle — density and reciprocity stepped down); recall@k of the full model per level; also run the PyGOD clique injector (`gen_structural_outlier(directed=True)`) as the crude comparison variant.
- [ ] Unit tests: injected events all inside the window and the member set; truth ids match; reproducible under seed.

**Verify:** pytest green; recall-vs-obviousness table produced; injected region excluded from training (assert: all injected events post-cutoff).

### Task S4.6 Metrics and results tables

**Files:** `src/citecurv/eval/metrics.py`; `tests/test_metrics.py`.

- [ ] `ranking_metrics(scores, labels)`: AUROC and average precision (the book's "ranking across all thresholds" measurement), precision@100 and @1000 (book (ii)), plus positives-found@k. One code path for own model and all baselines.
- [ ] Tables (CSV + markdown for the book): main comparison (full model vs 5 baselines, mean ± spread over seeds where applicable); ablation table (5 variants, curvature-removal delta highlighted — book success test (iii)); learned-combiner extra cell (L13) vs fixed weights; label-sensitivity run — re-join scores against the broadened family set from S0.6's recorded vocabulary and recompute the main table once (L12); shuffle-test line.
- [ ] Unit tests on toy score/label arrays with known AUROC/AP.

**Verify:** pytest green; every number in every table regenerates from `runs/` artifacts by one script (`eval/ablation.py --tables`) — no hand-edited numbers anywhere.

### Task S4.7 `[HUMAN]` Top-50 manual review

- [ ] Claude prepares the review packet: top 50 flagged test citations (full model, median seed), each with breakdown, endpoint titles/abstracts, links, and the category checklist from book 4.2.1 (cartel / forced self-citation / sneaked reference / paper-mill / other-suspicious / nothing).
- [ ] Maor + Gilboa fill the log (split 25/25 or double-code 10 for agreement — their call, recorded).
- [ ] Claude ingests the log → review summary table.

**Verify:** 50/50 rows filled; summary table in results pack.

### Task S4.8 Success-criteria checkpoint

The book's own bar (4.1.1–4.1.2), evaluated honestly:

- [ ] Top-of-list quality of the full model clearly above CurvGAD *and* above the de-curvatured TGN (book success definition).
- [ ] Curvature-removal delta is measurable and outside seed noise (book (iii): if the drop is small, the architectural claim failed its own test — report it as such; this outcome is a valid research result, not a plan failure).
- [ ] Shuffle test degrades scores; reproducibility holds.
- [ ] Write the honest verdict into `PROGRESS.md` + results pack.

**Verify:** each bullet has a number and a pass/fail/nuanced-call next to it.

### Task S4.9 Results pack and book hooks

- [ ] `docs/results_hooks.md`: every table/figure the Phase B book needs, with file pointers — main table, ablation table, injection table, review summary, latency (S2.5), dataset manifest numbers, slice rationale. English-only, APA-ready phrasing, no AI-tells.
- [ ] Update the NotebookLM project notebook (ba77abc5) with the results pack as a source; note the deck slides that need refreshed numbers (deck lives in Claude design — provide paste-ready edit prompts, per the established workflow).
- [ ] `PROVISIONAL(G3/G4)` markers resolved or carried forward explicitly.

**Verify:** every hook points at an existing artifact; nothing cites a number that `--tables` doesn't regenerate.

**Stage 4 exit criteria:** ablation + baseline + injection + sensitivity tables regenerate from artifacts; manual review ingested; success-criteria checkpoint written with honest numbers; results pack complete.

---

## 9. Cross-cutting rules

### 9.1 Colab discipline

- Five per-stage notebooks (section 3); every notebook's first cell: `pip install "citecurv @ git+https://github.com/Natboa/...@<tag>"` + the stage's pins — notebooks contain no logic beyond driving `src/`.
- Persist to Drive as few large files (npz/parquet); never folders with >10K items (Drive mount fails past that).
- Checkpoint-resume is mandatory for anything >30 min (sessions die on idle, limits, unit exhaustion).
- Runtime choice: CPU/high-RAM for curvature sweeps (cheaper units), GPU (T4 default ≈1.8 units/h; A100 ≈15 units/h, contended on plain Pro) for embedding, TGN, GNN baselines.
- Keys in Colab Secrets (`OPENALEX_API_KEY`), read via `google.colab.userdata`.
- Record unit burn per heavy run in `PROGRESS.md` (feeds the G1 Pro+ decision).

### 9.2 WSL2 fallback

Triggers: Colab funding lapses; Block A smoke FAIL; persistent session kills making S4.2/S4.3 infeasible. Fallback = the five-venv blueprint (research §6) on WSL2 Ubuntu + CUDA passthrough: `data`/`embed`/`curv`/`tgn`/`baseline` venvs with the same pins. Native Windows stays for writing, notebooks-as-files, API pulls, and the pure-Python modules (forman_directed, timeslice, doi, abstracts run anywhere).

### 9.3 Version pins (copy into `configs/pins.md` at S0.3)

| Stage env | Pins |
|-----------|------|
| data | python 3.12 · pyalex 0.21 · pandas · pyarrow · requests |
| embed | adapters 1.3.0 (pins transformers ~4.57.x) · torch 2.x CUDA |
| curv | GraphRicciCurvature 0.5.3.2 · POT 0.9.6 · networkit 11.2.1 · scipy ≤1.13.1 · networkx |
| tgn | vendored DyGLib · torch 2.x CUDA (pure PyTorch — no PyG) |
| baseline | pygod 1.1.0 · torch_geometric 2.5–2.6 · torch 2.x · xgboost (3.3+ on py3.12) · scikit-learn 1.9 |
| curvgad (sealed) | whatever `karish-grover/curvgad` requirements.txt says, in its own env, never merged |

### 9.4 Testing policy

- Unit: curvature (hand-computed cases — the book's 4.2.2(i) requirement), DOI/abstract utilities, converters, components, metrics, injector.
- Integration: full pipeline on a committed synthetic mini-graph fixture (~200 events) asserting output schema + value ranges (book 4.2.2(ii)) — runs in CI-less local pytest, no network, no GPU.
- Reproducibility: seeded double-run hash equality (book 4.2.2(iii)) — pilot (S3.4) and one full-scale spot check (S4.2).
- Leak: loader assertion (S1.7) + shuffle test (S4.3) + per-feature leak arguments in `docs/score_components.md`.

### 9.5 Writing hooks

Anything destined for the book/advisors follows the project writing rules (English-only, APA, anti-AI-tells, no "Part/Phase" framing inside book text). Results tables come only from `--tables` regeneration. Claims about tools/versions cite `configs/pins.md` or the dataset manifest.

## 10. Risk register

| Risk | Trigger sign | Mitigation | Fallback |
|------|--------------|-----------|----------|
| Colab session limits vs training length | S4.2 run can't finish in one session | Checkpoint-resume (mandatory anyway); chunk ablation matrix | Escalate G1 → Pro+ (24 h background); WSL2 local |
| 768-dim memory footprint | OOM in S1.7/S4.2 (3 GB per 1M nodes rule of thumb) | Slice is ≈300K nodes — headroom | `memory_projection_dim` knob: project SPECTER2 → 256 at init |
| torch 2.x breaks dormant DyGLib | Smoke Block B FAIL | Pin torch lower per DyGLib requirements (`>=1.8.1`) | Patch vendored code (documented in PATCHES.md) |
| ORC cost blowup | S2.3 arithmetic exceeds a session | Raise screen threshold; radius 1; enforce edge cap via nbr_topk | Sample flagged edges (report as approximation, honestly) |
| Label sparsity in test period | S4.1 can't find a cutoff with ≥100 positives | Widen slice years; loosen to broadened family (recorded as such) | Lean on injection track; report label-limited evaluation |
| adapters/transformers pin conflicts | embed env breaks another stage | Per-stage notebooks/venvs — never merge envs | SciNCL or SPECTER1 plain-stack fallback (research §4.2) |
| GraphRicciCurvature fork-only multiprocessing | Block A hang (should not happen on Colab) | Colab/WSL2 are fork-safe by construction | `proc=1` single-process for small flagged sets |
| Drive folder item limits / IO | Mount errors, silent write fails | Few-large-files contract (section 3); manifest hashes | Repo-committed small artifacts; local disk on WSL2 |
| OneDrive vs git contention | Lock/sync errors in working folder | Data ignored by git; small text only | Move clone to `C:\dev\`, leave pointer |
| CurvGAD env poisons others | Dependency solver fights | Sealed env, sealed notebook | Report CurvGAD from its published citation-dataset numbers only, marked as such |
| Advisor reverses label rule (G3) | Sign-off changes families | Scores stored label-free; re-join + S4.6 rerun is cheap | — |
| API cap / cost | Pull needs >1M results/day | Year-bucket the pull across days; resumable cursors | Trim slice in S1.3 criteria |

## 11. Progress tracking protocol

`PROGRESS.md` holds: task status table (todo / in-progress / done / blocked, with date + verify-output note), the pending-decisions ledger (section 2 live copy), the provisional-outputs list, unit-burn log, and an append-only session log (`YYYY-MM-DD — did X, verified Y, next Z`). Rules: no task marked done without its Verify line executed; every session ends with a `PROGRESS.md` update; once the repo exists, `PROGRESS.md` updates are part of each task's commit.

## Appendix A — source documents

- Phase A book: `Detecting Network Citation Anomalies Using Graph Curvature_book_project.docx` (folder root, 23/6 version) — architecture, FR/NFR, research stages, evaluation protocol.
- `Summeries from claude/Component research findings 2026-07-02.docx` — live verification of every component, D-items, environment blueprint, Colab addendum.
- `Summeries from claude/Book audit findings 2026-07-02.html` — 3 must-fix errors, small fixes, advisor questions Q1/Q2.
- Project memory: transformer decision (Option 1), retraction dataset findings (2026-06-17), writing-style rules.

## Appendix B — D-item dispositions (from the 2 Jul research, §5)

| Item | Disposition |
|------|-------------|
| D1 score components/weights | 4 components; staged fixed→learned (L13, S3.2, S4.6) |
| D2 temporal resolution | Continuous-time; slicing is first-class (L14, S2.1) |
| D3 explainability | Named breakdown only; PyG explainers future work (L15) |
| D4 ethics | Flag-rate-by-field/year sanity gate (S3.4a) |
| D5 cold start | k set in calibration; provisional k=3 (L19, S2.4) |
| D6 computational scoping | Screen/radius/cap knobs from measured sweeps (S2.3) |
| D7 baselines | Three tiers + CurvGAD + de-curvatured TGN (L17, S4.4) |
| D8 extra papers | CIDRE kept (injector design), CATCHSYNC kept (tier-1 features); rest dropped |
| D9 OpenAlex access | Key + cost arithmetic + cap handling (S0.4, S1.1, S4.0) |
| D10 client | pyalex + custom backoff (L8, S1.1) |
| D11 data filters | Retraction-dense slice via label-density query (L11, S1.3) |
| D13 label rule | Referencing/Attributions primary + sensitivity (L12, S0.6, S4.6) |
| D14 injection | CIDRE-pattern custom injector + PyGOD clique variant (L18, S4.5) |

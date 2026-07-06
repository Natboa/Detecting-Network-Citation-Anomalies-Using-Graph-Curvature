# Progress — Phase B implementation

Companion to `IMPLEMENTATION_PLAN.md`. Session protocol: read this file first, pick the next unblocked task, execute, run its Verify line, update here before ending the session. No task is `done` without its Verify output recorded.

Status values: `todo` · `in-progress` · `blocked(reason)` · `done(date)` · `skipped(reason)`.

## Task status

| Task | Tag | Status | Verify output / note |
|------|-----|--------|----------------------|
| S0.1 Create GitHub repo | HUMAN | done(2026-07-06) | Gilboa made it public (existed private since 2026-03-21 — explains the July-2 404). Anonymous API 200, `main` default; `Maores` has `push: true`. Was empty — README/LICENSE created in S0.2. |
| S0.2 Working folder → clone | | done(2026-07-06) | `git init -b main`, remote wired, explicit adds only (legacy files left untracked). First push `b8ecf30`; `.claude/`, `.impeccable/`, data all gitignored. |
| S0.3 Package skeleton | | done(2026-07-06) | `.venv` Python 3.12.8; `pip install -e .[dev]` clean; pyalex exactly 0.21; smoke suite 2 passed. Commit `a8dfc73`. |
| S0.4 OpenAlex key + cost meter | HUMAN part | todo | **Next human step (~5 min):** register at openalex.org/settings/api, set env `OPENALEX_API_KEY`, give me the mailto address for `configs/default.yaml`. Code side (env wiring + refuse-keyless guard) already done and tested in S1.1. |
| S0.5 Smoke notebook (4 gates) | COLAB-RUN | todo | Next Claude task: write `notebooks/00_smoke.ipynb`; then Maor runs it on Colab (free tier fine). |
| S0.6 RWDB snapshot freeze + counts | | done(2026-07-06) | Snapshot 2026-07-06: 70,771 records; Ref/Attr family 14,896 records / 14,842 unique DOIs (exact literal: `Concerns/Issues about Referencing/Attributions`); broadened candidates recorded (Paper Mill 11,796; Concerns/Issues about Peer Review 11,440; Compromised Peer Review 11,384; Taken via Peer Review 23) — no "Fake Peer Review" or citation-manipulation literal exists. OriginalPaperDOI non-null 96.17%. SHA256 in `data/labels/rwdb_counts.json` (committed); CSV local-only. 4 tests. Commit `dcd5b98`. |
| S0.7 Book carry-over fixes | | todo | parallel track, never blocks |
| S1.1 Data pull harness | | in-progress | Code + 5 mocked tests done (pagination, 429 backoff, non-retryable raise, cost ledger, parquet contracts, keyless-refusal guard). Commit `aff6b63`. **Remaining:** live 2-page pull verify — blocked(S0.4 key). |
| S1.2 Pilot pull + label join | | todo | blocked(S0.4 key) |
| S1.3 Slice selection → slice.yaml | | todo | PROVISIONAL(G3) on rationale; needs S1.2 |
| S1.4 Directed Forman module | | done(2026-07-06) | 3 tests green (6 hand cases, full-sweep consistency, weighted-unit-reduction over 3 seeded random graphs). Benchmark: 30,147 edges in 0.060 s (bar <5 s). Saucan-paper figure cases not attempted (paywalled source); covered by hand cases + S1.5 ORC cross-check. Commit `d37c785`. |
| S1.5 Directed ORC validation | COLAB-RUN | todo | toy expectations ready in tests/test_forman_directed.py CASES |
| S1.6 SPECTER2 encoding pilot | COLAB-RUN | todo | |
| S1.7 DyGLib integration (pilot train) | COLAB-RUN | todo | |
| S1.8 Text-init memory patch | optional | todo | default: skip unless S1.7 argues |
| S2.1 EventStore (curvature-at-t) | | done(2026-07-06) | 7 tests green incl. 200-event × 2-radius brute-force sweep and scored-instance-exclusion cases. Benchmarks (synthetic 100k events / 5k nodes, denser than real citation data): subgraph_at 0.17 ms r=1 / 10.15 ms r=2 per event; build 92 ms; full degrees advance 54 ms. Commit `c1ed353`. |
| S2.2 Forman screen sweep | | todo | needs pilot data (S1.2) |
| S2.3 Threshold/radius/cap sweep | COLAB-RUN | todo | |
| S2.4 Cold-start k | | todo | provisional k=3 active |
| S2.5 Latency measurement (NFR2) | | todo | |
| S3.1 Component definitions + impl | HUMAN review of note | todo | |
| S3.2 Combiner (fixed weights) | | todo | |
| S3.3 Breakdown/aggregation/export | | todo | |
| S3.4 Integrated pilot + sanity gates | COLAB-RUN | todo | |
| S4.0 Full-slice pull + preprocess | COLAB-RUN | todo | |
| S4.1 Time split | | todo | |
| S4.2 Main training runs (5 seeds) | COLAB-RUN | todo | |
| S4.3 Ablation matrix (25 + shuffle) | COLAB-RUN | todo | |
| S4.4 Baselines (3 tiers + CurvGAD) | COLAB-RUN | todo | |
| S4.5 Cartel injection track | | todo | |
| S4.6 Metrics + results tables | | todo | |
| S4.7 Top-50 manual review | HUMAN | todo | |
| S4.8 Success-criteria checkpoint | | todo | |
| S4.9 Results pack + book hooks | | todo | |

## Pending decisions ledger

| Gate | Question | Owner | Default in force | Status |
|------|----------|-------|------------------|--------|
| G1 | Colab Pro vs Pro+ | Maor → college | Pro (chunked runs) | open — Maor considering Pro+ (~162 ₪/mo); buying it closes this gate |
| G2 | Notebooks: deliverable or venue only | Maor → advisors | venue only | open |
| G3 | Label-rule sign-off (Referencing/Attributions) | Maor → advisors | Ref/Attr family — exact literal now known: `Concerns/Issues about Referencing/Attributions` | open |
| G4 | Audit Q1 — solution-description section in book | Maor → advisors | assume required later | open |
| G5 | Audit Q2 — page-limit accounting | Maor → advisors | book track only | open |
| G6 | Repo visibility timing | Maor + Gilboa | public at creation | resolved(2026-07-06): public — note the plan and progress files are world-readable |

## Provisional outputs

| Artifact | Provisional under | Cleared when |
|----------|-------------------|--------------|
| (none yet) | | |

## Unit-burn log (Colab)

| Date | Notebook / run | Runtime | Units (approx) |
|------|----------------|---------|----------------|
| (none yet) | | | |

## Session log

- 2026-07-06 (execution session) — Go received; executed S0.2, S0.3, S0.6, S1.1 (code), S1.4, S2.1 with TDD (red observed before every implementation). Suite: **25 tests passed**. Six commits pushed to `main`. One data-reality correction: RWDB has no "Fake Peer Review" literal — real peer-review literals recorded in rwdb_counts.json; broadened-family candidates now come from the live vocabulary. **Next:** (human, ~5 min) S0.4 OpenAlex key + mailto → unblocks S1.1-live + S1.2; (Claude) S0.5 smoke notebook, then S1.2/S1.3 once the key lands. S0.7 book fixes available any session.
- 2026-07-06 (later) — S0.1 done: repo public and push access confirmed from this machine (`Maores` collaborator). Book audit's 404-link item resolved by the same act. S0.2 was the next Claude task, awaiting go (first push = public).
- 2026-07-06 — Plan created and approved (guided flow, two decision rounds with Maor: Colab primary/WSL2 fallback; retraction-dense slice; Ref/Attr labels + sensitivity; staged fixed→learned weights; proceed-with-defaults gate policy; deliverable = this md pair + docx snapshot). Next: S0.1 (Maor creates the repo), then S0.2–S0.3 in the next Claude session.

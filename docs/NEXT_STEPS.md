# Next steps — Phase B from 2026-11-01

Written at the end of the 2026-10-31 cloud session that implemented the offline part of the plan
(see `PROGRESS.md`, session log of that date). This page is the order of work from here; the task
details stay in `IMPLEMENTATION_PLAN.md`.

## Where the project stands (one paragraph)

Every module of the pipeline now exists and is unit-tested (data harness, labels and slice
selection, SPECTER2 encoder, Forman screen and directed ORC, time-sliced graph store, vendored
DyGLib TGN with converter/train/inference, the four score components, combiner, rollups and export,
time split, cartel injector, three baseline tiers, metrics, ablation driver, interactive results
explorer) and runs end to end on synthetic data. Nothing has touched real OpenAlex data yet: the
API key (S0.4) is the single blocker for Stage 1. The session also produced one result that
changes the research plan (section 3).

## 1. First week — unblock (human steps)

| # | Step | Who | Unblocks |
|---|------|-----|----------|
| 1 | Register the OpenAlex key at `openalex.org/settings/api-key`; put it in Colab Secrets as `OPENALEX_API_KEY`; run `check_key()` once (`docs/colab_runbook.md` §9) | Maor | S1.1-live, S1.2, S1.3 |
| 2 | Run `notebooks/00_smoke.ipynb` on Colab (Python 3.12 runtime, T4) | Maor | S0.5 gate on the real venue |
| 3 | Advisor meeting — bring section 2 below | Maor + Gilboa | G3, the curvature question |
| 4 | Agree the split of work with Gilboa (proposal in section 5) | Maor + Gilboa | parallel progress |
| 5 | Review and merge the session branch `claude/amazing-archimedes-1tu9ny` | Maor | everything below |
| 6 | Optional, for future cloud sessions: allow `api.openalex.org` and `huggingface.co` in the cloud environment's network settings and install the Claude GitHub App on the repo | Maor / Gilboa | cloud sessions can pull data and push |

## 2. Decisions waiting for Maor / the advisors

Ordered by impact. Each has a recorded default, so work does not stop while it is open.

1. **The curvature question (new, highest impact).** At event time the plan's directed Forman
   screen equals `2 − refs(v)` and the plan's directed ORC equals `−(1−α)(1 − coupling share)`
   (proofs and checks in `docs/curvature_formulation_study.md`). On synthetic cartels no tested
   curvature formulation added detection power beyond simple counting controls. Options are listed
   in that study and appended to `docs/score_components.md`; the choice decides how the book frames
   its central claim. Default until decided: keep the plan's formulation, add the counting controls
   as a named baseline, and re-test on the real slice.
2. **S3.1 score-component definitions** (`docs/score_components.md`, D1–D12, PROVISIONAL).
3. **G3 label rule:** Ref/Attr family, Retraction-only notices, strict DOI join; adopt the
   2026-10-31 RWDB snapshot or keep July's (`docs/slice_shortlist.md`).
4. **Slice:** pick from the shortlist once the OpenAlex counts are measured (computing / cell-mol-bio
   / env-energy; slices end in 2023).
5. **TGN direction patch P5** (`third_party/DyGLib/PATCHES.md`): keep the citation direction (default on).
6. **GraphRicciCurvature pin:** 0.6.1 recommended (`docs/orc_version_comparison.md`).
7. **Book fixes:** adopt `docs/book/..._S0.7.docx` after checking it against any locally fixed copy;
   pick the typo candidates; check the Ekle & Eberle DOI once.
8. **Evaluation hygiene additions:** report metrics with the largest retraction batches held out
   (77.5 % of positives sit in 32 batches) and on test citations whose cited paper never appears in
   training (label-propagation risk).

## 3. What the synthetic work says about the research claim

- The machinery works end to end, is deterministic, and passes its leak checks.
- The planted cartels are easy to find by simple statistics (cited-paper age, reference overlap);
  any real-data claim must beat those statistics, not only CurvGAD and the de-curvatured TGN.
- The plan's curvature signals reduce exactly to reference-list counts at event time. Whether the
  book keeps them, replaces them, or reports the reduction as a finding is decision 1 in section 2.
- See `docs/rehearsal_report.md` for the end-to-end numbers on synthetic data.

## 4. Order of work after the key arrives

1. **Stage 1 on real data:** S1.1-live verify → S1.2 pilot pull + label join (`10_data`, RUN=pilot)
   → S1.3 slice measurement and `configs/slice.yaml` → S1.6 SPECTER2 on the pilot (`20_embed`, GPU)
   → S1.7 pilot TGN (`40_tgn`, GPU).
2. **Stage 2 calibration:** S2.2 screen sweep on the pilot (`30_curvature`) → S2.3 grid with
   `escalation_budget` → S2.4 cold-start k → S2.5 latency (components measured at 0.42 ms p50).
3. **Stage 3 integration pilot:** S3.1 approval → S3.2 calibration → S3.4 pilot run with the four
   sanity gates (`50_eval`) → the interactive explorer built from the pilot run.
4. **Stage 4 full evaluation:** S4.0 full pull (resumable) → S4.1 split → S4.2 five seeds → S4.3
   ablations → S4.4 baselines (CurvGAD "as released", sealed env) → S4.5 injection ladder → S4.6
   tables → S4.7 top-50 manual review → S4.8 verdict → S4.9 results pack.
5. **Book:** Phase B chapters from the results pack; the explorer screenshots serve as figures for
   the explainability requirement (FR7).

## 5. Suggested split with Gilboa

- **Track A — data and representation:** OpenAlex pulls, labels, slice, SPECTER2, TGN training and
  inference (notebooks 10, 20, 40).
- **Track B — geometry and evaluation:** curvature screen/ORC, components, combiner, baselines,
  injection, metrics and tables (notebooks 30, 50).
- **Together:** the curvature decision, the top-50 manual review (S4.7), the book.

## 6. Known risks carried forward

- Colab defaults to Python 3.13; the curvature stack needs a 3.12 runtime (or the documented
  `--no-deps` path).
- DyGLib is unmaintained and has no mid-run resume; per-event TGN outputs reach about 18 GB at 3M
  events in float32 (float16 halves it).
- PyGOD 1.1.0 needs four runtime fixes; DOMINANT and CurvGAD build dense N×N matrices (node cap
  decision).
- CurvGAD as released differs from its paper; it was not executed in this session.
- Label concentration: 57.7 % Hindawi, 77.5 % in 32 mass-retraction batches.

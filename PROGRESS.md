# Progress — Phase B implementation

Companion to `IMPLEMENTATION_PLAN.md`. Session protocol: read this file first, pick the next unblocked task, execute, run its Verify line, update here before ending the session. No task is `done` without its Verify output recorded.

Status values: `todo` · `in-progress` · `blocked(reason)` · `done(date)` · `skipped(reason)`.

## Task status

| Task | Tag | Status | Verify output / note |
|------|-----|--------|----------------------|
| S0.1 Create GitHub repo | HUMAN | done(2026-07-06) | Gilboa made it public (existed private since 2026-03-21 — explains the July-2 404). Verified: anonymous API 200, `visibility: public`, `default_branch: main`; `Maores` account on this machine has `push: true`. Repo is EMPTY (no branches/README/LICENSE) — those fold into S0.2. |
| S0.2 Working folder → clone | | todo | ready — repo empty, so also create README + MIT LICENSE here; first push publishes plan+progress (repo is public) |
| S0.3 Package skeleton | | todo | blocked by S0.2 |
| S0.4 OpenAlex key + cost meter | HUMAN part | todo | key registration is the human part |
| S0.5 Smoke notebook (4 gates) | COLAB-RUN | todo | Claude writes it after S0.3 |
| S0.6 RWDB snapshot freeze + counts | | todo | |
| S0.7 Book carry-over fixes | | todo | parallel track, never blocks |
| S1.1 Data pull harness | | todo | |
| S1.2 Pilot pull + label join | | todo | |
| S1.3 Slice selection → slice.yaml | | todo | PROVISIONAL(G3) on rationale |
| S1.4 Directed Forman module | | todo | |
| S1.5 Directed ORC validation | COLAB-RUN | todo | |
| S1.6 SPECTER2 encoding pilot | COLAB-RUN | todo | |
| S1.7 DyGLib integration (pilot train) | COLAB-RUN | todo | |
| S1.8 Text-init memory patch | optional | todo | default: skip unless S1.7 argues |
| S2.1 EventStore (curvature-at-t) | | todo | |
| S2.2 Forman screen sweep | | todo | |
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
| G1 | Colab Pro vs Pro+ | Maor → college | Pro (chunked runs) | open |
| G2 | Notebooks: deliverable or venue only | Maor → advisors | venue only | open |
| G3 | Label-rule sign-off (Referencing/Attributions) | Maor → advisors | Ref/Attr family | open |
| G4 | Audit Q1 — solution-description section in book | Maor → advisors | assume required later | open |
| G5 | Audit Q2 — page-limit accounting | Maor → advisors | book track only | open |
| G6 | Repo visibility timing | Maor + Gilboa | public at creation | open |

## Provisional outputs

| Artifact | Provisional under | Cleared when |
|----------|-------------------|--------------|
| (none yet) | | |

## Unit-burn log (Colab)

| Date | Notebook / run | Runtime | Units (approx) |
|------|----------------|---------|----------------|
| (none yet) | | | |

## Session log

- 2026-07-06 (later) — S0.1 done: repo public and push access confirmed from this machine (`Maores` collaborator). Book audit's 404-link item resolved by the same act. S0.2 is the next Claude task, awaiting go (first push = public).
- 2026-07-06 — Plan created and approved (guided flow, two decision rounds with Maor: Colab primary/WSL2 fallback; retraction-dense slice; Ref/Attr labels + sensitivity; staged fixed→learned weights; proceed-with-defaults gate policy; deliverable = this md pair + docx snapshot). Next: S0.1 (Maor creates the repo), then S0.2–S0.3 in the next Claude session.

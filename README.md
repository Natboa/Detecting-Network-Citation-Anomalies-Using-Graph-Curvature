# Detecting Network Citation Anomalies Using Graph Curvature

Capstone project 26-2-D-22, Software Engineering Department, Braude College of Engineering (research track).

The system detects anomalous citations in directed temporal citation networks by combining two signals: a temporal graph neural network (TGN via DyGLib, with SPECTER2-initialized node memory and bidirectional messaging) that learns what normal citation behavior looks like over time, and directed discrete Ricci curvature (a custom Forman-Ricci screen implemented from Saucan et al. 2019, with Ollivier-Ricci escalation via GraphRicciCurvature) that describes the local geometry of each citation edge. A citation is flagged when both views agree, and every score decomposes into named components a human reviewer can inspect.

**Authors:** Maor Saban, Gilboa Natan · **Advisors:** Dr. Renata Avros, Prof. Zeev Volkovich

## Repository map

- `IMPLEMENTATION_PLAN.md` — the executable Phase B plan (stages 0–4, task by task)
- `PROGRESS.md` — live task status, decision ledgers, session log
- `src/citecurv/` — the pipeline package (data, embedding, curvature, TGN adapter, scoring, evaluation)
- `configs/` — runtime configuration and version pins
- `notebooks/` — thin Colab drivers for the compute-heavy stages
- `tests/` — pytest suite

## Data sources

- [OpenAlex](https://openalex.org/) (CC0) — citation graph and paper metadata
- Retraction Watch database, publicly available via Crossref — weak positive labels

Code is MIT-licensed. Data artifacts are not stored in this repository.

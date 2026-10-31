# Curvature formulation study: what does curvature add beyond trivial controls?

> **SYNTHETIC DATA.** Every number in this note comes from the citecurv synthetic citation stream with planted
> cartels (`citecurv.data.synthetic`, `citecurv.eval.inject`). None is a measurement on real citations. The
> generator's defaults are stand-in values, not calibrated on the real slice.

Status: PROVISIONAL, a decision for Maor and the advisors (no gate in PROGRESS.md covers it yet). Written
2026-10-31 by agent W2c. Numbers: `results/curvature_study.json`. Code: `scripts/curvature_study.py` (study),
`src/citecurv/curvature/variants.py` (signal definitions), `tests/test_curvature_variants.py`.

## 1. Question and short answer

The book's central claim is that directed Ricci curvature adds detection power beyond a temporal model. Two
wave-1 agents found that, scored at event time t (graph state with ts <= t, NFR3), the plan's two curvature
signals collapse to trivial statistics, and that cited-paper age alone separates planted cartel citations well.
This study measures, on synthetic data, which curvature formulations carry anomaly signal beyond trivial controls.

- **The plan's two event-time curvature signals are exactly trivial statistics.** The Forman screen equals
  2 - (reference count of the cited paper) and the directed Ollivier-Ricci curvature is a strictly increasing
  function of the bibliographic-coupling share (Propositions 1 and 2; Spearman -1.000 and +1.000 on every run).
  Added to the controls they change logistic-regression AP by at most 0.003 in every setting.
- **Cited-paper recency dominates** this synthetic benchmark: age alone has AUROC 0.965-0.991, and a logistic
  regression on the tier-A controls reaches AP 0.97 (blatant), 0.52 (moderate), 0.44 (subtle) and 0.90 (clique)
  on the test sample.
- **With logistic regression, no curvature formulation adds more than 0.008 AP beyond the full control set
  (tier B)** in any setting, and none moves AUROC by more than 0.001 there (gradient boosting: at most +0.025 ±
  0.015, where it is the weaker learner). Against the narrower tier A, triangle-based formulations
  (augmented Forman S4b, undirected ORC S4a, reversed ORC S3a) add AP on the clique variant (+0.043 ± 0.021,
  +0.028 ± 0.006, +0.014 ± 0.001) and a little for moderate cartels (S4b +0.010 ± 0.005); these gains vanish once
  one non-curvature count, |refs(u) & citers_t(v)| ("u cites a paper that already cites v"), and the coupling
  share are added.
- Several formulations are strong single signals (co-citation Forman S5b, AUROC 0.88-0.96), but mostly as
  re-expressions of the cited paper's citation count (Spearman -0.97 for S5b), which is itself a recency proxy.
- **Audit mode** (curvature at t + 1 or 2 years, which violates NFR3 by design) adds nothing beyond simple audit
  counts such as the later co-citation count of the pair (|delta AP| <= 0.006 with logistic regression).
- **Recommendation (PROVISIONAL):** do not claim, on this evidence, that directed Ricci curvature adds detection
  power at event time; state the reductions in the book, replace ORC escalation by its closed form, add the
  trivial controls as a named baseline, and re-test on the real slice (section 6).

## 2. Three reductions (event time)

Notation: an event e = (u -> v, t) means paper u, published at t, cites v. refs(x) is the reference list of x,
citers_t(x) the papers with ts <= t that cite x, G_t the graph of all rows with ts <= t except the scored row
(CONTRACTS.md). The plan's signals are the directed Forman screen F(e) = 2 - in_deg_t(u) - out_deg_t(v)
(Saucan et al., 2019) and directed Ollivier-Ricci curvature (Ollivier, 2009) in the GraphRicciCurvature
convention (Ni et al., 2015): m_u puts mass alpha on u and 1 - alpha uniformly on the papers citing u, m_v puts
alpha on v and 1 - alpha uniformly on the papers v cites, the cost is the directed hop distance, and
kappa(u, v) = 1 - W1(m_u, m_v) / d(u, v).

The key premise is that **the citing paper is uncited at its own publication time**: pred_t(u) is empty. A paper
published at t can only be cited by papers published later (a citation points strictly back in time in the
generator), so in_deg_t(u) = 0 for every event. On the synthetic stream this holds for 100 % of events (asserted
in every run, Table 7). On real data it fails only through same-day citations, preprint or online-first dates, and
date errors; its share must be measured at S4.0, and Propositions 1 to 3 hold exactly for every event where it
holds.

**Proposition 1 (Forman screen).** If pred_t(u) is empty, F(e) = 2 - |refs(v)|.

*Derivation.* in_deg_t(u) = 0 by the premise. Every reference of v is a row at pub(v) < t, so out_deg_t(v) =
|refs(v)|. Hence F(e) depends on the cited paper's reference count only; it is the same for every citation v
receives, and the screen ranks citations by how long the cited paper's reference list is. The identity
F = 2 - |refs(v)| is asserted on every event of every run (the script aborts otherwise); Spearman
rho(F, |refs(v)|) = -1.000 (Table 3).

**Proposition 2 (directed ORC).** If pred_t(u) is empty and v cites k > 0 papers, of which s are also cited by u,

    kappa(u, v) = -(1 - alpha) * (1 - s / k),          and kappa(u, v) = 0 if k = 0.

*Derivation.* m_u is a Dirac at u, so the only transport plan moves each unit of m_v straight to u and
W1 = sum_x m_v(x) d(u, x). d(u, v) = 1. A reference w of v is at distance 1 if u also cites w, and otherwise at
distance 2 through u -> v -> w (never 0, since w is older than u). So W1 = alpha + (1 - alpha)(s/k + 2(k - s)/k) =
alpha + (1 - alpha)(2 - s/k) and kappa = 1 - W1 = -(1 - alpha)(1 - s/k). If k = 0 both measures are Diracs and
W1 = d(u, v) = 1. Every path used has length <= 2 inside the one-hop neighbourhoods of u and v, so the radius-1
subgraph already gives the exact value; radius 2 adds nothing (Table 7: GraphRicciCurvature 0.6.1 at radius 1
and 2 agrees with the closed form to 2e-15 on every checked event).

So the plan's ORC at event time is a strictly increasing function of the **coupling share** s / k, the share of
v's references that u also cites (bibliographic coupling; Kessler, 1963); with the convention share = 1 when
k = 0, kappa = -(1 - alpha)(1 - share) for every event. It ranks events exactly as that share does (Spearman
1.000, Table 3), and it costs one set intersection instead of an optimal-transport problem.

**Proposition 3 (co-citation curvature).** If pred_t(u) is empty, u has no neighbour in the time-t co-citation
graph (Small, 1973), so any co-citation statistic of the pair (u, v) is 0 and the augmented Forman curvature of
the pair (u, v) in that graph (virtual-edge convention, section 3.3) is 2 - deg_CC(v): again a property of v
alone. Its Spearman correlation with v's citation count is -0.97 (Table 3).

Two more signals are exact linear functions of counts by construction (|refs(u)| counts v, in_deg_t(v) excludes
the scored row): the reversed Forman variant S3b = 2 - (|refs(u)| - 1) - in_deg_t(v), and the augmented Forman
curvature S4b = 2 - (|refs(u)| - 1) - |refs(v)| - in_deg_t(v) + 3 (coupling + transitive), where "transitive" =
|refs(u) & citers_t(v)| counts papers u cites that themselves cite v (the only other triangle type when u is
uncited).

## 3. Method

### 3.1 Data and split

- Stream: `generate_citation_stream(n_works=12000, seed=s)` with the default knobs (8 fields, 2005-2025, nominal
  25 references per paper), seeds s = 0, 1, 2: about 294,000 natural citation events per seed.
- Cutoff c: the latest event time with at least 20 % of natural events after it (2023-01-10 for seed 0). Test events
  are those with t > c (`citecurv.eval.split.split_events`; the validation split is not used). The fit period is
  (c - 5 years, c].

### 3.2 Planted groups, positives and samples

- Four settings, each a separate run per seed: the three `OBVIOUSNESS` presets of `inject_cartel` (CIDRE-style
  donor/recipient cartels; Kojaku et al., 2021) and the clique variant `inject_clique` (10 cliques of 10 papers,
  drop share 0, members drawn from all fields and dates). Cartels: 10 per period for blatant and moderate, 20 for
  subtle (10 subtle cartels give only about 50 positives); members of a cartel have disjoint author sets.
- Every run plants groups twice in the same stream: first inside the fit period (members published in
  [c - 5 y, c - 1], injection seed s + 1000), then in the test period (`t_min = c`, injection seed s), with
  disjoint members. Positives = injected events: fit positives from the fit-period groups, test positives from the
  test-period groups. All test positives are after the cutoff (asserted).
- The subtle preset's 1,460-day window is longer than the test period (about 3 years), so the injector falls back
  to windows cut short by the data end (its documented behaviour); subtle test cartels are therefore squeezed in
  time somewhat more than fit-period ones.
- Samples, identical for every signal of a run: the **test sample** holds all test positives plus 10,000 random
  test negatives; the **fit sample** all fit positives plus 3,000 random natural events of the fit period
  (sampling seed (s, setting)). AUROC is unbiased under negative subsampling; AP and precision@100 are not
  comparable with full-population values, because positives are more frequent in the sample. For the cheap
  signals the JSON also holds metrics on the full test set (`single_full_test`).

### 3.3 Signals

Trivial controls (tier A0 = the minimal set specified for this study; tier A adds the two marked +; tier B adds the
two marked B). All are read at t with the scored row excluded.

| Name | Definition |
|---|---|
| `c_age` | age of the cited paper v at t (days) |
| `c_refs_v` | out_deg(v): v's reference count |
| `c_indeg_v` | in_deg_t(v): citations v has received by t |
| `c_recent_indeg_v` (+) | citations of v in (t - 365 d, t]: a temporal control in the spirit of `s_temp` |
| `c_refs_u` (+) | u's other references |
| `c_coupling`, `c_jaccard` | bibliographic coupling \|refs(u) & refs(v)\| and its Jaccard index |
| `c_cocite` | co-citation count of (u, v) by t (identically 0, Proposition 3; dropped from regressions) |
| `c_same_field`, `c_cosine` | same-field indicator; cosine of the synthetic embeddings |
| `b_coupling_share` (B) | \|refs(u) & refs(v)\| / \|refs(v)\| (1 when v cites nothing): S2's equivalent |
| `b_transitive` (B) | \|refs(u) & citers_t(v)\|: papers u cites that already cite v |

Curvature signals. S2 to S6 are event-time signals on G_t (ts <= t, scored row excluded); where a formula needs
the scored edge for its transport geometry or degrees, it is added back (it carries only the event itself, as in
`orc.escalate`). Definitions and leak arguments are in the docstrings of `citecurv.curvature.variants`.

| Name | Formulation |
|---|---|
| S1 `s1_forman` | plan screen F = 2 - in_deg_t(u) - out_deg_t(v) (`screen.forman_screen`) |
| S1 `s1_forman_abs` | two-sided screen \|F - median of F on the fit sample\| |
| S2 `s2_orc_dir` | plan directed ORC, alpha = 0.5 (closed form of Proposition 2; library-checked at radius 1 and 2) |
| S3a `s3a_orc_rev` | reversed directed ORC: m_u on u's references, m_v on v's citers by t, undirected hop metric |
| S3b `s3b_forman_rev` | reversed Forman 2 - (out(u) - 1) - (in_t(v) - 1) |
| S4a `s4a_orc_undir` | undirected ORC of (u, v) on G_t |
| S4b `s4b_forman_aug` | augmented Forman 4 - deg(u) - deg(v) + 3 #triangles (Samal et al., 2018) on G_t |
| S5a `s5a_bc_forman` | augmented Forman of the pair (u, v) in the bibliographic-coupling graph at t |
| S5b `s5b_cc_forman` | augmented Forman of the pair (u, v) in the co-citation graph at t |
| S6a `s6a_forman_dyn` | mean plan-Forman of v's incident edges at t minus at max(t - 365 d, v's first edge) |
| S6b `s6b_aug_shift` | mean arrival-time S4b of v's in-edges in (t - 365 d, t] minus that of earlier in-edges |
| S6c `s6c_orc_shift` | the same with arrival-time S2 |
| S7 `s7_forman_d{365,730}` | AUDIT MODE: plan Forman of (u, v) on the graph at t + 365 d or t + 730 d |
| S7 `s7_orc_dir_d{365,730}` | AUDIT MODE: plan directed ORC of (u, v) on the graph at t + delta |

For S5, (u, v) is added as a virtual edge when the two papers share no reference (or no citing paper), the same
convention as adding the scored edge for ORC. S6b and S6c are undefined when v has no in-edge on one side of the
window (always the case for a cited paper younger than 365 days); undefined values are imputed with the fit-sample
median. **S7 violates NFR3 by design**: at t + delta the citing paper u has been cited by later papers, among them
later members of the same cartel, so S7 uses information that did not exist at t. It is valid only for a
retrospective audit of citations at least delta old; the audit controls (in_deg(u), in_deg(v) and the
co-citation count of (u, v), all at t + delta) are its fair comparison. In the last delta of the stream the future
graph is truncated by the data end (share reported in the JSON, `share_test_audit_horizon_beyond_data_end`).

### 3.4 Evaluation

- Single-signal metrics: `citecurv.eval.metrics.ranking_metrics` on the test sample (AUROC, AP, P@100, found@100).
  The orientation of each signal is chosen on the fit sample (AUROC >= 0.5 as is, otherwise negated), never on
  test labels; the tables show it in brackets.
- Age-restricted view: test-sample events whose cited paper is no older than the oldest fit-period positive
  (threshold from the fit sample only). Every positive is in it; inside it, cited-paper recency can no longer carry
  the ranking. An AUROC below 0.5 there means that the orientation learned on all events reverses among recent
  cited papers.
- Equivalence: Spearman correlation of each curvature signal with its suspected trivial equivalent and with the
  closest control, and R² of an ordinary least-squares fit of asinh(signal) on the tier-A controls (log1p counts),
  on the test sample.
- Incremental value: a logistic regression (C = 1, standardised features, fitted on the fit sample only) on the
  controls alone versus the controls plus one curvature signal (or all of them), each scored on the test sample;
  reported as delta AP and delta AUROC, paired within a seed, mean ± sd over the three seeds. Counts enter as
  log1p, curvature as asinh, constant columns are dropped. Gradient boosting (`HistGradientBoostingClassifier`,
  150 iterations, 15 leaves) is a non-linear sensitivity check with the same design. Why this design: the model
  is fitted on earlier, disjoint planted groups, as a deployed calibration would be; the test events see those
  earlier groups in their history, which is realistic, and no test label enters any fit, orientation or
  threshold. Cross-fitting on test events was not used because it would tune on the evaluation period.
- Bases: A0 (minimal control set), A (A0 + recent citations of v + u's reference count), B (A + coupling share +
  transitive count), and for S7 the audit base (A + the three audit controls of the same delta).

### 3.5 Validation

- Fast-path premise, asserted per run: every event has t = pub(src) and pub(dst) < pub(src), so G_t is the
  subgraph induced by papers published by t and the study can use vectorised CSR arrays.
- Every fast-path signal is recomputed with the reference functions of `citecurv.curvature.variants` on 10 random
  test-sample events per run (S6b/S6c only where v has at most 40 in-edges); a difference above 1e-9 aborts.
- Seed 0, 12 events per setting, against GraphRicciCurvature 0.6.1 (POT 0.9.6, OTD, pairwise shortest paths):
  S2 closed form vs `orc.escalate` at radius 1 and 2; S4a vs `OllivierRicci` on the undirected radius-2 graph;
  S4b vs `FormanRicci(method="augmented")`; S7 directed ORC vs `orc.orc_directed` on the audit graph.
- `tests/test_curvature_variants.py`: hand-computed micro cases for every function and library cross-checks.
- Determinism (NFR4): two identical `--quick` runs gave bit-identical summaries. The reference functions'
  cross-check differences vary at the 1e-15 level between processes, because `EventStore.subgraph_at` iterates
  over a set of string ids whose order depends on Python's hash seed (summation order only).

## 4. Results

All results are SYNTHETIC. Mean ± sample sd over seeds 0, 1, 2; deltas are paired within a seed. Prevalence in the
test sample: 10.7 % (blatant), 2.5 % (moderate), 1.2 % (subtle), 4.3 % (clique); on the full test set it is about
ten times lower, so AP values in these tables are higher than full-population AP (for example, cited-paper age:
AP 0.60 / 0.13 / 0.13 / 0.13 on the full test set against 0.89 / 0.46 / 0.42 / 0.46 on the sample).
`python scripts/curvature_study.py --tables results/curvature_study.json` regenerates every table (it prints two
further gradient-boosting tables, base A0 and audit mode, which agree with the ones shown).

### 4.1 Counts

**Table 1. Event counts (mean ± sd over seeds)**

| Setting | test events | test positives | fit positives | test sample (pos) | age-restricted sample (pos) | age limit (days) |
|---|---|---|---|---|---|---|
| blatant | 60083 | 1195 ± 77 | 1086 ± 89 | 11195 (1195) | 1499 (1194) | 360 ± 3 |
| moderate | 59148 | 259 ± 12 | 222 ± 48 | 10259 (259) | 1000 (258) | 680 ± 5 |
| subtle | 59013 | 124 ± 8 | 119 ± 6 | 10124 (124) | 1942 (124) | 1337 ± 97 |
| clique | 59338 | 449 ± 1 | 449 ± 1 | 10449 (449) | 3035 (449) | 1750 ± 31 |


### 4.2 Single signals

**Table 2a. Single-signal AUROC on the test sample (mean ± sd over seeds; in brackets the orientation chosen on the fit sample: + means higher raw value = more anomalous)**

| Signal | blatant | moderate | subtle | clique |
|---|---|---|---|---|
| `c_age` | 0.991 ± 0.001 (-) | 0.979 ± 0.001 (-) | 0.984 ± 0.003 (-) | 0.965 ± 0.002 (-) |
| `c_refs_v` | 0.584 ± 0.017 (+) | 0.530 ± 0.037 (+) | 0.520 ± 0.057 (mixed) | 0.552 ± 0.081 (+) |
| `c_indeg_v` | 0.860 ± 0.002 (-) | 0.930 ± 0.005 (-) | 0.958 ± 0.003 (-) | 0.869 ± 0.005 (-) |
| `c_recent_indeg_v` | 0.438 ± 0.010 (-) | 0.703 ± 0.013 (-) | 0.824 ± 0.007 (-) | 0.536 ± 0.019 (-) |
| `c_refs_u` | 0.534 ± 0.005 (-) | 0.631 ± 0.009 (-) | 0.651 ± 0.032 (-) | 0.558 ± 0.035 (-) |
| `c_coupling` | 0.723 ± 0.007 (+) | 0.506 ± 0.045 (-) | 0.565 ± 0.028 (-) | 0.734 ± 0.007 (+) |
| `c_jaccard` | 0.756 ± 0.006 (+) | 0.487 ± 0.051 (mixed) | 0.551 ± 0.036 (-) | 0.770 ± 0.012 (+) |
| `c_cocite` | 0.500 ± 0.000 (+) | 0.500 ± 0.000 (+) | 0.500 ± 0.000 (+) | 0.500 ± 0.000 (+) |
| `c_same_field` | 0.558 ± 0.001 (+) | 0.559 ± 0.001 (+) | 0.558 ± 0.001 (+) | 0.863 ± 0.014 (-) |
| `c_cosine` | 0.508 ± 0.026 (mixed) | 0.521 ± 0.033 (mixed) | 0.524 ± 0.023 (mixed) | 0.869 ± 0.019 (-) |
| `b_coupling_share` | 0.707 ± 0.007 (+) | 0.515 ± 0.044 (-) | 0.573 ± 0.034 (-) | 0.725 ± 0.019 (+) |
| `b_transitive` | 0.716 ± 0.005 (+) | 0.567 ± 0.013 (-) | 0.651 ± 0.007 (-) | 0.768 ± 0.004 (+) |
| `s1_forman` | 0.584 ± 0.017 (-) | 0.530 ± 0.037 (-) | 0.520 ± 0.057 (mixed) | 0.552 ± 0.081 (-) |
| `s1_forman_abs` | 0.493 ± 0.002 (+) | 0.515 ± 0.065 (-) | 0.522 ± 0.026 (mixed) | 0.510 ± 0.054 (mixed) |
| `s2_orc_dir` | 0.707 ± 0.007 (+) | 0.515 ± 0.044 (-) | 0.573 ± 0.034 (-) | 0.725 ± 0.019 (+) |
| `s3a_orc_rev` | 0.871 ± 0.015 (+) | 0.786 ± 0.048 (+) | 0.840 ± 0.012 (+) | 0.758 ± 0.024 (+) |
| `s3b_forman_rev` | 0.768 ± 0.005 (+) | 0.842 ± 0.005 (+) | 0.859 ± 0.017 (+) | 0.779 ± 0.022 (+) |
| `s4a_orc_undir` | 0.718 ± 0.047 (+) | 0.547 ± 0.133 (mixed) | 0.604 ± 0.083 (-) | 0.716 ± 0.020 (-) |
| `s4b_forman_aug` | 0.810 ± 0.013 (+) | 0.806 ± 0.022 (+) | 0.799 ± 0.036 (+) | 0.834 ± 0.062 (+) |
| `s5a_bc_forman` | 0.653 ± 0.055 (+) | 0.707 ± 0.072 (+) | 0.682 ± 0.063 (+) | 0.617 ± 0.028 (-) |
| `s5b_cc_forman` | 0.884 ± 0.005 (+) | 0.937 ± 0.003 (+) | 0.957 ± 0.003 (+) | 0.882 ± 0.008 (+) |
| `s6a_forman_dyn` | 0.625 ± 0.013 (-) | 0.627 ± 0.012 (+) | 0.791 ± 0.013 (+) | 0.499 ± 0.077 (mixed) |
| `s6b_aug_shift` | 0.463 ± 0.075 (mixed) | 0.456 ± 0.013 (+) | 0.432 ± 0.033 (+) | 0.551 ± 0.013 (+) |
| `s6c_orc_shift` | 0.364 ± 0.007 (+) | 0.412 ± 0.013 (+) | 0.379 ± 0.005 (+) | 0.517 ± 0.020 (+) |
| `a_indeg_u_d365` | 0.827 ± 0.011 (+) | 0.664 ± 0.021 (+) | 0.528 ± 0.060 (mixed) | 0.784 ± 0.019 (+) |
| `a_indeg_v_d365` | 0.782 ± 0.005 (-) | 0.905 ± 0.005 (-) | 0.959 ± 0.003 (-) | 0.834 ± 0.011 (-) |
| `a_cocite_d365` | 0.862 ± 0.005 (+) | 0.595 ± 0.017 (+) | 0.499 ± 0.004 (mixed) | 0.891 ± 0.004 (+) |
| `s7_forman_d365` | 0.653 ± 0.012 (-) | 0.548 ± 0.038 (-) | 0.514 ± 0.057 (mixed) | 0.596 ± 0.073 (-) |
| `s7_orc_dir_d365` | 0.722 ± 0.038 (+) | 0.585 ± 0.047 (-) | 0.555 ± 0.048 (-) | 0.670 ± 0.049 (+) |
| `a_indeg_u_d730` | 0.779 ± 0.010 (+) | 0.606 ± 0.019 (+) | 0.502 ± 0.129 (mixed) | 0.726 ± 0.013 (+) |
| `a_indeg_v_d730` | 0.799 ± 0.004 (-) | 0.908 ± 0.007 (-) | 0.964 ± 0.002 (-) | 0.832 ± 0.008 (-) |
| `a_cocite_d730` | 0.858 ± 0.006 (+) | 0.594 ± 0.012 (+) | 0.501 ± 0.009 (mixed) | 0.891 ± 0.001 (+) |
| `s7_forman_d730` | 0.646 ± 0.012 (-) | 0.542 ± 0.039 (-) | 0.510 ± 0.049 (mixed) | 0.601 ± 0.070 (-) |
| `s7_orc_dir_d730` | 0.718 ± 0.031 (+) | 0.563 ± 0.056 (-) | 0.527 ± 0.045 (-) | 0.320 ± 0.060 (-) |


**Table 2b. Single-signal AP on the test sample (negatives subsampled) (mean ± sd over seeds; in brackets the orientation chosen on the fit sample: + means higher raw value = more anomalous)**

| Signal | blatant | moderate | subtle | clique |
|---|---|---|---|---|
| `c_age` | 0.890 ± 0.019 (-) | 0.459 ± 0.020 (-) | 0.419 ± 0.083 (-) | 0.457 ± 0.035 (-) |
| `c_refs_v` | 0.133 ± 0.007 (+) | 0.028 ± 0.004 (+) | 0.014 ± 0.004 (mixed) | 0.055 ± 0.018 (+) |
| `c_indeg_v` | 0.313 ± 0.014 (-) | 0.172 ± 0.008 (-) | 0.146 ± 0.008 (-) | 0.156 ± 0.006 (-) |
| `c_recent_indeg_v` | 0.097 ± 0.004 (-) | 0.047 ± 0.002 (-) | 0.041 ± 0.002 (-) | 0.046 ± 0.002 (-) |
| `c_refs_u` | 0.107 ± 0.004 (-) | 0.038 ± 0.001 (-) | 0.020 ± 0.001 (-) | 0.048 ± 0.005 (-) |
| `c_coupling` | 0.216 ± 0.023 (+) | 0.025 ± 0.003 (-) | 0.014 ± 0.001 (-) | 0.108 ± 0.007 (+) |
| `c_jaccard` | 0.385 ± 0.047 (+) | 0.025 ± 0.004 (mixed) | 0.014 ± 0.001 (-) | 0.314 ± 0.082 (+) |
| `c_cocite` | 0.107 ± 0.006 (+) | 0.025 ± 0.001 (+) | 0.012 ± 0.001 (+) | 0.043 ± 0.000 (+) |
| `c_same_field` | 0.119 ± 0.007 (+) | 0.029 ± 0.001 (+) | 0.014 ± 0.001 (+) | 0.211 ± 0.012 (-) |
| `c_cosine` | 0.099 ± 0.005 (mixed) | 0.024 ± 0.003 (mixed) | 0.013 ± 0.002 (mixed) | 0.227 ± 0.013 (-) |
| `b_coupling_share` | 0.197 ± 0.019 (+) | 0.026 ± 0.004 (-) | 0.015 ± 0.001 (-) | 0.107 ± 0.026 (+) |
| `b_transitive` | 0.190 ± 0.013 (+) | 0.029 ± 0.001 (-) | 0.018 ± 0.001 (-) | 0.106 ± 0.002 (+) |
| `s1_forman` | 0.133 ± 0.007 (-) | 0.028 ± 0.004 (-) | 0.014 ± 0.004 (mixed) | 0.055 ± 0.018 (-) |
| `s1_forman_abs` | 0.116 ± 0.005 (+) | 0.028 ± 0.008 (-) | 0.015 ± 0.001 (mixed) | 0.052 ± 0.015 (mixed) |
| `s2_orc_dir` | 0.197 ± 0.019 (+) | 0.026 ± 0.004 (-) | 0.015 ± 0.001 (-) | 0.107 ± 0.026 (+) |
| `s3a_orc_rev` | 0.437 ± 0.033 (+) | 0.153 ± 0.036 (+) | 0.150 ± 0.022 (+) | 0.221 ± 0.024 (+) |
| `s3b_forman_rev` | 0.225 ± 0.009 (+) | 0.128 ± 0.007 (+) | 0.084 ± 0.007 (+) | 0.111 ± 0.017 (+) |
| `s4a_orc_undir` | 0.222 ± 0.060 (+) | 0.028 ± 0.007 (mixed) | 0.015 ± 0.004 (-) | 0.085 ± 0.005 (-) |
| `s4b_forman_aug` | 0.346 ± 0.030 (+) | 0.104 ± 0.017 (+) | 0.052 ± 0.012 (+) | 0.316 ± 0.132 (+) |
| `s5a_bc_forman` | 0.148 ± 0.035 (+) | 0.047 ± 0.015 (+) | 0.021 ± 0.006 (+) | 0.056 ± 0.006 (-) |
| `s5b_cc_forman` | 0.348 ± 0.021 (+) | 0.189 ± 0.008 (+) | 0.153 ± 0.009 (+) | 0.170 ± 0.012 (+) |
| `s6a_forman_dyn` | 0.339 ± 0.024 (-) | 0.074 ± 0.006 (+) | 0.068 ± 0.003 (+) | 0.069 ± 0.021 (mixed) |
| `s6b_aug_shift` | 0.160 ± 0.021 (mixed) | 0.034 ± 0.002 (+) | 0.018 ± 0.001 (+) | 0.059 ± 0.005 (+) |
| `s6c_orc_shift` | 0.140 ± 0.008 (+) | 0.032 ± 0.001 (+) | 0.016 ± 0.001 (+) | 0.051 ± 0.001 (+) |
| `a_indeg_u_d365` | 0.331 ± 0.012 (+) | 0.047 ± 0.004 (+) | 0.013 ± 0.002 (mixed) | 0.159 ± 0.021 (+) |
| `a_indeg_v_d365` | 0.201 ± 0.011 (-) | 0.125 ± 0.009 (-) | 0.159 ± 0.015 (-) | 0.114 ± 0.007 (-) |
| `a_cocite_d365` | 0.704 ± 0.013 (+) | 0.072 ± 0.021 (+) | 0.012 ± 0.001 (mixed) | 0.711 ± 0.007 (+) |
| `s7_forman_d365` | 0.153 ± 0.006 (-) | 0.029 ± 0.004 (-) | 0.014 ± 0.003 (mixed) | 0.059 ± 0.018 (-) |
| `s7_orc_dir_d365` | 0.301 ± 0.063 (+) | 0.033 ± 0.004 (-) | 0.015 ± 0.001 (-) | 0.095 ± 0.020 (+) |
| `a_indeg_u_d730` | 0.259 ± 0.012 (+) | 0.033 ± 0.002 (+) | 0.013 ± 0.005 (mixed) | 0.107 ± 0.005 (+) |
| `a_indeg_v_d730` | 0.215 ± 0.011 (-) | 0.131 ± 0.013 (-) | 0.185 ± 0.016 (-) | 0.113 ± 0.005 (-) |
| `a_cocite_d730` | 0.676 ± 0.014 (+) | 0.061 ± 0.016 (+) | 0.012 ± 0.001 (mixed) | 0.695 ± 0.013 (+) |
| `s7_forman_d730` | 0.150 ± 0.006 (-) | 0.029 ± 0.004 (-) | 0.014 ± 0.003 (mixed) | 0.059 ± 0.018 (-) |
| `s7_orc_dir_d730` | 0.275 ± 0.059 (+) | 0.030 ± 0.005 (-) | 0.014 ± 0.001 (-) | 0.029 ± 0.003 (-) |


**Table 2c. Single-signal AUROC on the age-restricted test sample (mean ± sd over seeds; in brackets the orientation chosen on the fit sample: + means higher raw value = more anomalous)**

| Signal | blatant | moderate | subtle | clique |
|---|---|---|---|---|
| `c_age` | 0.722 ± 0.016 (-) | 0.716 ± 0.017 (-) | 0.911 ± 0.008 (-) | 0.866 ± 0.008 (-) |
| `c_refs_v` | 0.563 ± 0.031 (+) | 0.515 ± 0.042 (+) | 0.512 ± 0.042 (mixed) | 0.523 ± 0.087 (+) |
| `c_indeg_v` | 0.195 ± 0.005 (-) | 0.499 ± 0.005 (-) | 0.800 ± 0.020 (-) | 0.601 ± 0.010 (-) |
| `c_recent_indeg_v` | 0.195 ± 0.005 (-) | 0.479 ± 0.005 (-) | 0.725 ± 0.022 (-) | 0.448 ± 0.022 (-) |
| `c_refs_u` | 0.554 ± 0.016 (-) | 0.624 ± 0.014 (-) | 0.641 ± 0.023 (-) | 0.552 ± 0.039 (-) |
| `c_coupling` | 0.740 ± 0.014 (+) | 0.475 ± 0.051 (-) | 0.538 ± 0.033 (-) | 0.750 ± 0.005 (+) |
| `c_jaccard` | 0.774 ± 0.017 (+) | 0.477 ± 0.078 (mixed) | 0.520 ± 0.041 (-) | 0.787 ± 0.010 (+) |
| `c_cocite` | 0.500 ± 0.000 (+) | 0.500 ± 0.000 (+) | 0.500 ± 0.000 (+) | 0.500 ± 0.000 (+) |
| `c_same_field` | 0.548 ± 0.004 (+) | 0.551 ± 0.010 (+) | 0.553 ± 0.003 (+) | 0.866 ± 0.013 (-) |
| `c_cosine` | 0.516 ± 0.047 (mixed) | 0.521 ± 0.043 (mixed) | 0.517 ± 0.023 (mixed) | 0.876 ± 0.017 (-) |
| `b_coupling_share` | 0.737 ± 0.018 (+) | 0.478 ± 0.054 (-) | 0.539 ± 0.039 (-) | 0.753 ± 0.017 (+) |
| `b_transitive` | 0.861 ± 0.004 (+) | 0.406 ± 0.017 (-) | 0.519 ± 0.007 (-) | 0.881 ± 0.002 (+) |
| `s1_forman` | 0.563 ± 0.031 (-) | 0.515 ± 0.042 (-) | 0.512 ± 0.042 (mixed) | 0.523 ± 0.087 (-) |
| `s1_forman_abs` | 0.492 ± 0.013 (+) | 0.512 ± 0.061 (-) | 0.518 ± 0.020 (mixed) | 0.508 ± 0.050 (mixed) |
| `s2_orc_dir` | 0.737 ± 0.018 (+) | 0.478 ± 0.054 (-) | 0.539 ± 0.039 (-) | 0.753 ± 0.017 (+) |
| `s3a_orc_rev` | 0.475 ± 0.007 (+) | 0.595 ± 0.026 (+) | 0.783 ± 0.022 (+) | 0.738 ± 0.020 (+) |
| `s3b_forman_rev` | 0.505 ± 0.012 (+) | 0.625 ± 0.014 (+) | 0.695 ± 0.028 (+) | 0.587 ± 0.037 (+) |
| `s4a_orc_undir` | 0.840 ± 0.021 (+) | 0.589 ± 0.146 (mixed) | 0.508 ± 0.089 (-) | 0.670 ± 0.021 (-) |
| `s4b_forman_aug` | 0.633 ± 0.017 (+) | 0.610 ± 0.022 (+) | 0.631 ± 0.050 (+) | 0.726 ± 0.087 (+) |
| `s5a_bc_forman` | 0.597 ± 0.058 (+) | 0.650 ± 0.095 (+) | 0.632 ± 0.082 (+) | 0.688 ± 0.025 (-) |
| `s5b_cc_forman` | 0.215 ± 0.007 (+) | 0.536 ± 0.005 (+) | 0.802 ± 0.019 (+) | 0.638 ± 0.026 (+) |
| `s6a_forman_dyn` | 0.756 ± 0.015 (-) | 0.461 ± 0.008 (+) | 0.665 ± 0.015 (+) | 0.506 ± 0.100 (mixed) |
| `s6b_aug_shift` | 0.500 ± 0.000 (mixed) | 0.496 ± 0.007 (+) | 0.365 ± 0.014 (+) | 0.450 ± 0.003 (+) |
| `s6c_orc_shift` | 0.500 ± 0.000 (+) | 0.478 ± 0.004 (+) | 0.313 ± 0.013 (+) | 0.405 ± 0.013 (+) |
| `a_indeg_u_d365` | 0.832 ± 0.004 (+) | 0.666 ± 0.023 (+) | 0.531 ± 0.062 (mixed) | 0.783 ± 0.019 (+) |
| `a_indeg_v_d365` | 0.101 ± 0.007 (-) | 0.457 ± 0.016 (-) | 0.823 ± 0.021 (-) | 0.531 ± 0.019 (-) |
| `a_cocite_d365` | 0.859 ± 0.009 (+) | 0.594 ± 0.015 (+) | 0.499 ± 0.006 (mixed) | 0.891 ± 0.004 (+) |
| `s7_forman_d365` | 0.633 ± 0.024 (-) | 0.533 ± 0.042 (-) | 0.506 ± 0.044 (mixed) | 0.566 ± 0.079 (-) |
| `s7_orc_dir_d365` | 0.764 ± 0.026 (+) | 0.515 ± 0.038 (-) | 0.497 ± 0.052 (-) | 0.714 ± 0.043 (+) |
| `a_indeg_u_d730` | 0.785 ± 0.001 (+) | 0.606 ± 0.010 (+) | 0.505 ± 0.133 (mixed) | 0.724 ± 0.014 (+) |
| `a_indeg_v_d730` | 0.159 ± 0.013 (-) | 0.510 ± 0.027 (-) | 0.849 ± 0.015 (-) | 0.546 ± 0.010 (-) |
| `a_cocite_d730` | 0.853 ± 0.008 (+) | 0.590 ± 0.005 (+) | 0.505 ± 0.012 (mixed) | 0.890 ± 0.001 (+) |
| `s7_forman_d730` | 0.625 ± 0.025 (-) | 0.528 ± 0.042 (-) | 0.501 ± 0.042 (mixed) | 0.571 ± 0.075 (-) |
| `s7_orc_dir_d730` | 0.777 ± 0.016 (+) | 0.479 ± 0.048 (-) | 0.452 ± 0.047 (-) | 0.265 ± 0.055 (-) |


### 4.3 How much of each signal is a trivial statistic

**Table 3. Spearman correlation with the suspected trivial equivalent and with the closest control (moderate setting), and R² of the signal on tier-A controls (asinh / log1p scales, test sample)**

| Signal | suspected equivalent | rho | closest control | rho | R² A blatant | R² A moderate | R² A subtle | R² A clique |
|---|---|---|---|---|---|---|---|---|
| `s1_forman` | `c_refs_v` | -1.000 ± 0.000 | c_refs_v | -1.000 ± 0.000 | 0.98 ± 0.00 | 0.98 ± 0.00 | 0.98 ± 0.00 | 0.98 ± 0.00 |
| `s1_forman_abs` | `c_refs_v` | 0.360 ± 0.045 | c_refs_v | 0.360 ± 0.045 | 0.11 ± 0.04 | 0.14 ± 0.03 | 0.16 ± 0.03 | 0.12 ± 0.01 |
| `s2_orc_dir` | `b_coupling_share` | 1.000 ± 0.000 | b_coupling_share | 1.000 ± 0.000 | 0.62 ± 0.01 | 0.58 ± 0.03 | 0.60 ± 0.02 | 0.62 ± 0.01 |
| `s3a_orc_rev` | `b_transitive` | 0.439 ± 0.009 | c_same_field | 0.504 ± 0.006 | 0.54 ± 0.01 | 0.49 ± 0.02 | 0.49 ± 0.02 | 0.43 ± 0.02 |
| `s3b_forman_rev` | `c_indeg_v` | -0.727 ± 0.013 | c_indeg_v | -0.727 ± 0.013 | 0.88 ± 0.00 | 0.89 ± 0.00 | 0.89 ± 0.00 | 0.89 ± 0.00 |
| `s4a_orc_undir` | `b_transitive` | 0.405 ± 0.012 | c_coupling | 0.625 ± 0.016 | 0.73 ± 0.04 | 0.76 ± 0.05 | 0.76 ± 0.05 | 0.74 ± 0.04 |
| `s4b_forman_aug` | `c_indeg_v` | -0.670 ± 0.015 | c_indeg_v | -0.670 ± 0.015 | 0.79 ± 0.01 | 0.81 ± 0.00 | 0.80 ± 0.01 | 0.80 ± 0.00 |
| `s5a_bc_forman` | `c_coupling` | 0.327 ± 0.017 | c_same_field | 0.441 ± 0.017 | 0.16 ± 0.01 | 0.18 ± 0.03 | 0.18 ± 0.02 | 0.17 ± 0.01 |
| `s5b_cc_forman` | `c_indeg_v` | -0.974 ± 0.003 | c_indeg_v | -0.974 ± 0.003 | 0.74 ± 0.00 | 0.75 ± 0.00 | 0.75 ± 0.00 | 0.74 ± 0.01 |
| `s6a_forman_dyn` | `c_recent_indeg_v` | -0.416 ± 0.009 | c_refs_v | -0.507 ± 0.009 | 0.45 ± 0.02 | 0.45 ± 0.00 | 0.45 ± 0.01 | 0.43 ± 0.01 |
| `s6b_aug_shift` | `c_recent_indeg_v` | -0.348 ± 0.010 | c_indeg_v | -0.654 ± 0.007 | 0.36 ± 0.01 | 0.36 ± 0.01 | 0.36 ± 0.01 | 0.35 ± 0.01 |
| `s6c_orc_shift` | `c_recent_indeg_v` | -0.122 ± 0.018 | c_age | -0.658 ± 0.008 | 0.45 ± 0.03 | 0.46 ± 0.01 | 0.46 ± 0.01 | 0.45 ± 0.01 |
| `s7_forman_d365` | `a_indeg_u_d365` | -0.083 ± 0.015 | c_refs_v | -0.994 ± 0.001 | 0.97 ± 0.00 | 0.98 ± 0.00 | 0.97 ± 0.00 | 0.97 ± 0.00 |
| `s7_orc_dir_d365` | `a_indeg_u_d365` | -0.534 ± 0.016 | a_indeg_u_d365 | -0.534 ± 0.016 | 0.34 ± 0.02 | 0.43 ± 0.01 | 0.46 ± 0.03 | 0.47 ± 0.03 |
| `s7_forman_d730` | `a_indeg_u_d730` | -0.132 ± 0.011 | c_refs_v | -0.984 ± 0.003 | 0.96 ± 0.00 | 0.97 ± 0.00 | 0.97 ± 0.00 | 0.97 ± 0.00 |
| `s7_orc_dir_d730` | `a_indeg_u_d730` | -0.349 ± 0.028 | a_indeg_u_d730, b_coupling_share | 0.140 ± 0.445 | 0.32 ± 0.02 | 0.31 ± 0.00 | 0.33 ± 0.01 | 0.32 ± 0.02 |


R² on tier-B controls (JSON) is 0.9998 for S2 (the remaining 0.0002 is the asinh scale, not information), 0.98
for S1, 0.88 to 0.89 for S3b, 0.77 to 0.79 for S4a, 0.80 to 0.81 for S4b, 0.74 to 0.76 for S5b, 0.55 to 0.62 for
S3a, 0.16 to 0.18 for S5a and 0.35 to 0.48 for the S6 signals.

### 4.4 Incremental value at event time

**Table 4-lr-A0. Incremental value, logistic regression, base A0: delta AP / delta AUROC on the test sample (mean ± sd over seeds)**

| Added signal | blatant | moderate | subtle | clique |
|---|---|---|---|---|
| controls only (AP) | 0.945 ± 0.010 | 0.491 ± 0.018 | 0.427 ± 0.064 | 0.872 ± 0.019 |
| + `s1_forman` | +0.000 ± 0.000 / +0.0000 ± 0.0000 | +0.000 ± 0.000 / -0.0000 ± 0.0001 | +0.001 ± 0.002 / -0.0001 ± 0.0001 | +0.000 ± 0.001 / +0.0001 ± 0.0002 |
| + `s1_forman_abs` | +0.000 ± 0.000 / +0.0000 ± 0.0000 | -0.001 ± 0.005 / -0.0003 ± 0.0001 | -0.002 ± 0.003 / -0.0002 ± 0.0005 | +0.000 ± 0.001 / +0.0003 ± 0.0006 |
| + `s2_orc_dir` | +0.000 ± 0.001 / +0.0000 ± 0.0001 | +0.001 ± 0.001 / +0.0003 ± 0.0001 | -0.003 ± 0.002 / +0.0002 ± 0.0002 | +0.003 ± 0.002 / +0.0002 ± 0.0003 |
| + `s3a_orc_rev` | -0.000 ± 0.001 / -0.0000 ± 0.0001 | -0.006 ± 0.004 / -0.0017 ± 0.0004 | +0.009 ± 0.004 / +0.0003 ± 0.0003 | +0.006 ± 0.004 / +0.0017 ± 0.0005 |
| + `s3b_forman_rev` | +0.006 ± 0.002 / +0.0004 ± 0.0002 | +0.019 ± 0.003 / +0.0008 ± 0.0003 | +0.008 ± 0.013 / +0.0010 ± 0.0010 | +0.016 ± 0.011 / +0.0020 ± 0.0015 |
| + `s4a_orc_undir` | +0.010 ± 0.003 / +0.0007 ± 0.0001 | +0.006 ± 0.030 / -0.0009 ± 0.0026 | -0.009 ± 0.005 / -0.0004 ± 0.0003 | +0.030 ± 0.006 / +0.0042 ± 0.0020 |
| + `s4b_forman_aug` | +0.013 ± 0.005 / +0.0009 ± 0.0003 | +0.027 ± 0.004 / +0.0012 ± 0.0006 | +0.008 ± 0.012 / +0.0008 ± 0.0013 | +0.044 ± 0.009 / +0.0062 ± 0.0015 |
| + `s5a_bc_forman` | +0.001 ± 0.001 / +0.0000 ± 0.0001 | -0.000 ± 0.000 / +0.0000 ± 0.0001 | +0.002 ± 0.002 / -0.0001 ± 0.0001 | -0.000 ± 0.003 / +0.0002 ± 0.0003 |
| + `s5b_cc_forman` | +0.017 ± 0.002 / +0.0011 ± 0.0003 | +0.007 ± 0.005 / -0.0008 ± 0.0007 | +0.011 ± 0.005 / +0.0004 ± 0.0003 | +0.023 ± 0.005 / +0.0034 ± 0.0012 |
| + `s6a_forman_dyn` | +0.006 ± 0.002 / +0.0004 ± 0.0002 | +0.005 ± 0.005 / -0.0017 ± 0.0019 | +0.001 ± 0.003 / +0.0000 ± 0.0002 | +0.016 ± 0.012 / +0.0023 ± 0.0028 |
| + `s6b_aug_shift` | +0.000 ± 0.000 / -0.0000 ± 0.0001 | +0.000 ± 0.000 / -0.0000 ± 0.0001 | +0.001 ± 0.001 / +0.0001 ± 0.0002 | +0.004 ± 0.002 / +0.0010 ± 0.0005 |
| + `s6c_orc_shift` | +0.001 ± 0.001 / +0.0001 ± 0.0001 | -0.000 ± 0.001 / -0.0001 ± 0.0001 | +0.000 ± 0.001 / -0.0001 ± 0.0001 | +0.004 ± 0.001 / +0.0008 ± 0.0003 |
| + `all_event_time` | +0.029 ± 0.007 / +0.0018 ± 0.0005 | +0.029 ± 0.020 / -0.0011 ± 0.0021 | +0.003 ± 0.009 / +0.0003 ± 0.0016 | +0.089 ± 0.008 / +0.0105 ± 0.0011 |


**Table 4-lr-A. Incremental value, logistic regression, base A: delta AP / delta AUROC on the test sample (mean ± sd over seeds)**

| Added signal | blatant | moderate | subtle | clique |
|---|---|---|---|---|
| controls only (AP) | 0.971 ± 0.006 | 0.521 ± 0.025 | 0.441 ± 0.078 | 0.900 ± 0.018 |
| + `s1_forman` | +0.000 ± 0.000 / +0.0000 ± 0.0000 | -0.000 ± 0.000 / +0.0000 ± 0.0000 | -0.001 ± 0.002 / +0.0000 ± 0.0001 | +0.000 ± 0.000 / +0.0000 ± 0.0001 |
| + `s1_forman_abs` | -0.000 ± 0.000 / +0.0000 ± 0.0000 | +0.001 ± 0.007 / -0.0001 ± 0.0003 | +0.000 ± 0.000 / -0.0000 ± 0.0002 | +0.000 ± 0.000 / +0.0001 ± 0.0002 |
| + `s2_orc_dir` | -0.000 ± 0.000 / +0.0000 ± 0.0000 | -0.000 ± 0.001 / +0.0001 ± 0.0001 | -0.001 ± 0.002 / +0.0001 ± 0.0001 | +0.001 ± 0.000 / +0.0000 ± 0.0001 |
| + `s3a_orc_rev` | +0.002 ± 0.001 / +0.0002 ± 0.0001 | -0.005 ± 0.007 / -0.0004 ± 0.0003 | +0.005 ± 0.002 / +0.0001 ± 0.0005 | +0.014 ± 0.001 / +0.0024 ± 0.0002 |
| + `s3b_forman_rev` | +0.001 ± 0.000 / +0.0000 ± 0.0000 | +0.004 ± 0.001 / +0.0003 ± 0.0004 | -0.005 ± 0.004 / +0.0000 ± 0.0001 | +0.007 ± 0.003 / +0.0017 ± 0.0002 |
| + `s4a_orc_undir` | +0.004 ± 0.001 / +0.0002 ± 0.0001 | +0.009 ± 0.035 / -0.0002 ± 0.0021 | -0.014 ± 0.018 / -0.0003 ± 0.0003 | +0.028 ± 0.006 / +0.0035 ± 0.0012 |
| + `s4b_forman_aug` | +0.001 ± 0.003 / +0.0001 ± 0.0002 | +0.010 ± 0.005 / +0.0010 ± 0.0003 | -0.003 ± 0.003 / -0.0002 ± 0.0006 | +0.043 ± 0.021 / +0.0057 ± 0.0019 |
| + `s5a_bc_forman` | +0.001 ± 0.001 / +0.0000 ± 0.0000 | +0.000 ± 0.001 / +0.0000 ± 0.0001 | -0.002 ± 0.002 / -0.0001 ± 0.0002 | +0.003 ± 0.001 / +0.0003 ± 0.0002 |
| + `s5b_cc_forman` | +0.000 ± 0.000 / +0.0000 ± 0.0000 | -0.001 ± 0.000 / +0.0001 ± 0.0002 | +0.003 ± 0.002 / +0.0000 ± 0.0001 | +0.005 ± 0.002 / +0.0011 ± 0.0002 |
| + `s6a_forman_dyn` | -0.000 ± 0.001 / -0.0000 ± 0.0001 | -0.005 ± 0.004 / -0.0007 ± 0.0007 | +0.000 ± 0.001 / +0.0000 ± 0.0001 | -0.004 ± 0.011 / -0.0003 ± 0.0019 |
| + `s6b_aug_shift` | +0.000 ± 0.001 / +0.0000 ± 0.0001 | -0.000 ± 0.001 / +0.0000 ± 0.0000 | -0.001 ± 0.001 / -0.0001 ± 0.0002 | +0.001 ± 0.001 / +0.0004 ± 0.0001 |
| + `s6c_orc_shift` | +0.001 ± 0.001 / +0.0000 ± 0.0001 | -0.001 ± 0.001 / -0.0002 ± 0.0002 | +0.000 ± 0.000 / +0.0000 ± 0.0000 | +0.001 ± 0.002 / +0.0003 ± 0.0004 |
| + `all_event_time` | +0.008 ± 0.004 / +0.0005 ± 0.0002 | +0.003 ± 0.035 / -0.0009 ± 0.0026 | -0.016 ± 0.018 / -0.0009 ± 0.0006 | +0.065 ± 0.012 / +0.0079 ± 0.0012 |


**Table 4-lr-B. Incremental value, logistic regression, base B: delta AP / delta AUROC on the test sample (mean ± sd over seeds)**

| Added signal | blatant | moderate | subtle | clique |
|---|---|---|---|---|
| controls only (AP) | 0.983 ± 0.002 | 0.562 ± 0.030 | 0.439 ± 0.078 | 0.990 ± 0.003 |
| + `s1_forman` | +0.000 ± 0.000 / +0.0000 ± 0.0000 | -0.000 ± 0.002 / +0.0000 ± 0.0001 | -0.001 ± 0.002 / -0.0001 ± 0.0001 | +0.000 ± 0.000 / +0.0000 ± 0.0000 |
| + `s1_forman_abs` | +0.000 ± 0.000 / +0.0000 ± 0.0000 | -0.001 ± 0.006 / -0.0000 ± 0.0003 | -0.001 ± 0.001 / -0.0000 ± 0.0002 | -0.000 ± 0.000 / -0.0000 ± 0.0001 |
| + `s2_orc_dir` | +0.000 ± 0.000 / +0.0000 ± 0.0000 | -0.000 ± 0.001 / +0.0000 ± 0.0001 | -0.000 ± 0.000 / +0.0000 ± 0.0000 | -0.000 ± 0.000 / -0.0000 ± 0.0001 |
| + `s3a_orc_rev` | -0.000 ± 0.000 / +0.0000 ± 0.0000 | -0.003 ± 0.005 / -0.0001 ± 0.0005 | +0.008 ± 0.002 / +0.0001 ± 0.0006 | +0.001 ± 0.001 / -0.0000 ± 0.0001 |
| + `s3b_forman_rev` | +0.001 ± 0.000 / +0.0001 ± 0.0000 | +0.005 ± 0.004 / +0.0004 ± 0.0005 | -0.005 ± 0.004 / +0.0000 ± 0.0001 | -0.000 ± 0.001 / -0.0001 ± 0.0001 |
| + `s4a_orc_undir` | +0.000 ± 0.000 / +0.0000 ± 0.0001 | +0.001 ± 0.028 / -0.0003 ± 0.0015 | -0.014 ± 0.018 / -0.0003 ± 0.0003 | -0.000 ± 0.000 / +0.0000 ± 0.0000 |
| + `s4b_forman_aug` | +0.001 ± 0.001 / +0.0001 ± 0.0001 | +0.006 ± 0.004 / +0.0004 ± 0.0002 | -0.001 ± 0.002 / -0.0003 ± 0.0006 | +0.000 ± 0.001 / -0.0000 ± 0.0001 |
| + `s5a_bc_forman` | +0.000 ± 0.000 / +0.0001 ± 0.0001 | -0.000 ± 0.003 / -0.0000 ± 0.0002 | -0.000 ± 0.002 / -0.0001 ± 0.0002 | +0.001 ± 0.001 / +0.0000 ± 0.0000 |
| + `s5b_cc_forman` | +0.001 ± 0.000 / +0.0000 ± 0.0001 | +0.001 ± 0.000 / +0.0001 ± 0.0003 | +0.003 ± 0.003 / +0.0000 ± 0.0001 | +0.001 ± 0.001 / -0.0000 ± 0.0001 |
| + `s6a_forman_dyn` | +0.000 ± 0.000 / +0.0000 ± 0.0001 | -0.006 ± 0.003 / -0.0006 ± 0.0006 | +0.000 ± 0.001 / +0.0000 ± 0.0000 | -0.001 ± 0.001 / -0.0001 ± 0.0000 |
| + `s6b_aug_shift` | +0.000 ± 0.000 / +0.0001 ± 0.0001 | +0.000 ± 0.001 / -0.0000 ± 0.0001 | -0.001 ± 0.001 / -0.0001 ± 0.0001 | +0.000 ± 0.000 / -0.0000 ± 0.0001 |
| + `s6c_orc_shift` | +0.000 ± 0.000 / +0.0000 ± 0.0001 | -0.001 ± 0.001 / -0.0001 ± 0.0002 | -0.001 ± 0.001 / -0.0001 ± 0.0001 | +0.000 ± 0.000 / +0.0000 ± 0.0000 |
| + `all_event_time` | +0.002 ± 0.001 / +0.0001 ± 0.0001 | -0.003 ± 0.033 / -0.0011 ± 0.0021 | -0.014 ± 0.019 / -0.0010 ± 0.0006 | +0.001 ± 0.001 / +0.0000 ± 0.0001 |


**Table 4-hgb-A. Incremental value, gradient boosting, base A: delta AP / delta AUROC on the test sample (mean ± sd over seeds)**

| Added signal | blatant | moderate | subtle | clique |
|---|---|---|---|---|
| controls only (AP) | 0.985 ± 0.008 | 0.482 ± 0.034 | 0.210 ± 0.054 | 0.910 ± 0.025 |
| + `s1_forman` | +0.000 ± 0.000 / +0.0000 ± 0.0000 | +0.000 ± 0.000 / +0.0000 ± 0.0001 | +0.000 ± 0.000 / +0.0000 ± 0.0000 | +0.000 ± 0.000 / +0.0000 ± 0.0000 |
| + `s1_forman_abs` | +0.001 ± 0.001 / +0.0000 ± 0.0001 | -0.007 ± 0.020 / -0.0001 ± 0.0006 | +0.015 ± 0.020 / +0.0010 ± 0.0017 | -0.001 ± 0.003 / +0.0001 ± 0.0002 |
| + `s2_orc_dir` | -0.002 ± 0.002 / +0.0000 ± 0.0001 | -0.003 ± 0.001 / +0.0000 ± 0.0001 | +0.003 ± 0.013 / -0.0005 ± 0.0023 | -0.003 ± 0.002 / -0.0004 ± 0.0006 |
| + `s3a_orc_rev` | +0.006 ± 0.004 / +0.0005 ± 0.0002 | +0.008 ± 0.031 / +0.0007 ± 0.0015 | +0.012 ± 0.012 / +0.0005 ± 0.0022 | +0.054 ± 0.008 / +0.0048 ± 0.0011 |
| + `s3b_forman_rev` | -0.001 ± 0.001 / +0.0000 ± 0.0000 | -0.009 ± 0.004 / -0.0001 ± 0.0005 | +0.001 ± 0.012 / -0.0005 ± 0.0006 | -0.000 ± 0.001 / -0.0003 ± 0.0003 |
| + `s4a_orc_undir` | +0.002 ± 0.003 / +0.0002 ± 0.0003 | +0.017 ± 0.042 / +0.0007 ± 0.0032 | +0.013 ± 0.028 / -0.0011 ± 0.0051 | +0.037 ± 0.009 / +0.0033 ± 0.0014 |
| + `s4b_forman_aug` | +0.001 ± 0.001 / +0.0001 ± 0.0000 | -0.007 ± 0.019 / -0.0001 ± 0.0004 | -0.003 ± 0.022 / -0.0007 ± 0.0020 | +0.039 ± 0.007 / +0.0039 ± 0.0014 |
| + `s5a_bc_forman` | +0.000 ± 0.002 / +0.0000 ± 0.0001 | -0.008 ± 0.007 / -0.0006 ± 0.0005 | +0.038 ± 0.033 / +0.0031 ± 0.0025 | +0.009 ± 0.005 / +0.0009 ± 0.0008 |
| + `s5b_cc_forman` | -0.001 ± 0.001 / +0.0000 ± 0.0001 | +0.014 ± 0.016 / +0.0008 ± 0.0009 | +0.005 ± 0.037 / +0.0017 ± 0.0029 | +0.000 ± 0.001 / +0.0000 ± 0.0003 |
| + `s6a_forman_dyn` | -0.000 ± 0.001 / -0.0000 ± 0.0001 | +0.006 ± 0.013 / +0.0005 ± 0.0002 | +0.004 ± 0.016 / +0.0002 ± 0.0011 | -0.002 ± 0.003 / -0.0001 ± 0.0006 |
| + `s6b_aug_shift` | +0.000 ± 0.000 / +0.0000 ± 0.0000 | +0.001 ± 0.010 / -0.0000 ± 0.0005 | +0.004 ± 0.029 / +0.0010 ± 0.0014 | +0.002 ± 0.001 / +0.0002 ± 0.0005 |
| + `s6c_orc_shift` | +0.000 ± 0.000 / +0.0000 ± 0.0000 | -0.015 ± 0.004 / -0.0003 ± 0.0004 | +0.005 ± 0.004 / +0.0003 ± 0.0015 | +0.003 ± 0.003 / +0.0003 ± 0.0005 |
| + `all_event_time` | +0.006 ± 0.003 / +0.0005 ± 0.0001 | +0.034 ± 0.068 / +0.0017 ± 0.0036 | +0.046 ± 0.062 / +0.0028 ± 0.0055 | +0.063 ± 0.021 / +0.0057 ± 0.0028 |


**Table 4-hgb-B. Incremental value, gradient boosting, base B: delta AP / delta AUROC on the test sample (mean ± sd over seeds)**

| Added signal | blatant | moderate | subtle | clique |
|---|---|---|---|---|
| controls only (AP) | 0.994 ± 0.002 | 0.551 ± 0.051 | 0.220 ± 0.032 | 0.993 ± 0.002 |
| + `s1_forman` | +0.000 ± 0.000 / +0.0000 ± 0.0000 | +0.001 ± 0.001 / +0.0000 ± 0.0001 | +0.000 ± 0.000 / +0.0000 ± 0.0000 | +0.000 ± 0.000 / +0.0000 ± 0.0000 |
| + `s1_forman_abs` | -0.001 ± 0.000 / -0.0000 ± 0.0001 | -0.004 ± 0.005 / +0.0004 ± 0.0004 | +0.006 ± 0.011 / +0.0007 ± 0.0005 | +0.000 ± 0.001 / -0.0000 ± 0.0001 |
| + `s2_orc_dir` | +0.000 ± 0.000 / +0.0000 ± 0.0000 | +0.000 ± 0.000 / +0.0000 ± 0.0000 | +0.000 ± 0.000 / +0.0000 ± 0.0000 | +0.000 ± 0.000 / +0.0000 ± 0.0000 |
| + `s3a_orc_rev` | -0.000 ± 0.001 / +0.0000 ± 0.0001 | +0.002 ± 0.023 / +0.0001 ± 0.0012 | +0.002 ± 0.023 / +0.0007 ± 0.0016 | +0.000 ± 0.001 / -0.0000 ± 0.0001 |
| + `s3b_forman_rev` | -0.000 ± 0.001 / +0.0000 ± 0.0000 | +0.009 ± 0.009 / +0.0002 ± 0.0004 | -0.000 ± 0.006 / -0.0005 ± 0.0009 | -0.001 ± 0.001 / -0.0000 ± 0.0001 |
| + `s4a_orc_undir` | -0.001 ± 0.001 / -0.0001 ± 0.0001 | +0.015 ± 0.040 / +0.0006 ± 0.0028 | -0.005 ± 0.034 / -0.0012 ± 0.0047 | +0.000 ± 0.002 / +0.0001 ± 0.0003 |
| + `s4b_forman_aug` | -0.000 ± 0.001 / +0.0000 ± 0.0001 | +0.006 ± 0.005 / +0.0003 ± 0.0003 | -0.008 ± 0.016 / -0.0009 ± 0.0017 | +0.000 ± 0.001 / +0.0000 ± 0.0001 |
| + `s5a_bc_forman` | -0.001 ± 0.001 / -0.0000 ± 0.0001 | -0.008 ± 0.004 / -0.0003 ± 0.0007 | +0.025 ± 0.015 / +0.0029 ± 0.0016 | -0.000 ± 0.001 / +0.0000 ± 0.0000 |
| + `s5b_cc_forman` | -0.000 ± 0.000 / +0.0000 ± 0.0001 | +0.013 ± 0.024 / +0.0004 ± 0.0006 | +0.002 ± 0.022 / +0.0016 ± 0.0017 | +0.000 ± 0.000 / -0.0000 ± 0.0001 |
| + `s6a_forman_dyn` | -0.000 ± 0.000 / -0.0000 ± 0.0001 | -0.000 ± 0.012 / +0.0003 ± 0.0004 | +0.011 ± 0.001 / +0.0001 ± 0.0005 | +0.000 ± 0.000 / -0.0000 ± 0.0001 |
| + `s6b_aug_shift` | +0.000 ± 0.000 / +0.0000 ± 0.0000 | -0.002 ± 0.013 / -0.0002 ± 0.0003 | +0.004 ± 0.007 / +0.0007 ± 0.0014 | +0.000 ± 0.000 / +0.0000 ± 0.0000 |
| + `s6c_orc_shift` | +0.000 ± 0.000 / +0.0000 ± 0.0000 | -0.005 ± 0.020 / +0.0001 ± 0.0005 | +0.002 ± 0.001 / +0.0006 ± 0.0004 | +0.000 ± 0.000 / +0.0000 ± 0.0000 |
| + `all_event_time` | -0.000 ± 0.000 / +0.0000 ± 0.0001 | +0.023 ± 0.052 / +0.0012 ± 0.0029 | +0.034 ± 0.040 / +0.0032 ± 0.0037 | +0.000 ± 0.000 / +0.0001 ± 0.0002 |


**Table 5-lr. Incremental value on the age-restricted test sample, logistic regression, base A: delta AUROC**

| Added signal | blatant | moderate | subtle | clique |
|---|---|---|---|---|
| controls only, age-restricted (AUROC) | 0.942 ± 0.003 | 0.780 ± 0.014 | 0.920 ± 0.004 | 0.970 ± 0.004 |
| + `s1_forman` | +0.0000 ± 0.0002 | -0.0000 ± 0.0003 | -0.0002 ± 0.0005 | +0.0001 ± 0.0002 |
| + `s1_forman_abs` | -0.0001 ± 0.0004 | +0.0005 ± 0.0044 | -0.0003 ± 0.0011 | +0.0003 ± 0.0008 |
| + `s2_orc_dir` | -0.0000 ± 0.0003 | +0.0000 ± 0.0008 | +0.0003 ± 0.0002 | +0.0001 ± 0.0002 |
| + `s3a_orc_rev` | +0.0061 ± 0.0011 | -0.0059 ± 0.0074 | +0.0000 ± 0.0023 | +0.0054 ± 0.0002 |
| + `s3b_forman_rev` | +0.0009 ± 0.0003 | +0.0014 ± 0.0027 | -0.0001 ± 0.0003 | +0.0036 ± 0.0004 |
| + `s4a_orc_undir` | +0.0068 ± 0.0016 | +0.0026 ± 0.0242 | -0.0019 ± 0.0020 | +0.0091 ± 0.0026 |
| + `s4b_forman_aug` | +0.0051 ± 0.0011 | +0.0089 ± 0.0032 | -0.0010 ± 0.0022 | +0.0139 ± 0.0052 |
| + `s5a_bc_forman` | +0.0008 ± 0.0006 | +0.0001 ± 0.0008 | -0.0007 ± 0.0006 | +0.0007 ± 0.0002 |
| + `s5b_cc_forman` | +0.0001 ± 0.0001 | -0.0039 ± 0.0018 | -0.0004 ± 0.0004 | +0.0020 ± 0.0006 |
| + `s6a_forman_dyn` | +0.0002 ± 0.0010 | -0.0047 ± 0.0039 | +0.0000 ± 0.0002 | -0.0007 ± 0.0047 |
| + `s6b_aug_shift` | +0.0012 ± 0.0009 | +0.0000 ± 0.0009 | -0.0003 ± 0.0005 | +0.0008 ± 0.0005 |
| + `s6c_orc_shift` | +0.0009 ± 0.0010 | -0.0012 ± 0.0016 | -0.0001 ± 0.0001 | +0.0008 ± 0.0009 |
| + `all_event_time` | +0.0169 ± 0.0021 | -0.0024 ± 0.0228 | -0.0051 ± 0.0023 | +0.0207 ± 0.0027 |


**Table 5-hgb. Incremental value on the age-restricted test sample, gradient boosting, base A: delta AUROC**

| Added signal | blatant | moderate | subtle | clique |
|---|---|---|---|---|
| controls only, age-restricted (AUROC) | 0.957 ± 0.008 | 0.748 ± 0.014 | 0.816 ± 0.039 | 0.968 ± 0.013 |
| + `s1_forman` | +0.0000 ± 0.0001 | +0.0001 ± 0.0001 | +0.0001 ± 0.0002 | +0.0000 ± 0.0001 |
| + `s1_forman_abs` | -0.0001 ± 0.0011 | -0.0018 ± 0.0072 | +0.0063 ± 0.0103 | +0.0001 ± 0.0006 |
| + `s2_orc_dir` | -0.0013 ± 0.0015 | -0.0004 ± 0.0015 | -0.0023 ± 0.0123 | -0.0012 ± 0.0016 |
| + `s3a_orc_rev` | +0.0151 ± 0.0048 | +0.0088 ± 0.0205 | +0.0035 ± 0.0128 | +0.0183 ± 0.0044 |
| + `s3b_forman_rev` | -0.0005 ± 0.0010 | -0.0016 ± 0.0060 | -0.0026 ± 0.0031 | -0.0006 ± 0.0002 |
| + `s4a_orc_undir` | +0.0054 ± 0.0103 | +0.0090 ± 0.0440 | -0.0035 ± 0.0292 | +0.0124 ± 0.0056 |
| + `s4b_forman_aug` | +0.0021 ± 0.0019 | -0.0016 ± 0.0041 | -0.0030 ± 0.0110 | +0.0147 ± 0.0055 |
| + `s5a_bc_forman` | -0.0002 ± 0.0024 | -0.0085 ± 0.0060 | +0.0176 ± 0.0147 | +0.0034 ± 0.0030 |
| + `s5b_cc_forman` | -0.0003 ± 0.0026 | +0.0097 ± 0.0134 | +0.0099 ± 0.0155 | +0.0001 ± 0.0010 |
| + `s6a_forman_dyn` | -0.0009 ± 0.0011 | +0.0069 ± 0.0024 | +0.0022 ± 0.0052 | +0.0002 ± 0.0022 |
| + `s6b_aug_shift` | +0.0000 ± 0.0000 | -0.0003 ± 0.0062 | +0.0056 ± 0.0073 | +0.0007 ± 0.0018 |
| + `s6c_orc_shift` | +0.0000 ± 0.0000 | -0.0042 ± 0.0042 | +0.0027 ± 0.0088 | +0.0009 ± 0.0019 |
| + `all_event_time` | +0.0135 ± 0.0044 | +0.0228 ± 0.0500 | +0.0176 ± 0.0308 | +0.0216 ± 0.0111 |


### 4.5 Audit mode (violates NFR3 by design)

**Table 6-lr. AUDIT MODE incremental value, logistic regression: delta AP / delta AUROC**

| Model | blatant | moderate | subtle | clique |
|---|---|---|---|---|
| A + audit controls, delta = 365 d (AP) | 0.993 ± 0.002 | 0.568 ± 0.023 | 0.468 ± 0.073 | 0.981 ± 0.004 |
| + `s7_forman_d365` | +0.000 ± 0.000 / +0.0000 ± 0.0000 | -0.000 ± 0.000 / -0.0000 ± 0.0001 | -0.001 ± 0.002 / -0.0001 ± 0.0001 | +0.000 ± 0.000 / +0.0000 ± 0.0000 |
| + `s7_orc_dir_d365` | +0.000 ± 0.000 / +0.0000 ± 0.0000 | -0.001 ± 0.001 / -0.0001 ± 0.0001 | +0.005 ± 0.006 / +0.0000 ± 0.0002 | +0.000 ± 0.001 / +0.0000 ± 0.0003 |
| + `all_audit` | +0.000 ± 0.000 / +0.0000 ± 0.0000 | -0.002 ± 0.002 / -0.0002 ± 0.0002 | +0.004 ± 0.006 / -0.0001 ± 0.0003 | +0.000 ± 0.001 / -0.0000 ± 0.0003 |
| A + audit controls, delta = 730 d (AP) | 0.991 ± 0.003 | 0.551 ± 0.032 | 0.504 ± 0.088 | 0.981 ± 0.003 |
| + `s7_forman_d730` | +0.000 ± 0.000 / +0.0000 ± 0.0000 | -0.000 ± 0.001 / +0.0000 ± 0.0001 | -0.003 ± 0.004 / -0.0001 ± 0.0002 | +0.000 ± 0.000 / +0.0000 ± 0.0001 |
| + `s7_orc_dir_d730` | -0.001 ± 0.001 / -0.0000 ± 0.0001 | -0.004 ± 0.004 / -0.0006 ± 0.0009 | -0.003 ± 0.009 / -0.0002 ± 0.0005 | +0.001 ± 0.001 / +0.0002 ± 0.0001 |
| + `all_audit` | -0.000 ± 0.001 / -0.0000 ± 0.0001 | -0.005 ± 0.006 / -0.0006 ± 0.0009 | -0.006 ± 0.010 / -0.0004 ± 0.0006 | +0.001 ± 0.001 / +0.0002 ± 0.0002 |


For 35 % (delta = 365 days) and 67 % (delta = 730 days) of test events the audit horizon passes the end of the
stream, so their future graph is truncated, while fit-period events see a complete future. This is why the
orientation of S7 ORC at 730 days chosen on the fit period reverses on the clique test events (AUROC 0.27 to 0.32
in Table 2a).

### 4.6 Validation

**Table 7. Validation: fast path vs citecurv.curvature.variants (max abs difference over all checked signals) and vs GraphRicciCurvature (S2 closed form vs escalate radius 1 / 2; S4a / S4b / S7 ORC)**

| Setting | seed | events | fast vs variants | S2 vs library r1 / r2 | S4a / S4b / S7 vs library | share of events with uncited source |
|---|---|---|---|---|---|---|
| blatant | 0 | 10 | 3.6e-12 | 1.6e-15 / 1.6e-15 | 4.2e-15 / 0.0e+00 / 2.0e-15 | 1.000 |
| blatant | 1 | 10 | 5.5e-12 | - | - | 1.000 |
| blatant | 2 | 10 | 7.3e-12 | - | - | 1.000 |
| moderate | 0 | 10 | 3.8e-12 | 2.2e-15 / 2.2e-15 | 1.8e-15 / 0.0e+00 / 1.8e-15 | 1.000 |
| moderate | 1 | 10 | 3.7e-12 | - | - | 1.000 |
| moderate | 2 | 10 | 7.3e-12 | - | - | 1.000 |
| subtle | 0 | 10 | 2.4e-12 | 1.6e-15 / 1.6e-15 | 6.7e-15 / 0.0e+00 / 1.8e-15 | 1.000 |
| subtle | 1 | 10 | 3.2e-12 | - | - | 1.000 |
| subtle | 2 | 10 | 5.5e-12 | - | - | 1.000 |
| clique | 0 | 10 | 5.7e-12 | 1.1e-15 / 1.1e-15 | 1.4e-14 / 0.0e+00 / 1.1e-15 | 1.000 |
| clique | 1 | 10 | 4.5e-12 | - | - | 1.000 |
| clique | 2 | 10 | 7.0e-12 | - | - | 1.000 |


Cross-checks covered every signal on 10 events per run (S6b/S6c on 4 to 10 of them, where v has at most 40
in-edges). GraphRicciCurvature ran on escalation subgraphs of 371 to 1,564 edges (radius 1) and 54,000 to 79,000
edges (radius 2, pruned before transport) on average; radius 1 and 2 give the same value, as Proposition 2 predicts.

## 5. Findings

1. **Propositions 1 and 2 hold exactly on the synthetic stream.** S1 has Spearman -1.000 with the cited paper's
   reference count, S2 +1.000 with the coupling share, in every run; GraphRicciCurvature agrees with the closed
   form at radius 1 and 2 to 2e-15. As single signals S1 reaches AUROC 0.52 to 0.58 and S2 0.51 to 0.73; the
   two-sided screen |F - median| is at chance (0.49 to 0.52). Added to any control set, neither changes
   logistic-regression AP by more than 0.003; with gradient boosting the largest mean change is +0.005 (S2,
   sd 0.018). The plan's ORC escalation therefore buys, at event time, exactly one set intersection.
2. **Recency is the dominant trivial signal.** Cited-paper age alone gives AUROC 0.965 to 0.991, because planted
   cartel members cite each other's papers inside a 1- to 4-year window. Restricting the test sample to recent
   cited papers (Table 2c) lowers age to 0.72 to 0.91 and removes most of the apparent power of signals that
   track the cited paper's citation count.
3. **Strong single curvature signals are mostly citation counts in disguise.** S5b (co-citation Forman, AUROC
   0.88 to 0.96) is 2 - deg_CC(v) by Proposition 3 and has Spearman -0.97 with in_deg_t(v); in the age-restricted
   sample its AUROC falls to 0.22 (blatant, orientation reversed) to 0.80 (subtle). S3b (reversed Forman, 0.77 to
   0.86) is a linear combination of two degree controls.
4. **The formulations with non-degree structure are the triangle-based ones.** Reversed ORC (S3a) has the best
   single-signal AP among curvature signals (0.44 / 0.15 / 0.15 / 0.22; found@100 of 75 / 30 / 22 / 56) and only
   55 to 62 % of its variance is explained by the tier-B controls. Undirected ORC (S4a) and augmented Forman (S4b) also
   respond to triangles. Over tier A they add AP on the clique variant (S4b +0.043 ± 0.021, S4a +0.028 ± 0.006,
   S3a +0.014 ± 0.001, positive in all three seeds) and a little for moderate cartels (S4b +0.010 ± 0.005); in the
   age-restricted sample they add up to +0.014 AUROC (logistic regression) and +0.018 (gradient boosting).
5. **Those gains are explained by a counting statistic.** With tier B, which adds the coupling share and the
   transitive count |refs(u) & citers_t(v)| to tier A, no single curvature signal changes logistic-regression AP
   by more than 0.014 in absolute value in any setting and none changes AUROC by more than 0.001. The largest
   consistent gains are S3a on subtle cartels (+0.008 ± 0.002 AP) and S4b on moderate cartels (+0.006 ± 0.004).
   All twelve event-time curvature signals together change AP by -0.014 to +0.002. This is what the theory leads
   one to expect: ORC on graphs is controlled by triangle counts (Jost & Liu, 2014), and augmented Forman is, by
   its formula, degrees plus three times the triangle count (Samal et al., 2018); in a citation graph whose citing
   paper is uncited, the triangles of (u, v) are exactly coupling and transitive pairs.
6. **Gradient boosting gives the same picture.** Over tier B its gains with a mean above 0.01 AP are S5a
   (+0.025 ± 0.015) and S6a (+0.011 ± 0.001) on subtle cartels, where boosting is the weaker learner (base AP 0.22
   against 0.44 for logistic regression), so they close part of a gap rather than add to the best model, and S4a
   (+0.015 ± 0.040) and S5b (+0.013 ± 0.024) on moderate cartels, whose spread exceeds the mean.
7. **Curvature dynamics carry little.** S6a (change of the mean plan-Forman around v) has AUROC 0.50 to 0.79 alone
   and no consistent gain over tier A or B. S6b and S6c are undefined for 21 to 28 % of test events (cited papers
   younger than the window, which includes most cartel targets) and are near or below chance alone (0.36 to
   0.55).
8. **Audit mode does not rescue the plan's signals.** At t + delta the citing paper has citers, so the plan's ORC
   is no longer the coupling share, but it still adds nothing beyond three audit counts (|delta AP| <= 0.006 with
   logistic regression; with gradient boosting at most +0.018 ± 0.024); the later co-citation count of (u, v)
   alone reaches AP 0.70 (blatant) and 0.71 (clique) on the sample. The plan's Forman at t + delta is still 96 to
   98 % explained by the controls, mostly by the cited paper's reference count.

## 6. Recommendation (PROVISIONAL: for Maor and the advisors)

On this synthetic evidence, the book's central claim, that directed Ricci curvature adds detection power beyond a
temporal model, is not supported at event time for the planned formulations, and alternative formulations add at
most 0.008 AP (logistic regression) once the triangle counts they rest on are available as plain features. This
is a negative result and should be reported as one. It is not yet a result about real data (section 7).

Recommended path (option A), with alternatives:

- **A. Report the reductions and keep curvature as an interpretable view (recommended).** State Propositions 1 to
  3 in the book. Compute S2 by its closed form (`variants.orc_directed_uncited`) and call the library only for
  events whose citing paper is already cited at t; the S2.3 escalation budget then drops to that share. Add the
  trivial controls (age, citation counts, coupling share, transitive count) as a named baseline in the ablation
  (book 4.2.1) next to the TGN, and report curvature's increment over that baseline, whatever its sign. Flag for
  `docs/score_components.md`: at event time s_geom's |z_F| scores the cited paper's reference count, and
  s_curvdyn's target asinh(F) is asinh(2 - refs(v)), so "curvature dynamics" becomes "reference-count surprise"
  (decision D7).
- **B. Change the formulation.** Replace the plan's directed orientation by a triangle-sensitive one (reversed ORC
  S3a, undirected ORC S4a or augmented Forman S4b), which does carry structure beyond degrees. The claim must then
  be tested against tier-B controls on real data; here the increment over tier B is at most 0.008 AP with
  logistic regression. Cost: S3a and S4a need an optimal-transport problem per event (about 5 ms per event for all
  per-event signals together in this study's vectorised path; about 20 ms per call through GraphRicciCurvature,
  mostly process forking).
- **C. Reframe the task as a retrospective audit.** Curvature becomes non-degenerate at t + delta, but NFR3 and
  the evaluation design must change, and on this data simple audit counts (later co-citation of the pair, later
  citations of the citing paper) do at least as well. Not recommended on this evidence.
- **D. Keep the plan and test on real data first.** Real citations may differ: same-day and preprint citations
  break the premise of Propositions 1 to 3 for some events, and real cartels may not cite recent papers. The
  first S4.0 measurements, needed under every option, are the share of events whose citing paper is cited at t,
  and this script's harness run on the real slice with the weak labels.

## 7. Limitations of this evidence

- **Synthetic data.** The generator plants cartels whose members cite each other's recent papers inside a short
  window; real coercive citation or citation stacking may look different (older targets, citations spread over
  many citing papers, journal-level rather than paper-level structure). The strength of the age control is a
  property of this generator and these presets, not a finding about real data.
- **The premise of Propositions 1 to 3 is exact here and approximate on real data.** The generator never produces
  a citation to a paper published on the same day or later. OpenAlex publication dates, online-first versions and
  preprints can make in_deg_t(u) > 0 for some citing papers; for those events the propositions do not apply and
  the plan's signals carry more than the reduced statistic. The share of such events must be measured on the
  S4.0 slice before the reductions are quoted for real data.
- **Ceiling effects.** For blatant cartels the controls reach AUROC close to 1 on the test sample, which leaves
  little room for any added signal; the moderate and subtle settings are the informative ones.
- **Sample sizes.** Subtle runs have about 100 test positives per seed; three seeds give a rough spread, not a
  confidence interval. AP and precision@100 are measured on a sample with subsampled negatives (section 3.2).
- **Choices that could change the picture**, not explored for budget reasons: alpha (0.5 only), the 365-day S6
  window, weighted variants (coupling strength as edge weight), ORC on the full coupling or co-citation graph
  (only Forman-type curvature was computed there), and other learners than logistic regression and gradient
  boosting.
- **Audit mode** is evaluated on the same synthetic cartels, whose later members cite earlier members by
  construction; that is exactly the information audit mode reads, so its numbers are an upper bound for how
  visible such a cartel becomes in hindsight.

## 8. Reproduce

```bash
uv venv -p python3.12 /home/user/venvs/w2c-study
uv pip install -p /home/user/venvs/w2c-study/bin/python GraphRicciCurvature==0.6.1 POT==0.9.6 networkit==11.2.1 \
    "scipy<=1.13.1" scikit-learn pandas pyarrow pyyaml networkx pytest
PYTHONPATH=src nice -n 10 /home/user/venvs/w2c-study/bin/python scripts/curvature_study.py      # all numbers
PYTHONPATH=src python scripts/curvature_study.py --tables results/curvature_study.json         # the tables above
PYTHONPATH=src python -m pytest -q tests/test_curvature_variants.py
```

The full run took 1,352 s wall time and 1,342 s CPU on one core of the shared container (peak memory 795 MB);
about 5 ms per sampled event go to the per-event signals. Without GraphRicciCurvature the study runs and skips the
library block; without POT the transport problems fall
back to scipy's exact LP (slower, same values).

## 9. References

Jost, J., & Liu, S. (2014). Ollivier's Ricci curvature, local clustering and curvature-dimension inequalities on
graphs. *Discrete & Computational Geometry, 51*(2), 300-322. https://doi.org/10.1007/s00454-013-9558-1

Kessler, M. M. (1963). Bibliographic coupling between scientific papers. *American Documentation, 14*(1), 10-25.
https://doi.org/10.1002/asi.5090140103

Kojaku, S., Livan, G., & Masuda, N. (2021). Detecting anomalous citation groups in journal networks. *Scientific
Reports, 11*, 14524. https://doi.org/10.1038/s41598-021-93572-3

Ni, C.-C., Lin, Y.-Y., Gao, J., Gu, X. D., & Saucan, E. (2015). Ricci curvature of the Internet topology. In *2015
IEEE Conference on Computer Communications (INFOCOM)* (pp. 2758-2766). https://doi.org/10.1109/INFOCOM.2015.7218668

Ollivier, Y. (2009). Ricci curvature of Markov chains on metric spaces. *Journal of Functional Analysis, 256*(3),
810-864.

Samal, A., Sreejith, R. P., Gu, J., Liu, S., Saucan, E., & Jost, J. (2018). Comparative analysis of two
discretizations of Ricci curvature for complex networks. *Scientific Reports, 8*, 8650.
https://doi.org/10.1038/s41598-018-27001-3

Saucan, E., Sreejith, R. P., Vivek-Ananth, R. P., Jost, J., & Samal, A. (2019). Discrete Ricci curvatures for
directed networks. *Chaos, Solitons & Fractals, 118*, 347-360. https://doi.org/10.1016/j.chaos.2018.11.031

Small, H. (1973). Co-citation in the scientific literature: A new measure of the relationship between two
documents. *Journal of the American Society for Information Science, 24*(4), 265-269.
https://doi.org/10.1002/asi.4630240406

Verification: bibliographic details of Jost and Liu, Kessler, Kojaku et al., Ni et al., Samal et al., Saucan et
al. and Small were checked on 2026-10-31 against publisher, PubMed Central, RePEc, DBLP or arXiv records found by
web search (the publisher pages themselves were not opened). Ollivier (2009) is taken from
`docs/score_components.md` and was not re-checked here. The GraphRicciCurvature conventions (measures, Dirac
fallback, augmented Forman formula) were read from the installed 0.6.1 source and confirmed numerically in the
tests.

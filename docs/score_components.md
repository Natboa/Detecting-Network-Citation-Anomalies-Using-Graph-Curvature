# Score components: definitions (task S3.1)

> **PROVISIONAL — awaiting Maor's review (S3.1 gate).**
> How to review: section 1 lists every decision in this note as one table row with its default and the main
> alternative. Mark each row "approve" or "amend: ...". Sections 2 to 10 give the detail behind each row. The
> code (`src/citecurv/scoring/components.py`) implements this note exactly, and each row maps to one function
> and its tests (`tests/test_components.py`), so an amendment is a local change.

Every component is a raw number per citation event, oriented so that **higher means more anomalous**. The
combiner (task S3.2, `citecurv.scoring.combiner`) turns the raw values into z-scores fitted on training rows and
adds them with calibrated weights (section 9). Nothing in this note uses labels.

## 1. Decisions to approve

| # | Decision | Default in the code | Main alternative | Reason for the default |
|---|----------|---------------------|------------------|------------------------|
| D1 | Reference cell of an event | (field of the citing paper src, calendar year of t) | field of the cited paper dst; (src field, dst field) pairs | The event is the citing paper's act on its publication date, so the cell reads "citations made by field f papers in year y". Pair cells would be too sparse. |
| D2 | Reference window | Expanding, over events with ts < t (strict); fallback (field, year) -> (field, all earlier years) -> (all fields); a level needs >= `min_cell` = 30 earlier values and sd > 0 | Rolling 365-day window instead of the calendar year | Matches the plan's (field, year) cells and the S3.4 fairness gate. The strict rule makes every event at time t see the same reference regardless of tie order (section 2). |
| D3 | Normalisation inside a cell | Mean and sample standard deviation | Median and MAD | A spike holding more than half of a cell (common for Forman, section 4) makes MAD zero. Mean/sd also runs in O(1) memory per cell. |
| D4 | s_embed | -ln p_link from the TGN (mode `tgn`); fallback mode `static_cosine` for rehearsal before the TGN exists | Field-year z of -ln p_link | The decoder already conditions on both endpoints' states; the fallback lets the pipeline run end to end now. |
| D5 | How Forman and ORC combine in s_geom | max(\|z_forman\|, \|z_orc\|) for escalated events, \|z_forman\| otherwise | Root mean square of the two | The ORC reference holds only escalated (already extreme) events, so a typical ORC value there is not evidence of normality. With max, ORC can only add evidence, and `full` minus `forman_only` isolates what ORC adds. |
| D6 | s_temp statistic | Field-year z of x = ln((n_recent + 1) / (n_before + 1)) + ln(T_before / W): the cited paper's recent citation rate relative to its own earlier rate; W = 90 days | Raw burst count, or gap since the previous citation | Raw counts and gaps measure popularity: a highly cited paper always has many recent citations and short gaps. The ratio compares a paper with its own history. |
| D7 | s_curvdyn regressor | Ridge regression on PCA-reduced endpoint vectors (32 dimensions per endpoint, 97 features), target asinh(Forman), fitted on train rows only; score = \|residual\| / train RMSE | Small MLP (closer to CurvGAD) | Closed form: deterministic to 1e-6 without GPU nondeterminism, cheap at 3M events. An MLP can replace it behind the same interface. |
| D8 | Cold start | cold_start when the cited paper has fewer than k = 3 citation events with ts < t: its 1st, 2nd and 3rd citations are flagged, the 4th is the first scored | Score the k-th citation (rule "fewer than k - 1 prior") | Follows the data contract and L19 ("papers enter scoring after k events"). See the off-by-one note in section 7. |
| D9 | Cold-start rows in the components frame | Components are still computed; the row carries `cold_start = True`; the combiner excludes it from totals, statistics and metrics | Write NaN components for cold-start rows | S2.4 needs component values on low-history events to pick k (stability against prior-event count). |
| D10 | Missing or disabled components | NaN, never a filler value; the combiner treats NaN as z = 0 and gives a disabled component weight 0 | Impute a cell mean | NaN keeps "no evidence" distinct from "normal evidence" in the trace. |
| D11 | `no_temporal` ablation | Changes the encoder only (DyGLib memory / time encoding off); s_temp stays on | Also turn s_temp off | The plan defines the cell as a static encoder. Turning s_temp off as well would test a different question (both temporal parts at once). Open question for Maor. |
| D12 | z-statistics for model-based components | Fitted on train rows (contract) | Fit the s_embed (TGN) and s_curvdyn statistics on validation rows | On train rows both models are scored in-sample, so their train distributions sit slightly low and test z-scores slightly high (section 9). Open question for Maor. |

## 2. Notation and shared rules

An event e is a citation u -> v at time t (days since 1970-01-01, t = publication date of the citing paper u).
Events are processed in canonical order (t, event_id).

**Field and year.** field(u) is `works.field` when present, else the first entry of `works.concepts`, else
`"unknown"`. The plan speaks of (concept, year) cells. A paper carries many OpenAlex concepts at several levels,
so the first listed concept can be fine-grained and make cells small; S4.0 should store a top-level `field`
column so cells stay large. year(t) is the calendar year of t.

**Information rules.**

- *Per-event inputs* (counts, curvature, TGN outputs) may use graph state with ts <= t except the event's own row,
  as the data contract allows.
- *Reference statistics* (cell means and standard deviations) use only events with ts < t. Many events share
  a timestamp, because all references of one paper share its publication date. Under ts <= t the reference of
  an event would depend on how ties are broken (event_id order), and "simultaneous" events would inform each
  other. Under ts < t all events at time t see the identical reference whatever their order. It is simpler and
  leaves no room for a within-timestamp leak.

**Reference statistic.** For a raw value x(e) that is defined on some events (its "valid" events), let R_L(e) be
the values x(e') of valid events e' with t' < t that share e's key at level L:

- L0: (field(u), year(t)); L1: field(u); L2: all events.

The first level with |R_L| >= max(min_cell, 2) and sample sd > 0 gives mean m and sd s, and
z(e) = (x(e) - m) / s. If no level qualifies (only the first events of the whole stream), z = NaN. The level
used, n, m and s are recorded per row (section 10).

The window is expanding: a reference never drops old events. L0 restarts every 1 January, so early-January
events fall back to L1 until the new year's cell holds `min_cell` events.

**Cost.** One pass in time order with per-cell running sums (shifted sums, so large magnitudes do not cancel).
The vectorised path is O(E log E), dominated by sorting. Synthetic benchmark in this container (random events,
not real data): 1,000,000 events with 20 fields, 2 % escalation and 64-dimensional vectors scored in 13 s with
0.87 GB peak memory. Extrapolated linearly (not measured), 3M events take on the order of a minute and about
3 GB, which fits a Colab high-RAM runtime.

## 3. s_embed: embedding surprise

**Formula (mode `tgn`).** s_embed(e) = -ln clip(p_link(e), eps, 1 - eps), eps = 1e-7, where p_link is the
TGN decoder's probability of the observed edge (u, v) at t (Rossi et al., 2020; DyGLib, Yu et al., 2023),
computed from memory *before* the batch containing e updates memory (TGN output contract).

**Reference and normalisation.** None inside the component. The decoder already conditions on the states of both
endpoints; the combiner z-normalises. If p_link levels turn out to differ by field, the amendment is to apply
the section 2 cell z to -ln p_link.

**Leak argument.** Memory before e's batch holds only events from earlier batches, and batches follow time
order, so the state uses events with ts <= t. The TGN stage must also guarantee that temporal neighbour
sampling for e uses only interactions before t; the S1.7 leak assertion covers the memory side. The decoder weights are
trained on the train period, which precedes validation and test, so test p_link values are out of sample.

**Failure modes.**

- Decoder probabilities from negative-sampling training are not calibrated base rates; they are only used for
  ranking.
- Papers with little history have little memory, so their p_link can be low and s_embed high for that reason
  alone. Cold start removes the weakest cases; S2.4 should check s_embed against prior-event count.
- Train rows are scored in-sample (the decoder saw them), so their s_embed is lower than for comparable test
  rows; see D12.
- An event without a p_link row gets NaN.

**Fallback mode `static_cosine` (rehearsal only).** c(e) = cos(x_u, x_v) of the SPECTER2 vectors. With the
cell reference of earlier events' cosines (section 2), s_embed(e) = (m - c) / s: positive when the two papers are
less similar than usual for citations made in that field and year. Values are signed and not clipped. An event
whose endpoints lack vectors gets NaN. This mode lets the full pipeline run before the TGN is wired; reported
results use mode `tgn` unless a run says otherwise.

*Leak note for both modes.* A SPECTER2 vector depends only on the paper's own title and abstract, which exist from
its publication date. The encoder itself, however, was trained with a citation-informed objective (SPECTER,
Cohan et al., 2020; SPECTER2, Singh et al., 2023). If its training citations overlap our test period, the vectors
may carry information about test-period links. The time split cannot remove this, and it also affects the TGN,
whose node features are SPECTER2 vectors (L2). The book should state it as a limitation (to confirm against the
SPECTER2 training-data description at S1.6).

**What an investigator sees.** `p_link` and s_embed (tgn), or `cosine` with the cell mean, sd, n and level used
(static mode). Reading: "the model gave this citation probability 0.0004 (s_embed 7.8)".

## 4. s_geom: geometric anomaly

**Inputs** (curvature cache, produced by `citecurv.curvature`). F(e): unweighted directed Forman screen
F = 2 - in_deg_t(u) - out_deg_t(v) on the graph at t, own row excluded (Saucan et al., 2019). kappa(e): directed
Ollivier-Ricci curvature (Ollivier, 2009) of e on its radius-r subgraph at t, present only when the screen escalated e.

**Formula.**

- z_F(e) = (F - m_F) / s_F against the cell reference of earlier events' F.
- z_O(e) = (kappa - m_O) / s_O against the cell reference of earlier *escalated* events' kappa. ORC is only
  measured on escalated events, so its reference must be escalated events too.
- s_geom(e) = |z_F| for a non-escalated event, or an escalated event whose ORC failed;
  s_geom(e) = max(|z_F|, |z_O|) for an escalated event with kappa (D5).

Both tails count (|z|): very negative F means high-degree endpoints, F near 2 means an unusually sparse
neighbourhood.

**What the screen measures in practice.** This follows from the formula and should be checked in the S2.2
distribution report. The citing paper u is published at t, so in_deg_t(u) (citations u has received by its own
publication date) is usually zero or small. Then F is close to 2 - out_deg_t(v), that is, mostly the number of
in-slice references of the cited paper. A cited paper outside the slice has out_deg 0 and F near 2, which creates
a spike at the top of the distribution. The screen's s_geom therefore flags citations to papers with an unusual
in-slice reference count, plus citing papers already cited at publication. Most of the geometric signal is
expected from ORC (escalated events) and from s_curvdyn.

**Leak argument.** F and kappa use edges with ts <= t minus the event's own row (EventStore, S2.1; curvature cache
contract). The references use ts < t.

**Failure modes.**

- Heavy tails and the F spike make |z| non-Gaussian; it is a ranking statistic, not a p-value.
- Escalation uses global percentile thresholds (S2.3) while z is cell-relative, so an escalated event can have a
  modest |z_F| inside its own cell. Harmless: s_geom then rests on z_O.
- Publication dates recorded as 1 January (year-only dates, if present in the data) put many events on one day;
  they all fall back to L1 (section 2).
- Small cells use a coarser level; the level is recorded per row.

**What an investigator sees.** F, z_F, the Forman reference (level, n, mean, sd), and for escalated events kappa,
z_O and the ORC reference. Reading: "F = -182 against Medicine-2019 mean -31 (sd 37, n = 12,402): |z| = 4.1; not
escalated" (illustrative values, not measurements).

## 5. s_temp: temporal anomaly

**Statistic.** For e = u -> v at t, counting citation events of the cited paper v:

- n_recent = #{citations of v with t - W < ts <= t} - 1 (the event itself is removed; other citations on the
  same day count);
- n_before = #{citations of v with ts <= t - W};
- T_before = max(t - W - t_first(v), W) if n_before > 0, else W, where t_first(v) is v's first observed citation;
- x(e) = ln((n_recent + 1) / W) - ln((n_before + 1) / T_before) = ln((n_recent + 1) / (n_before + 1)) + ln(T_before / W).

x is the log ratio of v's citation rate in the trailing W days to its rate before that window, with +1
pseudo-counts so that x stays finite and small counts are shrunk. s_temp(e) = (x - m) / s against the cell
reference of earlier events' x (signed).

**Why this statistic (D6).** A raw burst count or the gap since the previous citation mostly measures popularity:
a steadily highly cited paper has many citations in any window and short gaps, so it would rank high on every
citation. The ratio compares v with its own history: a steadily popular paper scores near 0 and a sudden jump
scores high. The field-year reference then removes field differences in how fast papers gather citations. A
single gap is also noisy: references of one journal issue share a date, so gaps of 0 are ordinary.

**Window.** W = 90 days (`window_days`): wide enough to absorb month-level imprecision in publication dates,
narrow enough that an issue-sized burst stands out against a multi-year history. Provisional; S3.4 should check
30 and 365 as sensitivity values.

**Same-timestamp rule.** The counts use ts <= t minus the own row (contract policy), because the main burst
pattern is many citations on one day; under ts < t those citations would never see each other. The reference
statistics still use ts < t.

**Leak argument.** Only citations of v with ts <= t enter, the event's own row is removed, and t_first(v) <= t.
The reference uses ts < t.

**Failure modes.**

- Date granularity: if some dates are year-only, same-day "bursts" appear on 1 January. The cell reference
  absorbs part of this, since other events of that day sit in the same cell.
- New papers: once past cold start, a paper without history has x = ln(n_recent + 1), so any paper cited quickly
  after publication scores high. The field-year reference calibrates this only partly.
- t_first is the first *observed* citation, not the publication date; citations from outside the slice are
  invisible, so exposure is underestimated for some papers.
- s_temp cannot tell supportive from critical citations: a paper that becomes notorious also receives a burst.

**What an investigator sees.** n_recent, n_before, T_before, x and the reference. Worked example from the unit-test
fixture (synthetic data): a paper with 6 citations spread over days 0 to 500 receives 10 citations on day 1000.
Each of those events has n_recent = 9, n_before = 6, T_before = 910, so x = ln(10/7) + ln(910/90) = 2.67, against
a field-year reference of mean 0.32 (sd 0.50, n = 29): s_temp = 4.66. The same 10 citations at 100-day spacing
score between -1.3 and -0.5.

## 6. s_curvdyn: curvature dynamics

**Idea.** Borrowed from CurvGAD (Grover et al., 2025), which scores anomalies partly by how poorly edge curvature
can be reconstructed: a citation whose local geometry is inconsistent with what the two papers' representations
predict is suspicious. We keep the idea and use a much smaller model.

**Model.**

- Target y(e) = asinh(F(e)). asinh is linear near 0 and logarithmic in the tails, so the heavy negative tail of F
  does not dominate the squared loss.
- Endpoint vectors a, b: TGN z_src, z_dst at t (contract `tgn_event_embeddings.npz`, computed before memory
  update) when available; static SPECTER2 vectors of u and v otherwise. `inputs["curv_source"]` records which.
- PCA basis P with r = 32 components (`pca_dim`), fitted on endpoint vectors of train events (at most 20,000
  vectors, seeded sample).
- Features phi = [P(a), P(b), P(a) * P(b), cos(a, b)] (3r + 1 = 97), standardised with train mean and sd.
- Ridge regression, alpha = 1.0 (`ridge_alpha`), closed form, fitted on train rows (seeded subsample of at most
  500,000 rows, `max_fit_rows`).
- s_curvdyn(e) = |y(e) - y_hat(e)| / RMSE_train.

In the `ollivier_only` cell the target is asinh(kappa) on escalated events, and the model is fitted and applied
on escalated events only.

**Leak argument.** Features come from representations at t (TGN state before e's memory update, or text). The
target is the screen value at t. The PCA basis, standardisation, coefficients and RMSE all come from train rows,
and the train period precedes validation and test. The unit test changes every test row's target and vectors and
shows the fitted model is bit-for-bit unchanged. Train-row scores are in-sample, but with 97 features and 10^5
or more training rows the optimism is small.

**Failure modes.**

- Static vectors carry no degree information, so with them the residual mostly reads "this paper's reference count
  is unusual for its topic". TGN vectors carry activity through memory and should be more informative.
- If the representations predict F almost exactly (memory encoding degree), residuals shrink and the component
  carries little. Check R^2 on validation in S3.4.
- The graph grows over time, so degrees and F drift; residuals can drift with t. Check the residual mean by year on
  validation; if it drifts, standardise the residual within its cell (section 2) instead of by the global RMSE.
- Events without endpoint vectors, or without a target, get NaN. Without a train mask (and no fitted model passed
  in) the component is NaN and the code warns.

**What an investigator sees.** Observed and predicted asinh(curvature), the training RMSE and the source of the
vectors. Reading: "from the two papers' representations the model expected F of about -8; observed F = -190;
residual = 3.1 training RMSEs" (illustrative values, not measurements).

## 7. Cold start (L19, S2.4)

cold_start(e) = n_prior(v, t) < k, with n_prior = #{citations of v with ts < t} and k = `scoring.cold_start_k`
(provisional 3, finalised in S2.4). Only the cited paper is checked: the citing paper is new at t by definition.

With citations on distinct dates, a paper's 1st, 2nd and 3rd citations (0, 1, 2 earlier citations) are cold and
the 4th is the first scored. **Off-by-one note:** the S2.4 verify line says "events 1..k-1 carry the flag, event k
does not". That agrees with this rule only if events are counted from 0 (event j has j earlier citations). The
data contract ("fewer than k prior events") and L19 ("papers enter scoring after k events") give the rule above,
and the unit test fixes it. If Maor prefers the k-th citation to be scored, the rule becomes n_prior < k - 1.

The count is strict (ts < t), so all same-day citations of a new paper see 0 earlier citations and are cold: a
burst on a paper's first day is not scored. Those citations still count in later events' n_recent, so a burst that
continues is seen.

Cold-start rows keep their component values (D9; trace column `in_n_prior_dst` supports the S2.4 stability plot).
They are excluded from totals, ranking, z-statistics, calibration and metrics, and included in TGN training and
memory updates.

## 8. Ablation semantics (book 4.2.1, plan S4.3)

| Variant | s_embed | s_geom | s_temp | s_curvdyn | Other stages |
|---------|---------|--------|--------|-----------|--------------|
| `full` | on | max(\|z_F\|, \|z_O\|) | on | target F | Forman screen + ORC escalation |
| `no_curvature` | on | off (NaN) | on | off (NaN) | TGN trained without curvature features; curvature stage not run |
| `forman_only` | on | \|z_F\|, ORC ignored | on | target F | no escalation |
| `ollivier_only` | on | \|z_O\| on escalated events, NaN elsewhere | on | target kappa on escalated events, NaN elsewhere | escalation set = `matched_random_escalation(curvature, seed)`; Forman values are never read |
| `no_temporal` | on (from the ablated encoder) | as `full` | on (D11) | as `full` | DyGLib memory / time encoding off |

`ollivier_only`: the screen is replaced by a uniform random sample of events of the same size as the screen's
escalated set (global rate, seeded), so the cell spends the same ORC budget while no Forman signal chooses the
events. Most events then have no kappa, so s_geom and s_curvdyn are NaN for them and contribute nothing; that is
the fair reading of "Ollivier only at the same budget". A test checks that changing every Forman value leaves this
cell's output unchanged. Disabled components are all-NaN and listed in `attrs["disabled_components"]`.

## 9. How the combiner consumes the components

- Input: the components frame (`event_id, src_id, dst_id, t, s_embed, s_geom, s_temp, s_curvdyn, cold_start`).
  Cold-start rows are excluded from everything below and get `total = NaN`.
- ZStats: per component, mean and sd over train rows that are not cold-start and not NaN; frozen and applied
  unchanged to validation and test (never refitted on test).
- z_c = (s_c - mu_c) / sigma_c. A NaN component contributes z = 0 (no evidence either way); the number of
  components present per row should be kept for the investigator.
- A disabled component (all-NaN in the training rows) gets weight 0, and the remaining weights are renormalised
  on the simplex. total = sum_c w_c * z_c.
- Shapes differ: s_geom and s_curvdyn are folded (>= 0), s_temp and the static s_embed are signed, the TGN s_embed
  is >= 0 with a long right tail. z-normalisation is enough for ranking; no transform is applied here.
- D12: s_embed (TGN) and s_curvdyn are in-sample on train rows, so their train mean sits slightly low and test z
  slightly high for every test row alike. This shifts the total but does not reorder events within a single
  component. Fitting these two ZStats on validation rows (pre-cutoff, out of sample for both models) would remove
  the shift.

## 10. Traceability (NFR10) and the investigator view

- `score_events(..., with_inputs=True)` adds `in_*` columns: `in_field, in_year, in_p_link, in_cosine, in_forman,
  in_orc, in_escalated, in_z_forman, in_z_orc, in_temp_x, in_n_recent, in_n_before, in_exposure_days,
  in_n_prior_dst, in_curv_obs, in_curv_pred`, and for each reference (`embed`, `geom`, `orc`, `temp`) the level
  used (`0` field-year, `1` field, `2` global, `-1` none) and its n, mean and sd
  (`in_geom_level, in_geom_ref_n, in_geom_ref_mean, in_geom_ref_sd`, ...).
- `score_event(event_id, build_context(...))` returns a `ScoreBreakdown` (four components, `total=None` until the
  combiner fills it, `cold_start`, `inputs`) whose `inputs` dict uses the same names plus ids, variant, embedding
  mode, k and the s_curvdyn source, target and RMSE. A test checks that this single-event path agrees with the
  vectorised path on every event of a random stream, for every variant and both embedding modes.
- Since scores are deterministic for fixed data, config and seed (NFR4), any row of `scores.parquet` can be
  regenerated with its inputs from the cached curvature, TGN outputs and embeddings plus `config_hash` and `seed`.
  The S4.7 review packet should store the `in_*` columns for its 50 rows.
- Single-event cost (synthetic benchmark in this container, 200,000 random events): median 0.42 ms, 95th
  percentile 0.58 ms per event for the components alone, excluding the TGN memory read and curvature-at-t. S2.5
  measures the end-to-end budget on real data.

## 11. Configuration keys

Proposed under `scoring:` in `configs/default.yaml` (not yet merged):

```yaml
scoring:
  cold_start_k: 3              # provisional (L19), finalised in S2.4
  components:
    embed_mode: auto           # auto | tgn | static_cosine (static_cosine = rehearsal before the TGN exists)
    window_days: 90            # s_temp trailing window W in days; provisional, sensitivity 30/365 in S3.4
    min_cell: 30               # earlier values a reference cell needs before it is used (else coarser cell)
    curvdyn_pca_dim: 32        # PCA dimensions per endpoint vector for the s_curvdyn regressor
    curvdyn_ridge_alpha: 1.0   # ridge penalty on standardised features
    curvdyn_max_fit_rows: 500000  # seeded subsample cap of train rows for the s_curvdyn fit
    p_link_eps: 1.0e-7         # p_link clipped to [eps, 1 - eps] before -ln
```

## References

Bibliographic details were written in an environment without access to publisher pages; check them before they
enter the book.

- Cohan, A., Feldman, S., Beltagy, I., Downey, D., & Weld, D. S. (2020). SPECTER: Document-level representation
  learning using citation-informed transformers. In *Proceedings of the 58th Annual Meeting of the Association for
  Computational Linguistics* (pp. 2270–2282). Association for Computational Linguistics.
- Grover, K., Gordon, G. J., & Faloutsos, C. (2025). CurvGAD: Leveraging curvature for enhanced graph anomaly
  detection. In *Proceedings of the 42nd International Conference on Machine Learning* (PMLR 267).
- Ollivier, Y. (2009). Ricci curvature of Markov chains on metric spaces. *Journal of Functional Analysis, 256*(3),
  810–864.
- Rossi, E., Chamberlain, B., Frasca, F., Eynard, D., Monti, F., & Bronstein, M. (2020). *Temporal graph networks for
  deep learning on dynamic graphs* (arXiv:2006.10637). arXiv.
- Saucan, E., Sreejith, R. P., Vivek-Ananth, R. P., Jost, J., & Samal, A. (2019). Discrete Ricci curvatures for
  directed networks. *Chaos, Solitons & Fractals, 118*, 347–360.
- Singh, A., D'Arcy, M., Cohan, A., Downey, D., & Feldman, S. (2023). SciRepEval: A multi-format benchmark for
  scientific document representations. In *Proceedings of the 2023 Conference on Empirical Methods in Natural
  Language Processing*. Association for Computational Linguistics.
- Yu, L., Sun, L., Du, B., & Lv, W. (2023). Towards better dynamic graph learning: New architecture and unified
  library. In *Advances in Neural Information Processing Systems 36*.

## Update 2026-10-31: findings from the curvature formulation study

This section is appended; the text above is unchanged. It records what `docs/curvature_formulation_study.md`
(synthetic data only) implies for s_geom and s_curvdyn. No decision is taken here.

**Premise.** At event time the citing paper u is usually uncited: a paper published at t can be cited only by
papers published later, so in_deg_t(u) = 0. This holds for every event of the synthetic stream. On real data it
fails for same-day citations, preprint or online-first dates and date errors; the share of such events has not
been measured yet (first S4.0 measurement under every option below).

**Reductions** (study, section 2; refs(v) = the references of v present in the event table, so 0 for a cited
paper outside the slice):

- Forman screen (Proposition 1): F(e) = 2 - in_deg_t(u) - out_deg_t(v) = 2 - |refs(v)|. Section 4 above already
  expected F to be close to 2 - out_deg_t(v); under the premise the identity is exact. The study asserts it on
  every event and reports Spearman -1.000 between F and |refs(v)|.
- Directed Ollivier-Ricci curvature as planned (Proposition 2): kappa(u, v) = -(1 - alpha)(1 - s / k), where v
  cites k > 0 papers of which u also cites s, and kappa = 0 when k = 0. With coupling share = s / k (share = 1
  when k = 0), kappa = -(1 - alpha)(1 - coupling share): a strictly increasing function of the share of v's
  references that u also cites (bibliographic coupling; Kessler, 1963). GraphRicciCurvature 0.6.1 at radius 1
  and 2 agrees with this closed form to 2e-15 on every checked event; `citecurv.curvature.variants.
  orc_directed_uncited` computes it.

**What this means for the components as defined above.**

- s_geom (section 4, D5). For a non-escalated event, |z_F| measures how unusual the cited paper's in-slice
  reference count is compared with earlier citations of the same (field, year) cell. For an escalated event,
  |z_O| measures how unusual the coupling share of u and v is compared with earlier escalated events. Both are
  reference-list statistics; neither uses the citation structure around u.
- s_curvdyn (section 6, D7). The target asinh(F) equals asinh(2 - |refs(v)|), so the component scores how badly
  the endpoint representations predict the cited paper's reference count ("reference-count surprise" rather
  than curvature dynamics). In the `ollivier_only` cell the target is a function of the coupling share.
- s_embed and s_temp do not depend on curvature and are unaffected.
- In the study, adding the two plan signals to the trivial controls (cited-paper age, citation counts) changed
  logistic-regression AP by at most 0.003 in every synthetic setting.

**Open decisions for Maor** (options from the study's section 6, all PROVISIONAL; none is adopted here):

1. Option A (the study's recommendation): state Propositions 1 to 3 in the book; compute the planned ORC by its
   closed form and call the library only for events whose citing paper is already cited at t; add the trivial
   controls (age, citation counts, coupling share, the "u cites a paper that already cites v" count) as a named
   baseline in the ablation; describe s_geom and s_curvdyn by what they measure (the reductions above).
2. Option B: change the formulation to a triangle-sensitive variant (reversed ORC, undirected ORC or augmented
   Forman) and test it against the full control set on real data. In the study these added at most 0.008 AP
   once the counts they rest on were controls.
3. Option C: reframe the task as a retrospective audit (curvature at t + delta). This changes NFR3 and the
   evaluation design; the study does not recommend it on its evidence.
4. Option D: keep the plan unchanged and test on the real slice first (measure the share of events whose citing
   paper is cited at t, and rerun the study's harness with the weak labels).
5. Under any option: whether D5 (how Forman and ORC combine in s_geom) and D7 (the s_curvdyn target) keep their
   current definitions, and how the book names the two components.

Reference for this update: Kessler, M. M. (1963). Bibliographic coupling between scientific papers. *American
Documentation, 14*(1), 10-25.

# Results explorer

`citecurv.viz` writes one self-contained HTML page over the pipeline outputs. The page needs no server and makes no
network request: data, CSS and JavaScript are all inline. It opens from disk in any current browser and shows inside
a Colab or Jupyter cell. Python computes every number the page shows (rank order, z-scores and contributions,
reference histograms, PR/ROC curves, the D4 heatmap, rollups, local subgraphs with seeded layouts). The JavaScript
only draws.

It serves FR7 (named score breakdown), FR8 (paper and author rollups), FR9 (filter and export), NFR7 and NFR10
(every flag traceable to its inputs), the L15 explainability scope (named breakdown only), and the D4 flag-rate
sanity check (S3.4a).

## Build it

Demo page from synthetic data (written to `data/explorer/explorer_demo.html`, which is gitignored):

```bash
PYTHONPATH=src python scripts/build_explorer.py --demo
```

From pipeline outputs (only `--scores` is required; every other input switches on a panel):

```bash
PYTHONPATH=src python scripts/build_explorer.py \
  --scores data/runs/full_0/scores.parquet --events data/events.parquet --works data/works.parquet \
  --labels data/event_labels.parquet --curvature data/curvature_cache.parquet \
  --metrics runs/full_0/metrics.json --weights runs/full_0/weights.json --zstats runs/full_0/zstats.json \
  --table "Main comparison=results/main_table.csv" --data-note "" --out data/explorer/full_0.html
```

From Python:

```python
from citecurv.viz import build_explorer, show

path = build_explorer("data/explorer/full_0.html", scores, events=events, works=works, labels=labels,
                      curvature=curvature, weights=weights, zstats=zstats, metrics=metrics,
                      data_note="", top_n=500, radius=1, seed=0)
```

In Colab or Jupyter, `show(path)` displays the page in an `<iframe srcdoc>` (default height 900 px), which isolates
its styles and ids from the notebook. `IPython.display.HTML(open(path).read())` also works: the page scopes its
styles to its own root element. In Colab, open a built file in a full tab by downloading it from the file browser.

Packaging note: the page template, CSS and JavaScript live in `src/citecurv/viz/assets/`. A wheel built from
`pyproject.toml` (for example `pip install git+https://...` in Colab) only includes them once
`[tool.setuptools.package-data] citecurv = ["viz/assets/*"]` is declared. Without it, `build_explorer` raises a
`FileNotFoundError` that names this fix. Editable installs and `PYTHONPATH=src` work as they are.

## Inputs

All inputs follow `CONTRACTS.md`. `scores` must hold one run (one `model_variant`, one `seed`); the function
refuses a mixed frame.

| Argument | Contract | Enables |
|---|---|---|
| `scores` (required) | scores file: `event_id, src_id, dst_id, t, s_*, total, cold_start, model_variant, seed, config_hash` | ranking, histograms, rollups, detail breakdown |
| `events` | `event_id, src_id, dst_id, t` | local subgraph at t, timeline of the cited paper |
| `works` | `work_id, title, pub_date, venue_id, concepts`, optional `field` | titles, field and venue filters, D4 heatmap |
| `authorships` | `work_id, author_id` | author rollup |
| `labels` | `event_id, label` | PR/ROC, precision@k, label markers, positives filter |
| `curvature` | curvature cache (`forman`, `orc`, `escalated`, ...) | cached Forman/ORC of the scored edge, ORC edge colours |
| `components` | components frame | its extra input columns appear in the NFR10 inputs table |
| `weights` | dict, Series, 4-sequence, or object with `.weights` | exact `w_c * z_c` contributions |
| `zstats` | `{c: (mean, std)}`, `{c: {"mean", "std"}}`, `{"mean": {...}, "std": {...}}`, DataFrame, or an object with `mean`/`std` attributes | exact z-scores |
| `metrics` | dict, for example `ranking_metrics` output; nested dicts are flattened | run metrics table |
| `tables` | `{name: DataFrame}` | comparison and ablation tables |
| `cartels` | members DataFrame `cartel_id, work_id, role, level`, or `{"members": df, "event_ids": [...]}` | cartel view |

When `weights` or `zstats` are missing, the page falls back to equal weights and to z statistics computed from the
scores file itself, and says so in a "Read before interpreting" card. When the contributions do not add up to the
stored total, the detail view shows the gap and treats the stored total as authoritative.

Other keyword arguments: `top_n` (events with a precomputed detail view, default 500), `radius` (1 or 2 hops of
graph context), `max_nodes` (drawing cap, default 40), `flag_share` (share of ranked events counted as flags,
default 0.01), `n_rollup` (papers and authors kept, default 200), `seed` (layout seed), `title`, `data_note`.

`data_note` is shown in a banner. Any note containing "synthetic" gets the prominent synthetic-data banner, and a
page built with `cartels` is always marked as containing synthetic injection. The function default is
"synthetic demo data", so a real-data page must pass its own note (an empty string shows no banner).

## Views

**Overview.** Dataset tiles (events, papers, time span, cold-start share, ranked events, flag threshold). Histogram of
the total score with the threshold marked. When labels exist: AUROC, average precision, precision@10/100/1000, and
PR and ROC curves of the total score. An optional overlay shows each component used alone as a ranking. The D4 check
is a field by year heatmap of where the top 1 % of ranked events fall. Cells are coloured by their share of all flags
(or by the within-cell flag rate relative to the overall rate). Any cell above 50 % is outlined and the check reads
FAIL. Then the combiner inputs (weights, frozen z statistics), the run metrics, the given tables, and a list of
inputs that were not provided.

**Ranked list.** The top `top_n` ranked events, plus cold-start events in the same score range (shown unranked,
hidden by default). Filters: date window, field and venue (either paper), minimum total, cold-start toggle,
positives-only toggle, and text search over titles, paper ids and event ids. Columns: rank, citing to cited paper,
date, total, the four component z-values as small bars from -4 to +4 with their values, and the label. Column
headers sort; the rank button opens the detail view. "Export CSV" downloads every filtered row (all pages) with raw
values and z-scores.

**Citation detail ("why this was flagged").**
- A waterfall of `w_c * z_c` per component, running from 0 to the total.
- One card per component: raw value, z, percentile, the reference histogram with the event marked, and the
  component's formula and definition.
- The local subgraph at time t. Hops run in either direction, edges keep their direction, only events with
  ts <= t are used, and the scored citation is excluded from the context but drawn highlighted. Nodes are coloured
  by role or by field (the three most common fields get fixed colours). Edges are coloured by the Forman screen
  F(a->b) = 2 - in_deg_t(a) - out_deg_t(b), recomputed at t from event counts, or by the cached ORC of the latest
  escalated instance at or before t.
- A timeline of citations received by the cited paper up to t, with this citation counted in the last bin.
- The full inputs table (every stored column of the scores row, plus components and curvature cache columns)
  and the component definitions.

**Papers & authors.** Paper score = mean of the 5 highest totals among the ranked events the paper takes part in,
as citing or cited paper (fewer when it has fewer). The count is the number of those events at or above the flag
threshold. These follow the S3.3 definition and are computed here from the scores. Author score = the same aggregate
over the distinct events touching any of the author's papers. Selecting an entry lists its highest-scoring
citations, with links into the detail view.

**Cartels (synthetic).** Shown only when `cartels` is given. It holds:
- Recall per obviousness level at a chosen k (rank <= k counts as caught; cold-start events cannot be caught).
- The member graph, with donor and recipient roles and injected edges marked caught or missed.
- The injected events table.

Recall-by-level entries found in `metrics` are listed next to the computed ones.

## Conventions and limits

- Rank order: total descending, ties broken by the canonical order (t, event_id). Cold-start events and rows
  without a finite total get no rank.
- Reference histograms use all ranked events in the file, not only training rows. The z-scores themselves use the
  frozen statistics when given.
- In the heatmap, the field of an event is the field of the citing paper (its publication date is t).
- Subgraph cap: u and v are always kept. Ring-1 neighbours are ranked by the number of core nodes they touch, then
  by links, recency and id. With radius 2, ring 1 gets at most half the budget, and ring 2 is drawn from neighbours
  of the shown ring-1 nodes. The caption reports shown/total counts.
- Curvature colour classes: |F| <= 2 is neutral. The breaks are the median and the 90th percentile of the larger
  |F| over all drawn edges. ORC uses fixed breaks 0.05 / 0.25 / 0.5.
- Determinism: same inputs and seed give a byte-identical file (seeded spring layouts, sorted JSON keys, rounded
  floats, no timestamps). Size: about 2.8 MB for the synthetic demo at `top_n=500` (bound tested: under 8 MB).
- The component definitions shown on the page come from `src/citecurv/viz/component_text.py`, the one place to
  edit. They restate the S3.1 proposed defaults and are marked PROVISIONAL until `docs/score_components.md` is
  approved.
- Some notebook front ends block downloads from sandboxed frames. If "Export CSV" does nothing there, open the HTML
  file directly in a browser.

## Design

The look follows the project's tokens: warm off-white background, white bordered cards with 12 px radius, and the
"Segoe UI" system font stack. A dark token set follows `prefers-color-scheme`, and a header button switches between
auto, light and dark. Chart colours were validated with the dataviz palette checker:
- The four components use categorical slots 1 to 4, which pass the adjacent-pair CVD and normal-vision gates in
  both modes. Two light-mode slots sit below 3:1 contrast, so every component mark carries its name and a data
  table.
- The total score is neutral ink.
- Curvature uses a blue/red diverging scale with a gray midpoint.
- The heatmap uses a one-hue blue ramp.
- The status colours (with icon and text) appear only on the D4 verdict.

Every chart has a data table, hover tooltips and keyboard reading: focus the chart, then use the arrow keys. Tabs
follow the WAI-ARIA tab pattern, and the layout works at 375 px width without horizontal page scrolling.

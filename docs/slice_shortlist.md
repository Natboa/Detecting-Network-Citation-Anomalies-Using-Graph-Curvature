# Slice shortlist for the OpenAlex pull (task S1.3, preparation)

> **PROVISIONAL(G3).** Positive labels follow the Referencing/Attributions rule (L12), which still
> awaits advisor sign-off. All counts below are from the Retraction Watch side. The OpenAlex-side
> criteria (slice size, edges, labeled papers inside the slice) are not measured yet, because
> api.openalex.org was unreachable when this note was written. Sources, all in `results/`:
> `rwdb_density_2026-07-06_*` (tables named below by suffix), `rwdb_counts_2026-10-31.json`,
> `openalex_subfield_sizes_2026-10-06.json`.

## Label source and counting rule

Labels come from the Retraction Watch data, publicly available via Crossref since Crossref acquired
the database in September 2023 (Retraction Watch, 2026). The frozen snapshot of 2026-07-06 (SHA256
`ae6ea4b0…`) stays the label source. It is byte-identical to the upstream file at commit `21d7124d`
(data for 2026-07-02), so it can be downloaded again exactly. A *paper* is a distinct normalized
original-paper DOI. Only notices of nature *Retraction* count: the Referencing/Attributions family
has 14,596 such papers, plus 177 under expressions of concern, 92 under corrections and 2 under
reinstatements (`natures.csv`). The dataset's own documentation states that expressions of concern
and corrections are collected less completely than retractions (Retraction Watch, 2026).

A refreshed copy (2026-10-31, commit `a98cc7e2`) adds 2,934 records and 555 family DOIs net. Of the
561 new family DOIs, 182 were first published in 2023, 104 in 2024 and 88 in 2025
(`rwdb_counts_2026-10-31.json`). It is recorded for comparison only.

## Shape of the label family (snapshot 2026-07-06)

- **Few publishers, few events.** Hindawi accounts for 8,421 of the 14,596 papers (57.7%). Thirty-two
  same-day batches of 50 or more papers from one publisher hold 11,310 papers (77.5%); the largest is
  655 papers retracted by IOS Press on 2025-04-17, all from the *Journal of Intelligent & Fuzzy
  Systems*. The ten largest journals hold 46.7% (`summary.json`, `mass_events.csv`).
- **Few years.** By original publication year: 2,730 papers from 2021 and 7,290 from 2022, then
  1,832 (2023), 910 (2024) and 63 (2025) (`orig_year.csv`). The median time from publication to
  retraction is 1.34 years and the 90th percentile 2.93 years (`summary.json`), so the 2024 and 2025
  cohorts are still filling, as the refresh above shows. Candidate slices therefore end in 2023.
- **Overlap with other reasons.** 67.9% of the papers also carry "Concerns/Issues about Peer
  Review", 46.1% "Paper Mill" and 22.3% "Compromised Peer Review" (`cooccurrence.csv`). The broadened
  family (Referencing/Attributions, paper mill or peer review) has 25,226 papers.

## Shortlist

Every candidate covers original publication years 2018–2023. The test window is assumed to be
2022–2023; the 2023 column is the stress case where the time split (S4.1) leaves one test year.
S1.3 asks for at least 300 labeled papers, at least 100 of them in the test period
(`candidates.csv`; subfield names from OurResearch, 2026c).

| Candidate | RWDB subjects or venues | Ref/Attr papers | 2022–23 | 2023 | Broadened | Top publisher | OpenAlex starting filter (subfields) |
|---|---|---:|---:|---:|---:|---|---|
| computing | Computer Science; Data Science | 5,859 | 3,923 | 808 | 8,690 | Hindawi 62.5% | 1702 Artificial Intelligence, 1705 Computer Networks |
| env_energy | Environmental Sciences; Energy | 1,184 | 888 | 312 | 1,754 | Hindawi 41.3% | 2310 Pollution, 2105 Renewable Energy |
| cell_mol_bio | Biology - Cellular; Biology - Molecular; Genetics; Biochemistry | 1,497 | 1,012 | 108 | 3,873 | Hindawi 72.3% | 1307, 1312, 1311, 1303 |
| oncology | Medicine - Oncology; Biology - Cancer | 1,080 | 783 | 79 | 2,720 | Hindawi 78.5% | 2730 Oncology, 1306 Cancer Research |
| venues_cs | the ten journals with most computing positives | 4,894 | 3,572 | 644 | 6,044 | Hindawi 70.7% | source IDs, to resolve |

Ref/Attr papers per original publication year, 2018 to 2023: computing 107, 259, 252, 1,318,
3,115, 808; env_energy 16, 29, 55, 196, 576, 312; cell_mol_bio 17, 43, 85, 340, 904, 108;
oncology 14, 24, 48, 211, 704, 79; venues_cs 47, 190, 274, 811, 2,928, 644.

Suggested order of measurement: (1) **computing**, the most labels in both test windows;
(2) **env_energy**, which passes both windows with the lowest single-publisher share (41.3%);
(3) **cell_mol_bio**, which passes both windows narrowly and whose broadened set is 2.6 times its
primary set, the highest ratio here, which suits the L12 sensitivity run. **oncology** fails the
one-year stress case (79 papers). **venues_cs** is a fallback in case topic sets capture too few
labels; a venue slice makes "journal and year" a shortcut for the label.

## What the OpenAlex side still has to show

- **Size.** In the topic hierarchy OpenAlex (Priem et al., 2022) published with its October 2026
  re-classification (OurResearch, 2026c), each candidate's subfield set holds 2.4 to 15.4 million
  works over all years, against a target of about 300,000; single topics in these subfields have
  medians of 23,000 to 67,000 works (`openalex_subfield_sizes_2026-10-06.json`). Each slice will be
  a set of topics chosen by labeled papers per work (`pick_topics`), not a whole subfield.
- **Topic churn.** OpenAlex replaced every work's topics on 2026-10-06 (OurResearch, 2026c); in these
  subfields 16% to 49% of the earlier works kept their subfield (same file). Concepts are
  deprecated and not maintained (OurResearch, 2026b). Measure only after that date and freeze the
  S4.0 pull by work IDs.
- **Capture.** How many RWDB-side positives OpenAlex places inside each topic set is unknown until
  the labeled DOIs are looked up.

## Decisions for Maor

1. Snapshot: keep 2026-07-06 (default in force) or adopt 2026-10-31.
2. Notice types: *Retraction* only (used here) or every type (the current `load_labels` default).
3. Concentration: with 77.5% of positives in 32 batches, a score could learn venue and year rather
   than citation behaviour; report S4.6 metrics also with the largest batches held out.
4. End year 2023 for the slice, because later cohorts are still being retracted.

## Commands once the key exists

Colab: open `notebooks/10_data.ipynb`, set `TAG` and `STAGE = "S1.3"`, run all; set `CHOSEN` to the
best narrowed candidate, re-run the second S1.3 cell, copy `slice.yaml` to `configs/slice.yaml`.
The DOI lookup takes 292 calls (one per 50 DOIs); each candidate then takes 12 count calls plus one
per 50 labeled DOIs in its topic set. A list call costs 10 credits and a free key has 100,000
credits a day (OurResearch, 2026a); group-by calls use the same list endpoint. Locally:

```bash
export OPENALEX_API_KEY=...        # never commit it
PYTHONPATH=src python - <<'EOF'
from citecurv.data import slice_query as sq
from citecurv.data.rwdb import load_labels
labels = load_labels("data/labels/rwdb_2026-07-06.csv", families=["referencing/attributions"])
labels = labels[labels.retraction_nature == "Retraction"]
lookup, stats = sq.lookup_labeled_works(sorted(set(labels.doi_norm.dropna())))
print(stats)                       # dois_found / dois_queried = strict DOI found-rate (S1.2)
for cand in sq.SHORTLIST[:4]:
    broad = sq.measure_slice(cand)
    pick = sq.pick_topics(broad["works_by_topic"], sq.labeled_by_topic(lookup, cand["years"]))
    narrow = sq.topic_candidate(cand, pick["topics"])
    dois = sorted(set(lookup.loc[lookup.topic_id.isin(pick["topics"]), "doi_norm"]))
    m = sq.measure_slice(narrow, dois=dois)
    print(narrow["name"], m["works_total"], m["edges_est"], m["labeled_total"], m["labeled_test"], m["passes"])
EOF
```

The calls behind `measure_slice` for **computing** (the other candidates change only the subfield
list; every URL is in `summary.json`):

```text
/topics?filter=subfield.id:1702&per-page=1&select=id,display_name,subfield,field
/works?filter=primary_topic.subfield.id:1702|1705,publication_year:2018-2023&group_by=publication_year
/works?filter=primary_topic.subfield.id:1702|1705,publication_year:2018-2023&group_by=primary_topic.id
/works?filter=primary_topic.subfield.id:1702|1705,publication_year:2018-2023,has_references:true&group_by=publication_year
/works?filter=primary_topic.subfield.id:1702|1705,publication_year:2018-2023,is_retracted:true&group_by=publication_year
/works?filter=primary_topic.subfield.id:1702|1705,publication_year:2018-2023&sample=100&seed=0&per_page=100&select=id,publication_year,referenced_works
/works?filter=doi:https://doi.org/<doi1>|...|https://doi.org/<doi50>&select=id,doi,publication_year,primary_topic,primary_location&per_page=100
/rate-limit?api_key=<key>          # credits_used must rise after the first calls
```

All hosts are `https://api.openalex.org`, corpus *core* (no `include_xpac`); filter names follow
OurResearch (2026a). The tables here
regenerate with `write_density_tables(load_records("data/labels/rwdb_2026-07-06.csv"), "results",
"2026-07-06", SHORTLIST)`.

## References

OurResearch. (2026a). *OpenAlex technical documentation* [Source repository, commit 8cec9db].
GitHub. https://github.com/ourresearch/openalex-docs

OurResearch. (2026b). *OpenAlex technical documentation: Concepts* [Source repository]. GitHub.
https://github.com/ourresearch/openalex-docs/tree/main/api-entities/concepts

OurResearch. (2026c). *openalex-topic-classification* (Version 2.0.0) [Computer software]. GitHub.
https://github.com/ourresearch/openalex-topic-classification

Priem, J., Piwowar, H., & Orr, R. (2022). *OpenAlex: A fully-open index of scholarly works, authors,
venues, institutions, and concepts*. arXiv. https://arxiv.org/abs/2205.01833

Retraction Watch. (2026). *Retraction Watch data* [Data set]. Crossref.
https://gitlab.com/crossref/retraction-watch-data

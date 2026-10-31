# Literature and tooling refresh (2026-10-31)

Prepared for the citecurv capstone (agent A9). Scope: research and tooling changes since about mid-2025 that bear on
`IMPLEMENTATION_PLAN.md` (approved 2026-07-06; its component research dates from early July 2026), plus a few older
items that the plan or the Phase A reference list does not include.

**Status tags.**
- `[verified: fetched page]`: I fetched the page or file named in the entry, and the summary keeps to what it showed.
  Where only a code repository or package index page was fetched (not the paper), the tag says so.
- `[snippet only]`: I saw the claim only in search-result snippets or third-party summaries. Treat it as unconfirmed
  until someone opens the primary source.

**Access limits in this session.** From this container only github.com, raw.githubusercontent.com, gitlab.com and
pypi.org could be fetched. arxiv.org, openreview.net, proceedings.mlr.press, dl.acm.org, link.springer.com, nature.com,
ojs.aaai.org, aclanthology.org, retractionwatch.com, crossref.org, every openalex.org host, huggingface.co,
colab.research.google.com and the Google blogs were blocked (DNS failure or proxy 403). As a result, most papers below
are `[snippet only]` unless their GitHub repository confirmed the details. The OpenAlex entries are verified against
the documentation source files that OpenAlex keeps on GitHub (`ourresearch/docs`, retired 2026-08-11, and
`ourresearch/openalex-help`, the source of help.openalex.org).

**Five findings that matter most.**
1. Colab's default runtime moved to Python 3.13 on 2026-08-25. The plan assumes Python 3.12 as the Colab default, and
   the curvature pin `scipy<=1.13.1` has no Python 3.13 wheel (B5, B4).
2. Vendored DyGLib calls `random.sample()` on a `set`, which raises `TypeError` on Python 3.11 and later. Smoke Block B
   will fail at data loading unless this is patched (A5.1).
3. OpenAlex concepts are frozen and new works no longer get concept tags; topic assignments for all works were replaced
   in October 2026. The slice query (S1.3) and the `concepts` column in `works.parquet` need to move to topics (B1.4,
   B1.5).
4. OpenAlex now ignores `mailto`, limits a free key to 30 requests per second, requires a key for cursor paging past
   the first page, and returns 429 both for "too fast" and for "daily budget spent" (B1.1, B1.3).
5. `adapters` 1.3.0 requires `transformers~=4.57.6`, while Colab now ships transformers 5.22.0. The SPECTER2 stack
   still installs, but only in an environment that downgrades transformers (B7).

---

## Part A. Research

### A1. Citation cartels, citation manipulation, citation rings

**A1.1** Gupta, P., Udandarao, V., & Bandi, S. S. S. (2026). *Trust-aware citation cartel ranking in scholarly
knowledge graphs* (arXiv:2607.06528). arXiv. https://arxiv.org/abs/2607.06528
`[snippet only]`
The authors build a DBLP-derived citation graph of 500,000 papers and about 4.87 million edges, label 205,897 citation
pairs for citation intent with a large language model, train a SciBERT student, and apply it to 2.04 million edges.
They rank paper communities with a Composite Cartel Index that combines internal density, citation inflation,
reciprocity, semantic superficiality, degree assortativity and a trust-weighted PageRank shift. They stress that dense
internal citation alone does not show a cartel, because legitimate research communities are also dense.
Implication for us: **cite in book.** It supports our choice to require agreement between independent signals and to
treat flags as candidates for human review; several of its index components overlap with our tier-1 statistical
features.

**A1.2** Liu, J., Wei, S., Cheng, X., & Wang, G. (2026). Designing a computational framework for identifying
suspicious citation groups in heterogeneous academic networks: A graph learning approach. *Information Systems
Frontiers*. https://link.springer.com/article/10.1007/s10796-026-10770-y
`[snippet only]` (author list taken from a secondary news write-up, not from the publisher page)
The framework combines local citation-intensity signals with global network-role signals to flag suspicious citation
groups among journals and papers. The snippet frames existing methods as relying mainly on local signals such as
pairwise citation intensity or node-level anomalies.
Implication for us: **watch.** Confirm the author list on the publisher page before citing.

**A1.3** *Citation cliques in low impact journals* (2026; arXiv:2605.11930; author names not retrieved).
https://arxiv.org/abs/2605.11930
`[snippet only]`
Using Crossref records, the study matches authors publishing in low- and high-influence venues by subject and h5 index
and reports, in a 10% sample of 9,431 matched pairs, 6.7 times higher co-author citation rates and 4.7 times higher
reciprocity in the low-impact group. An Isolation Forest with a cohesion-based filter isolates 277 author profiles, 93.5%
of them in the case group.
Implication for us: **watch.** Relevant to the per-author rollup (FR8) if author IDs survive the slice pull.

**A1.4** Erdoğan, C., Daniel, B., Wotka, B., Sai, A., & Iamnitchi, A. (2026). *Citation farming on ResearchGate:
Blatant and effective* (arXiv:2604.13784). arXiv. https://arxiv.org/abs/2604.13784
`[snippet only]`
The authors examine almost 3,000 papers uploaded by five ResearchGate accounts suspected of running a citation-boosting
service and build paper- and author-level citation networks from them. Many of these papers cite a small set of
authors disproportionately.
Implication for us: **cite in book** as recent evidence of organised citation boosting (motivation section). Whether
such uploads appear in OpenAlex was not checked.

**A1.5** Kojaku, S., Livan, G., & Masuda, N. (2021). Detecting anomalous citation groups in journal networks.
*Scientific Reports, 11*, 14524. https://doi.org/10.1038/s41598-021-93572-3
`[snippet only]` (older item)
CIDRE flags groups of journals that exchange citations well above a null model that accounts for communities and
journal size. A 2022 author correction states that it detected 11 of the 22 groups suspended from Journal Citation
Reports in or before the suspension year.
Implication for us: **cite in book.** The plan builds the synthetic injector on the CIDRE donor/recipient pattern (L18),
but the Phase A reference list does not include this paper.

**A1.6** Liu, J., Bai, X., Wang, M., Tuarob, S., & Xia, F. (2024). Anomalous citations detection in academic networks.
*Artificial Intelligence Review, 57*(4), 103. https://doi.org/10.1007/s10462-023-10655-5
`[snippet only]` (older item)
ACTION defines anomalous citations as citations to papers not relevant to the citing paper and detects them with
non-negative matrix factorization and network representation learning on a heterogeneous network of journals, papers and
authors. The authors built three anomalous-citation datasets for evaluation.
Implication for us: **cite in book.** It is the follow-up to GLAD (Liu et al., 2022), which the Phase A book already
cites.

**A1.7** Fister, I., Jr., Fister, I., & Perc, M. (2016). Toward the discovery of citation cartels in citation networks.
*Frontiers in Physics, 4*, 49. https://doi.org/10.3389/fphy.2016.00049
`[snippet only]` (older item)
A perspective article that defines citation cartels as groups of authors who cite each other disproportionately more
than they cite other groups working on the same subject.
Implication for us: **cite in book** as the definition source for "citation cartel".

**A1.8** Retraction Watch. (2026, March 30). *A citation alert led researchers to a network of fake articles. But who is
benefiting?* https://retractionwatch.com/2026/03/30/fake-articles-plagiarism-preprints-arxiv-ssrn-citation-network/
`[snippet only]`
Reports fake preprints on SSRN and other servers that appear designed to inflate citations to named researchers; who
uploaded them is unclear and the named beneficiaries deny involvement.
Implication for us: **watch.** A possible worked example for the manual-review categories (S4.7); non-academic link only.

### A2. Paper-mill detection with citation or reference signals

**A2.1** Scancar, B., Byrne, J. A., Causeur, D., & Barnett, A. G. (2026). Machine learning based screening of potential
paper mill publications in cancer research: Methodological and cross sectional study. *BMJ, 392*, e087581.
https://doi.org/10.1136/bmj-2025-087581
`[snippet only]`
A BERT model fine-tuned on titles and abstracts only, trained on 2,202 retracted paper-mill papers and validated on
independent data (accuracy 0.91), flagged 261,245 of 2,647,471 cancer papers from 1999 to 2024 (9.87%). The authors state
that flags are not confirmed cases of fraud.
Implication for us: **cite in book.** A text-only detector at scale, which contrasts with our citation-structure signal;
it also shows retracted paper-mill papers being used as training labels elsewhere.

**A2.2** Scancar, B., Byrne, J. A., Causeur, D., & Barnett, A. G. (2026). Suspected distortion of citations in
high-impact cancer journals. *Proceedings of the National Academy of Sciences*. https://doi.org/10.1073/pnas.2618402123
(preprint: bioRxiv, https://www.biorxiv.org/content/10.64898/2026.05.25.727627v1)
`[snippet only]` (published author list not confirmed; the four names come from the preprint)
The study argues that reference lists can serve as a first-line screen for paper-mill output, because they are fixed by
the authors and may show a mill's recycling pattern more clearly than incoming citations. A BERT-based model flags about
12% of 33,159 papers; a secondary blog reports that flagged papers received about twice as many citations.
Implication for us: **cite in book**, and consider one tier-1 feature: the share of a citing paper's references that
point to works already retracted at time t (retraction date at or before t, so the feature does not leak future label
information).

**A2.3** Candal-Pedreira, C., et al. (2024). Retracted papers originating from paper mills: A cross-sectional analysis
of references and citations. *Journal of Clinical Epidemiology, 172*, 111397.
https://doi.org/10.1016/j.jclinepi.2024.111397
`[snippet only]` (older item; full author list to be completed from the DOI record)
From 416 paper-mill papers retracted in 2022 (14,411 references, 8,479 citations), 473 of the references and citations
were themselves retracted for paper-mill origin, and about 42% of the source papers were referenced or cited by other
retracted papers. The retracted-only network formed clusters of 53, 48 and 44 papers.
Implication for us: **cite in book.** Direct evidence that paper-mill retractions cluster in the citation graph, which
supports the broadened-label sensitivity run (L12 includes "Paper Mill") and the reference-neighbourhood feature in A2.2.

**A2.4** Zhang, J., Liu, J., E, H., Hu, T., Qiao, X., & Tang, Z. (2025). A paper mill detection model based on citation
manipulation paradigm. *Journal of Data and Information Science*. https://doi.org/10.2478/jdis-2025-0003
`[snippet only]`
The authors describe three citation patterns linked to paper-mill activity and propose PDCN, which combines BERT title
features with structural features from a heterogeneous graph. They report 81.85% accuracy and an F1 score of 80.49% on
their own dataset.
Implication for us: **cite in book.** Code availability was not checked, so not a baseline candidate yet.

**A2.5** *Improving machine learning-based paper mill detection in cancer research using full text, image and citation
data* (Research Square preprint rs-11097507; author names not retrieved). https://www.researchsquare.com/article/rs-11097507/v1
`[snippet only]`
Compares an abstract-only baseline, full-text, image and fused models; one citation signal counts references to known
paper-mill articles. It reports text as the most valuable signal and images as useful but less reliable.
Implication for us: **watch.**

**A2.6** Porter, S. J., & McIntosh, L. D. (2024). Identifying fabricated networks within authorship-for-sale
enterprises. *Scientific Reports, 14*. https://doi.org/10.1038/s41598-024-71230-8
`[snippet only]` (older item; only title, authors, venue and date were seen)
Implication for us: **watch** for the author-level rollup (FR8).

### A3. Edge-level anomaly detection on continuous-time dynamic graphs

**A3.1** Hua, F., Qi, Y., Wei, Z., Tian, Y., Xu, C., Wu, X., Li, J., & Guo, J. (2026). BAG: Benchmarking anomaly
detection on dynamic graphs. *Proceedings of the AAAI Conference on Artificial Intelligence, 40*(17).
https://doi.org/10.1609/aaai.v40i17.38510 (code: https://github.com/frhuaaa/BAG)
`[verified: fetched page]` for the repository README (model list, datasets, acknowledgements); venue, DOI and findings
are `[snippet only]`.
The README lists 25 models: static (GCN, GIN, ChebNet, SGC, GAT, GT, RFGraph, XGBGraph), discrete-time (Dy_GrAE, T_GCN,
EvolveGCN-O/H, MPNN_LSTM, AddGraph) and continuous-time (JODIE, DyRep, TGN, TGAT, TCL, CAWN, GraphMixer, DyGFormer,
FreeDyG, SLADE, SAD), on 10 datasets; it acknowledges DyGLib, GADBench and PyTorch Geometric Temporal and states no
licence. Snippets report that continuous-time models detect anomalous edges better than discrete-time ones, and that
organic anomalies are harder than injected ones because of temporal anomalies and strong class imbalance.
Implication for us: **cite in book** (it backs the continuous-time framing, L14) and **add as baseline (optional)**: its
harness runs SLADE and SAD, which are published dynamic-graph anomaly detectors that L17 currently lacks. With no
licence stated, it could only be run as a sealed environment, not vendored.

**A3.2** Poštuvan, T., Grohnfeldt, C., Russo, M., & Lovisotto, G. (2024). Learning-based link anomaly detection in
continuous-time dynamic graphs. *Transactions on Machine Learning Research*. https://arxiv.org/abs/2405.18050
(code: https://github.com/timpostuvan/CTDG-link-anomaly-detection)
`[verified: fetched page]` for the repository README; the taxonomy description is `[snippet only]`. Older item.
The paper proposes a taxonomy of edge-level anomalies in continuous-time graphs and a generator of synthetic anomalies
that are consistent in time, structure and context; the code (MIT) ships a `link_anomaly_generator.py` with a
`temporal-structural-contextual` type, adapts TGB baseline code, uses TGN as its main example, and also evaluates on
graphs with organic anomalies (LANL, DARPA-THEIA).
Implication for us: **change plan (S4.5).** Add their temporal-structural-contextual edge anomalies as a second
injection type next to the CIDRE cartel injector, so the injection track also covers single anomalous citations, not
only dense groups. Cite in book as the reference for synthetic link anomalies.

**A3.3** Tian, Y., et al. (2026). *Learning discriminative and generalizable anomaly detector for dynamic graph with
limited supervision* (arXiv:2602.20019). ICML 2026 poster. https://arxiv.org/abs/2602.20019
`[snippet only]` (five co-authors not retrieved)
SDGAD is a model-agnostic framework for continuous-time dynamic graph models. It encodes how a current interaction
deviates from its historical context and learns a decision boundary from a normalizing-flow likelihood of normal data,
using a few labelled anomalies when available; experiments cover six datasets with real and synthetic anomalies.
Implication for us: **watch**; cite in book if verified. Our setting (weak, sparse Retraction Watch positives) is the
"limited supervision" case it addresses.

**A3.4** Chen, J., Fu, S., Ma, Z., Feng, M., Wirjanto, T. S., & Peng, Q. (2025). Semi-supervised anomaly detection with
extremely limited labels in dynamic graphs. In *DASFAA 2025*. https://arxiv.org/abs/2501.15035
`[snippet only]`
EL2-DGAD uses a transformer-based dynamic graph encoder for the case of very few labelled anomalies.
Implication for us: **watch.**

**A3.5** *Beat the counter first: A baseline for temporal-graph anomaly detectors* (2026; arXiv:2608.15965; author
names not retrieved). https://arxiv.org/abs/2608.15965
`[snippet only]`
Only the title and an acknowledgement mentioning production systems at Microsoft were visible; the method and results
were not.
Implication for us: **watch.** Read before S4.4; the title suggests simple counting baselines, which our tier-1
statistical baselines would cover.

### A4. Curvature for anomaly and community detection; directed curvature

**A4.1** Grover, K., Gordon, G. J., & Faloutsos, C. (2025). CurvGAD: Leveraging curvature for enhanced graph anomaly
detection. In *Proceedings of the 42nd International Conference on Machine Learning* (PMLR 267).
https://proceedings.mlr.press/v267/grover25a.html (code: https://github.com/karish-grover/curvgad)
`[verified: fetched page]` for the repository and its `requirements.txt`; no follow-up paper by the authors was found
(`[snippet only]`).
The repository (MIT, 3 commits, no visible activity after 2025) lists `torch>=1.9.0`, `torch-geometric>=2.0.0`,
`dgl>=0.9.0`, `geoopt>=0.5.0`, `GraphRicciCurvature>=0.5.3`, `pygod>=0.4.0`, `catboost>=1.0.0` and others without upper
bounds, and also lists the standard-library names `argparse`, `warnings`, `json`, `os`, `time`, `random`, `itertools`,
`collections`. On PyPI, `warnings`, `os`, `json`, `random`, `itertools` and `collections` do not exist (so
`pip install -r requirements.txt` stops with an error), while `time` and `argparse` resolve to unrelated packages. DGL's
latest PyPI release is 2.2.1 (2024-05-13) with wheels for Python 3.8 to 3.12 only.
Implication for us: **change plan (S4.4).** Write the sealed-environment recipe now: drop the standard-library lines,
pin `GraphRicciCurvature==0.5.3.2`, use a Python 3.12 runtime, and pick a DGL build that matches the torch version; keep
the existing fallback (report CurvGAD's published numbers, marked as such) if the environment cannot be built.

**A4.2** Chatterjee, T., Albert, R., Thapliyal, S., Azarhooshang, N., & DasGupta, B. (2021). Detecting network
anomalies using Forman-Ricci curvature and a case study for human brain networks. *Scientific Reports, 11*.
https://doi.org/10.1038/s41598-021-87587-z
`[snippet only]` (older item)
The paper treats change detection between time points of a network as anomaly detection, using the change in
Forman-Ricci curvature of edges with a statistical test, and reports that Forman and Ollivier-Ricci curvatures are poorly
correlated on their data.
Implication for us: **cite in book.** It is the closest prior work to our curvature-dynamics component (`s_curvdyn`)
and supports computing both curvatures (L3) rather than treating one as a proxy for the other.

**A4.3** Sun, L., Wan, Q., Zhou, S., Huang, Z., & Yu, P. S. (2026). *RiemannGL: Riemannian geometry changes graph deep
learning* (arXiv:2602.10982). arXiv. https://arxiv.org/abs/2602.10982
`[snippet only]`
A position and survey paper arguing for intrinsic manifold structure in graph neural networks; snippets say it lists
extending geometric (curvature) insights to dynamic graphs as a future direction.
Implication for us: **cite in book** with care, as support for the claim that curvature on dynamic graphs is still open.

**A4.4** Tian, Y., Lubberts, Z., & Weber, M. (2025). Curvature-based clustering on graphs. *Journal of Machine Learning
Research, 26*(52). https://www.jmlr.org/papers/v26/24-0781.html
`[snippet only]`
Uses discrete Ricci curvatures and their flows to find single-membership and mixed-membership communities, the latter
on the line graph.
Implication for us: **cite in book** as background on curvature and community structure.

**A4.5** Park, Y. J., & Li, D. (2025). Lower Ricci curvature for efficient community detection. *Transactions on
Machine Learning Research*. https://arxiv.org/abs/2401.10124
`[snippet only]`
Lower Ricci Curvature has linear complexity and is bounded above by Balanced Forman curvature, which is bounded above by
Ollivier-Ricci curvature; removing low-curvature edges before community detection improved several methods.
Implication for us: **watch.** It is undirected, so not a drop-in for our directed screen; a possible cheap comparison
if the ORC budget (S2.3) is tight.

**A4.6** Wiesler, E. (2026). *Ollivier-Ricci curvature of Riemannian manifolds and directed graphs with applications to
graph neural networks* (arXiv:2604.14211). arXiv. https://arxiv.org/abs/2604.14211
`[snippet only]`
A thesis whose last part extends Ollivier-Ricci curvature to directed graphs; it states that only two earlier papers
attempted a directed Ollivier-Ricci definition, one of them Saucan et al. (2019).
Implication for us: **watch.** If confirmed, it supports describing directed ORC as little studied.

**A4.7** *HyMSS-GAD: A hybrid multi-stage framework for multi-view graph anomaly detection with structural,
contextual, and geometric reasoning* (2026). *Scientific Reports*. https://www.nature.com/articles/s41598-026-42823-2
`[snippet only]` (author names not confirmed)
A static graph anomaly detector with a curvature-aware decoder based on Ollivier-Ricci curvature, evaluated on Cora,
Citeseer, PubMed, ACM and Amazon; it cites CurvGAD and leaves temporal graphs to future work.
Implication for us: **watch.**

**A4.8** Topping, J., et al. (2022), Balanced Forman curvature, now in GraphRicciCurvature 0.6.0 as
`FormanRicci(method="balanced")`. Source: https://github.com/saibalmars/GraphRicciCurvature (FormanRicci.py)
`[verified: fetched page]` (library source)
The library's `FormanRicci` still logs that directed graphs are not supported and converts directed input to
undirected; the new balanced variant is for simple, unweighted graphs.
Implication for us: **ignore** for the screen. It confirms L4: a custom directed Forman module is still needed.

### A5. Temporal graph learning backbones after TGN

**A5.1** Yu, L., Sun, L., Du, B., & Lv, W. (2023). Towards better dynamic graph learning: New architecture and unified
library (DyGLib). https://github.com/yule-BUAA/DyGLib
`[verified: fetched page]` (commit list, issue list, issues #22 and #23, `utils/DataLoader.py`)
The last commit is dated 2023-09-28 ("update paper bibtex"); 8 issues are open, and issue #23 (December 2025, a user
report that switching the negative-sampling strategy to "historical" or "inductive" gives the same results as "random")
has no maintainer reply. `utils/DataLoader.py` line 115 calls `random.sample(test_node_set, ...)` on a `set`; issue #22
(May 2025) reports the failure and a commenter traces it to the Python version. Since Python 3.11, `random.sample` no
longer accepts sets (CPython change, `[snippet only]`). Line 81 also hard-codes `NODE_FEAT_DIM = EDGE_FEAT_DIM = 172`
and asserts that real features are not wider, which the 768-dimensional SPECTER2 features violate.
Implication for us: **change plan (S1.7 and S0.5 Block B).** DyGLib is unmaintained. Add a documented patch before
Block B: `random.sample(sorted(test_node_set), ...)` (sorted rather than `list` so that the result depends only on the
seed, NFR4). Patch 1 must cover both `NODE_FEAT_DIM` and `EDGE_FEAT_DIM` (same line, and again near line 192). If we ever
use non-random negative sampling, test issue #23 first.

**A5.2** Chmura, J., Huang, S., Ngo, T. G. B., Parviz, A., Poursafaei, F., Leskovec, J., Bronstein, M., Rabusseau, G.,
Fey, M., & Rabbany, R. (2025). *TGM: A modular and efficient library for machine learning on temporal graphs*
(arXiv:2510.07586). arXiv. https://github.com/tgm-team/tgm
`[verified: fetched page]` for the README and the PyPI record of `tgm-lib`; the ICLR 2026 poster listing and speed-up
figures are `[snippet only]`.
TGM (MIT) implements TGN, DyGFormer, TGAT, GraphMixer, TPNet, EdgeBank and others, loads TGB datasets, and warns that it
is in beta and may introduce breaking changes; `tgm-lib` 0.1.0b1 was released on 2026-01-24. Snippets report an average
7.8 times speed-up over DyGLib.
Implication for us: **watch.** A maintained fallback if vendored DyGLib cannot be made to run on current torch; it
would need its own custom-data loader and a new decision, since L1 names DyGLib.

**A5.3** Huang, S., et al. (2023). Temporal Graph Benchmark for machine learning on temporal graphs (NeurIPS 2023
Datasets and Benchmarks), and Gastinger, J., et al. (2024). TGB 2.0: A benchmark for learning on temporal knowledge
graphs and heterogeneous graphs (NeurIPS 2024 Datasets and Benchmarks). https://github.com/shenyangHuang/TGB
`[verified: fetched page]` for the README and the PyPI record of `py-tgb`; the TGB 2.0 findings are `[snippet only]`.
`py-tgb` 2.3.0 was released on 2026-07-27; the README says download links from versions before 2.3.0 are deprecated
and that torch is now optional. TGB 2.0 adds eight temporal knowledge-graph and heterogeneous datasets; snippets report
that simple heuristics are often competitive and that most methods fail to run on the largest datasets.
Implication for us: **cite in book** (evaluation-protocol context; the Phase A list cites neither). We do not use TGB
data, so the download change does not affect us.

**A5.4** Tian, Y., Qi, Y., & Guo, F. (2024). FreeDyG: Frequency enhanced continuous-time dynamic graph model for link
prediction. In *ICLR 2024*. https://github.com/Tianxzzz/FreeDyG
`[verified: fetched page]` (README: venue, MIT licence, acknowledgement of the DyGFormer code)
Implication for us: **ignore** as a backbone (L1 fixes plain TGN). It appears in BAG (A3.1) as a detector.

**A5.5** Yi, L., Peng, J., Zheng, Y., Mo, F., Wei, Z., Ye, Y., Yue, Z., & Huang, Z. (2025). TGB-Seq benchmark:
Challenging temporal GNNs with complex sequential dynamics. In *ICLR 2025*. https://arxiv.org/abs/2502.02975
`[snippet only]`
Eight datasets built to minimise repeated edges, so that models must learn sequential patterns; current methods degrade
markedly and train slowly on them.
Implication for us: **cite in book.** In our data a (citing, cited) pair occurs at most once, because a reference list is
fixed at publication, so the TGB-Seq finding is the relevant warning against expecting link-prediction quality reported
on repeat-heavy datasets.

**A5.6** Peng, J., Wei, Z., & Ye, Y. (2025). TIDFormer: Exploiting temporal and interactive dynamics makes a great
dynamic graph transformer. In *KDD 2025*. https://arxiv.org/abs/2506.00431
`[snippet only]`
Implication for us: **ignore** (L1).

**A5.7** Dizaji, A., Tjandra, B. A., Hamidi, M., Huang, S., & Rabusseau, G. (2025). *T-GRAB: A synthetic diagnostic
benchmark for learning on temporal graphs* (arXiv:2507.10183; MLoG-GenAI workshop at KDD 2025).
https://arxiv.org/abs/2507.10183
`[snippet only]`
Synthetic tasks testing whether temporal GNNs capture periodicity, delayed cause and effect, and long-range
dependencies, with 11 methods evaluated.
Implication for us: **ignore.**

### A6. Scientific-document embeddings beyond SPECTER2

**A6.1** Singh, A., D'Arcy, M., Cohan, A., Downey, D., & Feldman, S. (2023). SciRepEval / SPECTER2 (already cited).
Code and usage: https://github.com/allenai/SPECTER2
`[verified: fetched page]` (README)
Usage still loads `allenai/specter2_base` with `adapters.AutoAdapterModel` and attaches `allenai/specter2` as the
proximity adapter; the most recent note in the README is from December 2023, and no transformers version is pinned.
Implication for us: **watch.** L6 stands; the version problem is in B7.

**A6.2** Ostendorff, M., Rethmeier, N., Augenstein, I., Gipp, B., & Rehm, G. (2022). Neighborhood contrastive learning
for scientific document representations with citation embeddings. In *EMNLP 2022*. https://github.com/malteos/scincl
`[verified: fetched page]` (README)
SciNCL (`malteos/scincl`) loads with plain `AutoTokenizer`/`AutoModel` and uses the [CLS] embedding; no adapters
library is needed.
Implication for us: **watch.** It remains the right fallback in the risk register, and B7 makes that fallback more
likely to be needed.

**A6.3** *SciEmbed: Citation-context supervision for scientific document embeddings* (2026). In *Findings of the
Association for Computational Linguistics: EMNLP 2026*. https://aclanthology.org/2026.findings-emnlp.2057/
`[snippet only]` (authors not retrieved; model cards say they were withheld for double-blind review)
A 149M-parameter ModernBERT-base embedder trained with citation-context sentences as the contrastive signal; it reports
66.85 on SciRepEval and claims to exceed SPECTER2 and SciNCL, but the SciRepEval paper itself reports about 69.1 for
SPECTER2 Base, so the comparison settings need checking.
Implication for us: **watch.**

**A6.4** General-purpose embedders: Granite Embedding R2 (IBM; arXiv:2508.21085, Apache-2.0, 768-dimensional, 8,192-token
context) and Qwen3 Embedding (arXiv:2506.05176, 0.6B to 8B parameters). https://arxiv.org/abs/2508.21085,
https://arxiv.org/abs/2506.05176
`[snippet only]` (author lists not retrieved)
Neither result showed a scientific-document benchmark (SciRepEval or SciDocs) score.
Implication for us: **ignore.**

### A7. Older items the plan or the Phase A reference list misses

Kojaku et al. (2021, A1.5), Fister et al. (2016, A1.7), Liu et al. (2024, A1.6), Candal-Pedreira et al. (2024, A2.3),
Chatterjee et al. (2021, A4.2), Poštuvan et al. (2024, A3.2) and the TGB papers (A5.3). Kojaku et al. and Poštuvan et
al. bear directly on the injection design (L18, S4.5); Chatterjee et al. bears directly on the curvature-dynamics
component.

---

## Part B. Tooling and data-source changes

### B1. OpenAlex

**B1.1** OpenAlex. (2026, October 15). *Authentication*. https://help.openalex.org/api/authentication/ (source:
https://github.com/ourresearch/openalex-help/blob/main/content/api/authentication.md)
`[verified: fetched page]` (source file, `updated: 2026-10-15`)
Per-second limits are 10 requests without a key, 30 with a free key or prepaid usage, and 50 on an annual plan. A free
key is required for cursor paging past the first page (the paging page, updated 2026-10-14, says following
`next_cursor` without a key returns 401). The key goes in the `api_key` query parameter or an `Authorization: Bearer`
header and is issued at openalex.org/settings/api-key. An agent can request a key for a person's email with
`POST https://api.openalex.org/api-keys`; it works only after that person clicks the emailed activation link. A 429 is
returned both when the daily budget is spent and when requests come too fast.
Implication for us: **change plan (S0.4, S1.1).** Update the key URL in S0.4; throttle the harness to at most 30
requests per second; on a 429, call `GET /rate-limit` to tell "too fast" (retry shortly) from "budget spent" (stop and
resume after midnight UTC).

**B1.2** OpenAlex. (2026, October 26). *Pricing overview* and *Example costs*.
https://help.openalex.org/access/pricing/ (source: `content/access/pricing.md`, `content/access/example-costs.md` in
ourresearch/openalex-help)
`[verified: fetched page]` (source files)
Every account gets $1 of API usage per day, reset at midnight UTC; prepaid usage is sold in $1 steps, is used only after
the daily budget, and expires three months after the last purchase. List+filter calls cost $0.10 per 1,000 ($0.0001 per
call), searches $1 per 1,000, single-entity lookups are free, and the free daily budget covers about 10,000 list calls
(1,000,000 results at `per_page=100`). Each response's `meta` block carries `cost_usd`. Annual plans start at
$5,000 per year (Member, $20/day budget).
Implication for us: **watch.** The plan's arithmetic (about $0.0001 per page, 1 million results per day) still holds.
Two cheap improvements: record `meta.cost_usd` instead of a constant, and join Retraction Watch DOIs to OpenAlex with
`filter=doi:a|b|...` (up to 100 DOIs per list call) or free single lookups.

**B1.3** OpenAlex. (2026, October 2). *Deprecations*. https://help.openalex.org/api/deprecations/ (source:
`content/api/deprecations.md` in ourresearch/openalex-help)
`[verified: fetched page]` (source file)
The polite pool was replaced by API keys in February 2026 and the `mailto` parameter is ignored. `host_venue`,
`grants` and the `has_ngrams` filter are removed; concepts and `x_concepts` are deprecated (the latter "will be
removed soon"); `include_xpac` and the `is_xpac` filter are deprecated in favour of `corpus=`;
`sustainable_development_goals_aurora` is to be removed in November 2026. `concept.id`, `concepts.id` and `x_concepts.id`
are listed as deprecated filter aliases whose canonical form is `topics.id`.
Implication for us: **change plan (small).** `data.mailto` in `configs/default.yaml` and `pyalex.config.email` in the
harness have no effect; S0.4's Verify line should not depend on them. The harness already reads `primary_location`, not
`host_venue`.

**B1.4** OpenAlex. (2026, October 26). *Concepts (deprecated)*. https://help.openalex.org/data/concepts/ (source:
`content/data/concepts.md`)
`[verified: fetched page]` (source file)
The page states that concepts are frozen, "no longer recomputed or expanded — so new works are not tagged with them",
and that the concept tree is not maintained. The `concepts` field on works is kept for continuity.
Implication for us: **change plan (S1.1, S1.3, section 3 contract).** A concept filter would silently under-count recent
works, and the current harness stores `concepts` in `works.parquet`. Switch the slice filter to topics (for example
`primary_topic.subfield.id:` for precision, or `topics.subfield.id:` for recall) and store `primary_topic`, its subfield
and field IDs (and the up-to-three `topics`) instead of, or next to, `concepts`.

**B1.5** OpenAlex. (2026, October 26). *Topics*. https://help.openalex.org/data/topics/ (source: `content/data/topics.md`)
`[verified: fetched page]` (source file)
The hierarchy has 4 domains, 26 fields, 252 subfields and 4,516 topics. In October 2026 a new classifier replaced the
old one: it reads title, abstract and source name and does not use citations; topic names and IDs did not change, but
assignments to works did, and FWCI and citation percentiles changed for most works. Old assignments, frozen on
2026-10-05, can be downloaded from the v2.0.0 release of `ourresearch/openalex-topic-classification`; about 10% of
works have no topic.
Implication for us: **change plan (S1.3).** Record in `configs/slice.yaml` that topic membership comes from the
post-October-2026 classifier, with the pull date. A side benefit for the book: slice membership is decided from text,
not from the citation graph under study. Check how many abstract-less works the topic filter drops.

**B1.6** OpenAlex. (2026, October 23). *Expansion corpus*. https://help.openalex.org/data/works/corpus/ (source:
`content/data/works/corpus.md`)
`[verified: fetched page]` (source file)
Works now come in a curated core (320M+, the default) and an expansion of about 190M records added in the November 2025
"Walden" update; `corpus=core|expansion|all` was added in August 2026, and mixing it with the legacy `include_xpac`
returns an error.
Implication for us: **change plan (small).** Write `corpus: core` explicitly into `slice.yaml` so that recorded counts
stay reproducible.

**B1.7** ourresearch. (2026). *docs* (README) [GitHub repository]. https://github.com/ourresearch/docs
`[verified: fetched page]`
The repository was retired on 2026-08-11; developer documentation now lives at help.openalex.org, and
developers.openalex.org redirects there. The repository's `llms.txt` and the current pages agree that `per_page` is
capped at 100; the paging page adds that `per_page=200` is still accepted as legacy behaviour but deprecated.
Implication for us: **watch.** Point plan and book links at help.openalex.org; keep `page_size: 100` (the YAML comment
"live accepts 200 today" now refers to deprecated behaviour).

### B2. pyalex

de Bruin, J. (2026). *PyAlex* (Version 0.21) [Computer software]. https://pypi.org/project/pyalex/ ; issue #100:
https://github.com/J535D165/pyalex/issues/100
`[verified: fetched page]` (PyPI page and release history; `pyalex/api.py` on the main branch; issue #100)
0.21 (2026-02-23) is still the latest release. The README says an API key is required from 2026-02-13. On the main
branch, retries use urllib3 `Retry` with `max_retries=0` by default and `retry_http_codes=[429, 500, 503]`, and the key
is sent as a Bearer header (I did not confirm that the 0.21 release matches main). Issue #100 (opened 2026-03-08, still
open, fix in open PR #101) reports that pyalex discards the `X-RateLimit-*` headers and that its retries treat a spent
daily budget like a short rate limit. The README's credit numbers (100,000 credits per day, 1 credit per list call)
disagree with the OpenAlex documentation (about 10,000 list calls per day); the OpenAlex pages are the authority.
Implication for us: **change plan (refines L8).** Keep pyalex 0.21 and the custom backoff, keep pyalex's own
`max_retries` at 0 so retries do not stack, and add the `/rate-limit` check from B1.1. Fix the harness docstring:
pyalex has retry settings; the problem is that they cannot tell the two kinds of 429 apart.

### B3. Retraction Watch and Crossref

**B3.1** Crossref. (2026). *retraction-watch-data* [GitLab repository; README and commit history].
https://gitlab.com/crossref/retraction-watch-data
`[verified: fetched page]` (README and the GitLab commits API)
The README says the dataset was generated on 2026-10-30; an automated job commits a new `retraction_watch.csv` every
day. It states that Crossref acquired the database in September 2023 "and have made it publicly available", and lists
20 columns (Record ID through Notes); changes to field names and vocabulary are recorded by Retraction Watch in its user
guide (Appendix D).
Implication for us: **watch.** L9 stands. The plan calls it a "19-column CSV"; the README lists 20. `rwdb.py` checks
only a required subset of columns, so nothing breaks, but the wording should be fixed.

**B3.2** Crossref. (2025, January 29). *Retraction Watch retractions now in the Crossref API* [Blog post].
https://www.crossref.org/blog/retraction-watch-retractions-now-in-the-crossref-api/
`[snippet only]`
Retraction Watch retractions and corrections appear in the `update-to` field of Crossref REST API work records, with a
`source` of `publisher` or `retraction-watch`; the `record-id` matches the Record ID in the CSV, and
`filter=update-type:retraction` lists retractions. The Labs API annotations are deprecated, and Crossref asks users to
cite the source.
Implication for us: **watch.** A second way to check retraction status per DOI, but without the Reason field, so the
GitLab CSV stays the label source. Add the citation request to the data-sources note.

**B3.3** Retraction Watch. (n.d.). *Retraction Watch Database user guide, Appendix D: Changes*.
https://retractionwatch.com/retraction-watch-database-user-guide/retraction-watch-database-user-guide-appendix-d-changes/
`[snippet only]`
The latest entry seen in snippets is dated 2025-02-28 (definition change for "Cites Retracted Work"); no 2026 entry was
visible.
Implication for us: **watch.** Re-check Appendix D before freezing the final label snapshot for S4.1, since the
Referencing/Attributions literal is the positive-label rule (L12).

### B4. GraphRicciCurvature

GraphRicciCurvature maintainers. (2026). *GraphRicciCurvature* (Version 0.6.1) [Computer software].
https://pypi.org/project/GraphRicciCurvature/ ; https://github.com/saibalmars/GraphRicciCurvature
`[verified: fetched page]` (PyPI release history and metadata, `CHANGELOG.md`, the releases page, `pyproject.toml`,
`OllivierRicci.py`, `FormanRicci.py`, issue list)
After two years without releases, 0.5.4 (2026-08-13), 0.6.0 and 0.6.1 (both 2026-08-24) appeared. 0.5.4 fixes weighted
Ollivier-Ricci normalisation to use the graph shortest-path metric and batches pairwise shortest-path queries; 0.6.0 adds
Balanced Forman curvature; 0.6.1's notes list only the shortest-path batching. The source still takes predecessors at
the source node and successors at the target for directed graphs, still runs its pool with `fork` only, and every release
including 0.6.1 still requires `scipy<=1.13.1`. SciPy 1.13.1 publishes wheels only for Python 3.9 to 3.12. No issues
are open.
Implication for us: **change plan (L5 wording and pins).** "Dormant upstream" is no longer true. Keep the 0.5.3.2 pin for
reproducibility; our ORC is unweighted, so the 0.5.4 fix should not change our values, but S1.5 could run both 0.5.3.2
and 0.6.1 on the toy graphs to confirm. Upgrading does not remove the SciPy ceiling, so the curvature environment needs
Python 3.12 (see B5).

### B5. Google Colab runtime: Python 3.13

Google Colaboratory. (2026, August 18). *Upgrade to Python 3.13* (Issue #6081) [GitHub issue].
https://github.com/googlecolab/colabtools/issues/6081 ; Google Colaboratory. (2026). *backend-info* [GitHub
repository; `os-info.txt`, `pip-freeze.txt`, commit history]. https://github.com/googlecolab/backend-info
`[verified: fetched page]`
The issue announced the change, a maintainer marked it live on 2026-08-25, and it was closed on 2026-09-07; users who
need Python 3.12 can choose runtime version 26.07, 26.04 or older through "Change runtime version", but the choice does
not persist across sessions. backend-info (latest commits late October 2026; the files carry a note that they may not
match the production container exactly) lists Python 3.13.18, torch 2.11.0, transformers 5.22.0, numpy 2.4.6,
scipy 1.16.3, networkx 3.7 and xgboost 3.4.1. I found no statement of how long the 3.12 runtimes remain available.
Implication for us: **change plan (global constraint, section 9.1, S0.5).** "Python 3.12 everywhere (Colab default)" is
no longer true. Either every notebook's first cell checks `sys.version_info[:2] == (3, 12)` and tells the user to select
runtime 26.07, or the stage pins are re-tested on Python 3.13. The curvature and CurvGAD stages cannot move to 3.13
while `scipy<=1.13.1` holds (B4, A4.1).

### B6. Colab benefits inside Google AI plans

Google for Developers. (2026). *Colab is now part of your Google AI plan* [Blog post].
https://developers.googleblog.com/colab-is-now-part-of-your-google-ai-plan/
`[snippet only]` (Google blog and news sites were blocked)
Reports dated around 2026-09-22 say that Google AI Plus and Pro subscribers get priority access to faster accelerators,
and that Google AI Ultra adds Premium GPUs and background execution. Standalone Colab Pro and Pro+ continue, and one
report says compute units from both kinds of subscription add up. No official time limit for Ultra background execution
was found.
Implication for us: **watch.** Gate G1 (Pro vs Pro+) gains options, but background execution, the reason G1 exists, is
reported only for Ultra among the Google AI plans. Confirm on Google's official page before anyone buys.

### B7. adapters and transformers (SPECTER2)

AdapterHub. (2026). *adapters* (Version 1.3.0) [Computer software]. https://pypi.org/project/adapters/ ;
https://github.com/adapter-hub/adapters/releases ; Hugging Face. (2026). *transformers* [PyPI release history].
https://pypi.org/project/transformers/
`[verified: fetched page]` (PyPI JSON metadata, the adapters `setup.py` on main, the releases page)
adapters 1.3.0 (2026-04-26) declares `transformers~=4.57.6` (at least 4.57.6 and below 4.58), and its release notes say
it is built for transformers 4.57.x; the main branch has the same requirement, and no release mentions transformers 5.
transformers 5.0.0 came out on 2026-01-26 and 5.22.1 on 2026-10-30; 4.57.6 (2026-01-16) is the last 4.57 release.
Implication for us: **change plan (pins).** In `configs/pins.md` replace "transformers ~4.57.x" with the exact pair
`adapters==1.3.0`, `transformers==4.57.6`, installed in the embed runtime only; Block C should print the resolved
versions. Keep the SciNCL fallback (A6.2).

### B8. PyGOD and PyTorch Geometric

PyGOD Team. (2024). *pygod* (Version 1.1.0) [Computer software]. https://pypi.org/project/pygod/ ; PyG Team. (2026).
*torch_geometric* [PyPI release history]. https://pypi.org/project/torch-geometric/
`[verified: fetched page]` (PyPI)
PyGOD 1.1.0 (2024-02-04) is still the latest release; its classifiers stop at Python 3.11, and torch and torch_geometric
must be installed separately. torch_geometric has released 2.7.0 (2025-10-15), 2.8.0 (2026-06-05) and 2.8.1
(2026-10-12).
Implication for us: **watch.** No PyGOD change. Block D will show whether PyG 2.6 still works with the torch of the
chosen runtime; if not, try a newer PyG and record it.

### B9. Other pinned packages

PyPI records for networkit, POT, xgboost and scikit-learn. https://pypi.org/project/networkit/ ,
https://pypi.org/project/POT/ , https://pypi.org/project/xgboost/ , https://pypi.org/project/scikit-learn/
`[verified: fetched page]` (PyPI JSON)
networkit 11.2.1 (2026-01-09) has wheels for Python 3.10 to 3.13 (11.2.2 followed on 2026-09-15); POT 0.9.6 has wheels
for 3.9 to 3.13 (0.9.7.post1 is newest); xgboost 3.4.1 requires Python 3.12 or later; scikit-learn 1.9.1 was released
on 2026-09-10.
Implication for us: **ignore.** These pins remain installable.

---

## (i) Ranked actions for the project

1. **Fix the Colab Python version before S0.5 runs (B5).** Add a first-cell check for Python 3.12 to every notebook,
   with the instruction to pick runtime version 26.07 (the choice does not persist), and amend the global constraint
   and section 9.1. Without this, Block A tries to build SciPy 1.13.1 from source on Python 3.13.
2. **Patch vendored DyGLib before Block B (A5.1).** `random.sample(sorted(test_node_set), ...)` in
   `utils/DataLoader.py`, plus the `NODE_FEAT_DIM`/`EDGE_FEAT_DIM` patch, both recorded in `PATCHES.md`.
3. **Move the slice and the works contract from concepts to topics (B1.4, B1.5).** Filter on subfield-level topic
   IDs, add `primary_topic` and its subfield and field IDs to `works.parquet`, and write the classifier date, the pull
   date and `corpus: core` into `slice.yaml` (needs a plan edit to section 3 and S1.3).
4. **Update the OpenAlex harness (B1.1, B1.3, B2).** Throttle to at most 30 requests per second, use `GET /rate-limit`
   after a 429 to separate "too fast" from "budget spent", stop until midnight UTC on the latter, log `meta.cost_usd`,
   drop the reliance on `mailto`, and fix the S0.4 key URL.
5. **Pin the SPECTER2 stack exactly (B7).** `adapters==1.3.0` with `transformers==4.57.6` in the embed runtime; keep
   SciNCL as the tested fallback.
6. **Write the CurvGAD sealed-environment recipe now (A4.1).** Strip standard-library names from its requirements, pin
   GraphRicciCurvature 0.5.3.2, use Python 3.12 and a DGL build matching torch; fall back to published numbers if
   needed.
7. **Re-check gate G1 against the Google AI plan option (B6)** once the official terms are confirmed; background
   execution is reported only for Ultra.
8. **Add a second synthetic anomaly type to S4.5 (A3.2):** Poštuvan et al.'s temporal-structural-contextual link
   anomalies (MIT code), next to the CIDRE cartel injector. PROVISIONAL until Maor agrees.
9. **Decide whether to add SLADE (through BAG or its own code) as a dynamic-graph anomaly baseline (A3.1).** This
   changes L17, so it needs Maor's decision.
10. **Add the missing references to the book track (S0.7)** after checking each on the publisher page: Kojaku et al.
    (2021), Chatterjee et al. (2021), Poštuvan et al. (2024), Hua et al. (2026), Candal-Pedreira et al. (2024), Liu et al.
    (2024), Fister et al. (2016), the TGB papers, and Scancar et al. (2026).

## (ii) Items that change a locked decision, a contract or a pin

- **Global constraint "Python 3.12 everywhere (Colab default)" and section 9.1:** Colab's default is Python 3.13 since
  2026-08-25 (B5). Python 3.12 is available only by selecting an older runtime version in every session.
- **Section 9.3 pins, curv env:** `scipy<=1.13.1` (required by every GraphRicciCurvature release up to 0.6.1) has no
  Python 3.13 wheel (B4). The pin stays valid only on a Python 3.12 runtime.
- **Section 9.3 pins, embed env:** "transformers ~4.57.x" should read `transformers==4.57.6` with `adapters==1.3.0` (B7).
- **L1 (DyGLib TGN):** decision stands, but DyGLib is unmaintained (last commit 2023-09-28) and needs a Python 3.11+
  patch (A5.1); TGM is the maintained fallback if torch 2.11 breaks it (A5.2).
- **L5 (GraphRicciCurvature 0.5.3.2, "dormant upstream"):** the pin can stay, but the "dormant" justification is out of
  date: three releases appeared in August 2026 (B4).
- **L7 (OpenAlex, free key, $1/day):** still correct. Details changed: `mailto` ignored, 30 requests per second, cursor
  paging requires a key, `per_page=200` deprecated (B1.1, B1.3, B1.7).
- **L8 (pyalex 0.21 plus custom 429 backoff):** stands; the backoff must also handle budget exhaustion (B2).
- **L10 (Colab primary):** stands, with the runtime-version step above; the Google AI plan options in B6 should be
  re-checked under G1.
- **L11 (slice by "concept × year-range"):** must become topic (subfield) × year, because concepts are frozen and new
  works are no longer tagged (B1.4); topic assignments changed in October 2026 (B1.5).
- **Section 3 contract, `works.parquet`:** the `concepts` column should be replaced or supplemented by `primary_topic`,
  `primary_subfield`, `primary_field` (B1.4).
- **L17 (baseline set):** unchanged unless Maor adds SLADE (A3.1). The CurvGAD sealed environment needs the recipe in
  A4.1 to be buildable at all.
- **L18 (synthetic injection):** unchanged unless the second anomaly type from A3.2 is adopted.
- **L9 and S0.6 wording:** the Retraction Watch CSV has 20 columns per its README, not 19 (B3.1); no code change.

## (iii) What I could not verify

- Every item tagged `[snippet only]`, which includes all arXiv, OpenReview, AAAI, ACM, Springer, Nature, BMJ, PNAS and
  JMLR pages: those hosts were blocked, so author lists, venues and numbers come from search snippets or secondary
  summaries. Author lists are missing for A1.3, A2.5, A3.5, A4.7, A6.3 and A6.4, and incomplete for A2.3 and A3.3.
- The official Google page on Colab in Google AI plans (B6), and current Colab Pro/Pro+ prices and unit allowances.
- How long Colab keeps the Python 3.12 runtime versions available (B5).
- Whether SciPy 1.13.1 would build from source on Colab's Python 3.13, whether DyGLib runs on torch 2.11, whether PyG 2.6
  and PyGOD 1.1.0 work with the chosen runtime's torch, and whether `adapters==1.3.0` with `transformers==4.57.6`
  installs cleanly next to Colab's preinstalled packages. None of these were executed; S0.5 should answer them.
- Whether the pyalex 0.21 release behaves like the main-branch source I read (Bearer header, retry defaults).
- The live openalex.org/pricing page; the OpenAlex figures above come from OpenAlex's documentation source files on
  GitHub, which may lag or lead the deployed site by days.
- The BAG repository's licence (none stated), the content of "Beat the counter first" (A3.5), and the GitHub issue
  #23 claim about DyGLib negative sampling (a user report, not reproduced).

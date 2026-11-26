"""Short definitions of the four score components shown in the explorer (NFR10 traceability).

This is the ONE place the explorer takes this text from. It restates the definitions in docs/score_components.md
(task S3.1, decisions D1-D12) as implemented in src/citecurv/scoring/components.py. That note awaits Maor's
review, so every entry is provisional. When the note or the code changes, edit this dict (and keep `STATUS`
honest).
"""

STATUS = ("PROVISIONAL: restates docs/score_components.md (S3.1, awaiting Maor's review) as implemented in "
          "citecurv.scoring.components; the note is authoritative.")

COMPONENT_TEXT = {
    "s_embed": {
        "name": "Embedding surprise",
        "formula": "s_embed = -ln p_link(u, v, t)   (fallback static_cosine: (m_cell - cos(x_u, x_v)) / sd_cell)",
        "text": (
            "Negative log-probability of the observed citation under the TGN link decoder, computed from memory "
            "before this event's batch updates it (p clipped to [1e-7, 1 - 1e-7]). High when the temporal model "
            "did not expect u to cite v at t. Before the TGN exists (rehearsal), the static_cosine fallback scores "
            "how much less similar the two SPECTER2 vectors are than earlier citations of the same (field, year) "
            "cell (D4)."
        ),
    },
    "s_geom": {
        "name": "Geometric anomaly",
        "formula": ("orc_dir formulation (full, no_temporal, full_learned from 2026-11-26): s_geom = -z(orc_dir), "
                    "orc_dir = -(1 - alpha)(1 - n_shared / n_refs_v). Plan formulation (forman_only, plan_full, "
                    "runs before 2026-11-26): s_geom = |z_forman|, or max(|z_forman|, |z_orc|) when escalated"),
        "text": (
            "orc_dir formulation: the directed Ollivier-Ricci curvature of the citation u -> v at t, computed by its "
            "closed form from the share of v's references that u also cites (docs/curvature_formulation_study.md, "
            "Proposition 2; exact when u has no citer at t, flagged otherwise), z-scored against earlier events "
            "(ts < t) of the citing paper's (field, year) cell with fallback to the field, then all events. "
            "One-sided: lower curvature than the cell (few shared references) scores higher; NaN when v's "
            "reference list is unknown. Plan formulation: z-score of the directed Forman screen "
            "F = 2 - in_deg_t(u) - out_deg_t(v) against the same kind of reference, escalated events also with "
            "their ORC z-score, s_geom the larger absolute value (D1, D2, D5); at event time F mostly equals 2 "
            "minus the cited paper's reference count (Proposition 1)."
        ),
    },
    "s_temp": {
        "name": "Temporal anomaly",
        "formula": "s_temp = z_cell(ln((n_recent + 1) / (n_before + 1)) + ln(T_before / W)),  W = 90 days",
        "text": (
            "The cited paper's citation rate in the last W = 90 days relative to its own earlier rate (counts of its "
            "citations with ts <= t, this event excluded), z-scored against earlier events (ts < t) of the citing "
            "paper's (field, year) cell. A steadily popular paper scores near 0; a sudden jump against its own "
            "history scores high (D6)."
        ),
    },
    "s_curvdyn": {
        "name": "Curvature dynamics",
        "formula": ("orc_dir formulation: s_curvdyn = (ridge(phi(z_u, z_v)) - orc_dir) / RMSE_train. "
                    "Plan formulation: s_curvdyn = |asinh(F) - ridge(phi(z_u, z_v))| / RMSE_train"),
        "text": (
            "Residual of a ridge regression that predicts the curvature from the two endpoint vectors (TGN "
            "embeddings at t, else SPECTER2), reduced by PCA to 32 dimensions each plus their product and cosine. "
            "PCA, scaling, coefficients and RMSE are fitted on training rows only (D7). orc_dir formulation: the "
            "target is orc_dir and the residual is signed, high when the observed curvature is lower than the "
            "papers' representations predict. Plan formulation: the target is asinh(Forman screen), mostly "
            "asinh(2 - the cited paper's reference count) at event time, and the residual is absolute."
        ),
    },
}

COMBINER_TEXT = (
    "total = sum over components of w_c * z_c, with z_c = (s_c - mean_c) / sd_c. The z statistics are fitted on "
    "training rows only and frozen; the weights lie on the simplex and are per-run artifacts (S3.2)."
)

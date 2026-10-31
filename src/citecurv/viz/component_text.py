"""Short definitions of the four score components shown in the explorer (NFR10 traceability).

This is the ONE place the explorer takes this text from. It restates the proposed defaults of IMPLEMENTATION_PLAN.md
task S3.1, which docs/score_components.md fixes after Maor's review; until that note is approved every entry is
provisional. When the note changes, edit this dict (and keep `status` honest).
"""

STATUS = "PROVISIONAL: proposed defaults from IMPLEMENTATION_PLAN.md S3.1; docs/score_components.md is authoritative once approved."

COMPONENT_TEXT = {
    "s_embed": {
        "name": "Embedding surprise",
        "formula": "s_embed = -log p_link(u, v, t)",
        "text": (
            "Negative log-probability of the observed citation under the TGN link decoder, computed from memory "
            "before this event's batch updates it. High when the temporal model did not expect u to cite v at t."
        ),
    },
    "s_geom": {
        "name": "Geometric anomaly",
        "formula": "s_geom = |F(u->v, t) - mean_cell| / sd_cell",
        "text": (
            "Absolute z-score of the event's directed Forman screen value F = 2 - in_deg_t(u) - out_deg_t(v) "
            "(Ollivier-Ricci where escalated) against the reference distribution of its (field, year) cell, "
            "built from events with ts <= t only."
        ),
    },
    "s_temp": {
        "name": "Temporal anomaly",
        "formula": "s_temp = deviation of v's citing-gap pattern from the (field, year) norm",
        "text": (
            "How unusual the timing is: time since v's previous citation and the burst count in a trailing window, "
            "compared with the norm of the cell. The exact statistic is fixed in the S3.1 note."
        ),
    },
    "s_curvdyn": {
        "name": "Curvature dynamics",
        "formula": "s_curvdyn = |F_pred(z_u, z_v) - F_obs(u->v, t)|",
        "text": (
            "Residual between the edge curvature predicted from the endpoint temporal embeddings at t (a small "
            "MLP, CurvGAD-style) and the observed screen curvature. High when geometry and representation disagree."
        ),
    },
}

COMBINER_TEXT = (
    "total = sum over components of w_c * z_c, with z_c = (s_c - mean_c) / sd_c. The z statistics are fitted on "
    "training rows only and frozen; the weights lie on the simplex and are per-run artifacts (S3.2)."
)

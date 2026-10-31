"""Results explorer: a self-contained offline HTML page over the pipeline outputs (FR7-FR9, NFR7, NFR10)."""

from citecurv.viz.demo import make_demo_data
from citecurv.viz.explorer import build_explorer, explorer_payload, iframe_html, render_html, show

__all__ = ["build_explorer", "explorer_payload", "iframe_html", "make_demo_data", "render_html", "show"]

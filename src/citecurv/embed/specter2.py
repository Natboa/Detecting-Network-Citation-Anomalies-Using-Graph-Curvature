"""SPECTER2 encoding: encode(papers, batch_size=32) -> (N, 768) float32.

base model allenai/specter2_base + proximity adapter allenai/specter2; title-only fallback (FR3).
Implemented in task S1.6.
"""


def encode(papers, batch_size=32):
    raise NotImplementedError("task S1.6")

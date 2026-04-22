"""Retrieval-Aware Cross-Lingual Query Rewriting for Korean Questions.

This package provides the building blocks for comparing four retrieval query
strategies under a fixed retriever and corpus:

- ``raw``: use the original Korean question as the retrieval query.
- ``translate``: use a provided English target query as a translation baseline.
- ``supervised``: use a supervised seq2seq rewriter.
- ``retrieval_aware``: use a retrieval-aware seq2seq rewriter.
"""

from __future__ import annotations

__all__ = [
    "__version__",
]

__version__ = "0.1.0"

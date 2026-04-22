"""Fixed BM25-style retriever for the cross-lingual rewrite project.

The retriever is built from the standard library only. It is intentionally
simple so that:

* every method runs against the same deterministic scoring function;
* there are no hidden dependencies on heavyweight search backends;
* Korean and English inputs can both produce meaningful matches.

Korean text has no word separators, so we produce two families of tokens:

1. ASCII alphanumeric runs, lowercased. These act as English tokens and also
   catch latinized code identifiers inside Korean questions (e.g. ``BM25``).
2. Hangul syllable unigrams **and** bigrams. Character n-grams are a standard
   poor-man's stand-in for word segmentation when a morphological analyzer is
   unavailable, and they still provide useful term frequencies for BM25.

The retriever does not update state once constructed.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Sequence

from .data import CorpusDocument, RetrievedDocument


_ENGLISH_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_]*")


def _is_hangul_syllable(ch: str) -> bool:
    codepoint = ord(ch)
    if 0xAC00 <= codepoint <= 0xD7A3:
        return True
    if 0x1100 <= codepoint <= 0x11FF:
        return True
    if 0x3130 <= codepoint <= 0x318F:
        return True
    return False


def tokenize(text: str) -> list[str]:
    """Tokenize text into case-folded English tokens and Korean n-grams.

    Returns a list in insertion order. Duplicate tokens are preserved so
    downstream term frequencies are accurate.
    """

    if not text:
        return []
    lower = text.lower()
    tokens: list[str] = list(_ENGLISH_TOKEN_RE.findall(lower))

    # Korean: syllable unigrams + adjacent bigrams from contiguous runs.
    current_run: list[str] = []

    def _flush(run: list[str]) -> None:
        if not run:
            return
        tokens.extend(run)
        for i in range(len(run) - 1):
            tokens.append(run[i] + run[i + 1])

    for ch in text:
        if _is_hangul_syllable(ch):
            current_run.append(ch)
        else:
            _flush(current_run)
            current_run = []
    _flush(current_run)
    return tokens


@dataclass(frozen=True)
class _IndexedDoc:
    doc_id: str
    text: str
    source_language: str | None
    title: str | None
    length: int
    term_freq: Counter[str]


class BM25Retriever:
    """A deterministic BM25-style retriever over a fixed corpus.

    Parameters
    ----------
    documents:
        The corpus to index. The retriever keeps references to these objects
        and returns them (as :class:`RetrievedDocument` wrappers) at query
        time.
    k1:
        BM25 ``k1`` parameter. Default 1.5.
    b:
        BM25 length-normalization parameter. Default 0.75.
    """

    def __init__(
        self,
        documents: Sequence[CorpusDocument],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if k1 <= 0:
            raise ValueError("k1 must be positive")
        if not 0.0 <= b <= 1.0:
            raise ValueError("b must be between 0 and 1")
        self._k1 = k1
        self._b = b
        self._indexed: list[_IndexedDoc] = []
        doc_freq: Counter[str] = Counter()
        for doc in documents:
            field_text = " ".join(filter(None, [doc.title, doc.text]))
            tokens = tokenize(field_text)
            tf = Counter(tokens)
            for term in tf:
                doc_freq[term] += 1
            self._indexed.append(
                _IndexedDoc(
                    doc_id=doc.doc_id,
                    text=doc.text,
                    source_language=doc.source_language,
                    title=doc.title,
                    length=len(tokens),
                    term_freq=tf,
                )
            )
        self._doc_freq = doc_freq
        self._avgdl = (
            sum(d.length for d in self._indexed) / len(self._indexed)
            if self._indexed
            else 0.0
        )
        self._num_docs = len(self._indexed)
        self._idf: dict[str, float] = {}
        if self._num_docs:
            for term, df in self._doc_freq.items():
                self._idf[term] = math.log(
                    1 + (self._num_docs - df + 0.5) / (df + 0.5)
                )

    def __len__(self) -> int:
        return self._num_docs

    @property
    def num_documents(self) -> int:
        return self._num_docs

    def document_ids(self) -> list[str]:
        return [d.doc_id for d in self._indexed]

    def score(self, query: str, doc_id: str) -> float:
        """Return the BM25 score of ``doc_id`` for the given query."""

        for doc in self._indexed:
            if doc.doc_id == doc_id:
                query_tokens = tokenize(query)
                return self._score_document(query_tokens, doc)
        raise KeyError(f"Unknown doc_id: {doc_id}")

    def _score_document(self, query_tokens: Iterable[str], doc: _IndexedDoc) -> float:
        if not self._num_docs or doc.length == 0:
            return 0.0
        score = 0.0
        norm = 1 - self._b + self._b * (doc.length / max(self._avgdl, 1e-9))
        seen: set[str] = set()
        for term in query_tokens:
            if term in seen:
                continue
            seen.add(term)
            idf = self._idf.get(term)
            if idf is None:
                continue
            tf = doc.term_freq.get(term, 0)
            if tf == 0:
                continue
            numerator = tf * (self._k1 + 1)
            denominator = tf + self._k1 * norm
            score += idf * (numerator / denominator)
        return score

    def retrieve(
        self,
        query: str,
        *,
        top_k: int,
        positive_doc_id: str | None = None,
    ) -> list[RetrievedDocument]:
        """Retrieve the top ``top_k`` documents for ``query``.

        Ties on score are broken by ascending ``doc_id`` so the ranking is
        stable across runs.
        """

        if top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        if not self._num_docs:
            return []

        query_tokens = tokenize(query)
        scored: list[tuple[float, str, _IndexedDoc]] = []
        for doc in self._indexed:
            score = self._score_document(query_tokens, doc)
            scored.append((score, doc.doc_id, doc))
        # Stable sort: highest score first, then ascending doc_id.
        scored.sort(key=lambda item: (-item[0], item[1]))
        results: list[RetrievedDocument] = []
        for rank, (score, _doc_id, doc) in enumerate(scored[:top_k], start=1):
            results.append(
                RetrievedDocument(
                    doc_id=doc.doc_id,
                    text=doc.text,
                    rank=rank,
                    score=float(score),
                    source_language=doc.source_language,
                    is_positive=(positive_doc_id is not None and doc.doc_id == positive_doc_id),
                )
            )
        return results


__all__ = [
    "BM25Retriever",
    "tokenize",
]

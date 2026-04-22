# Final Report - Retrieval-Aware Cross-Lingual Query Rewriting

> Fill in each section. Numbers come from `output/analysis/comparison.csv`,
> `output/analysis/slice_summary.csv`, and `output/analysis/failure_cases.jsonl`
> produced by `scripts/compare_runs.py`.

## 1. Problem statement

- **Task.** Answer Korean questions by retrieving from a corpus that contains
  English technical documents (and Korean distractors). A good English
  retrieval query should surface the intended English document even when the
  user types Korean.
- **Hypothesis.** A retrieval-aware query rewriter should beat a pure
  supervised rewriter, and both should beat the translate baseline and the
  raw Korean query.

## 2. Methods compared

| Method            | Description                                                           |
|-------------------|-----------------------------------------------------------------------|
| `raw`             | Use `question_ko` directly as the retrieval query.                    |
| `translate`       | Use the gold `target_query` (translate-then-retrieve upper bound).    |
| `supervised`      | Fine-tune a seq2seq model to map `question_ko -> target_query`.       |
| `retrieval_aware` | Supervised objective plus a BM25-margin retrieval loss term.          |

All four methods share the same fixed retriever (`retriever.py`) and the same
corpus (`data/smoke_corpus.jsonl`). Only the retrieval query changes.

## 3. Experimental setup

- **Retriever.** Fixed BM25-style retriever with English word tokens and
  Korean character unigrams + bigrams. Parameters: `k1=1.5`, `b=0.75`.
- **Top-k.** `<fill in from the config you used>`.
- **Seed.** `<fill in>`.
- **Training.** `<mock | real; epochs; batch size; learning rate>`.
- **`retrieval_loss_weight`.** `<fill in>`.
- **Hardware.** `<CPU / GPU>`.

## 4. Aggregate results

Paste the relevant rows from `output/analysis/comparison.csv`:

| run_id | method | example_count | error_count | recall_at_k | mrr | ndcg | source_diversity | english_source_ratio | korean_source_ratio | faithfulness |
|--------|--------|---------------|-------------|-------------|-----|------|------------------|----------------------|---------------------|--------------|
| ...    | raw    | ...           | ...         | ...         | ... | ...  | ...              | ...                  | ...                 | ...          |
| ...    | translate | ...        | ...         | ...         | ... | ...  | ...              | ...                  | ...                 | ...          |
| ...    | supervised | ...       | ...         | ...         | ... | ...  | ...              | ...                  | ...                 | ...          |
| ...    | retrieval_aware | ...  | ...         | ...         | ... | ...  | ...              | ...                  | ...                 | ...          |

### Ranking summary

- Best recall@k: `<method>`
- Best MRR: `<method>`
- Best faithfulness: `<method>`
- Biggest delta vs. raw: `<method>` (`<metric>` improved by `<delta>`).

## 5. Slices

From `output/analysis/slice_summary.csv`, highlight at least the `pure_ko`
vs. `mixed_ko_en` slices:

| method | query_type | example_count | recall_at_k | mrr | ndcg |
|--------|------------|---------------|-------------|-----|------|
| ...    | pure_ko    | ...           | ...         | ... | ...  |
| ...    | mixed_ko_en| ...           | ...         | ... | ...  |

Observations:

- `<describe where the gap between raw and rewriters is largest>`.
- `<describe whether retrieval_aware > supervised on any slice>`.

## 6. Failure modes

From `output/analysis/failure_cases.jsonl` counts (`report.md` already shows
a summary table):

| method | positive_not_retrieved | low_mrr | low_source_diversity | low_faithfulness | empty_query | runtime_error |
|--------|------------------------|---------|----------------------|------------------|-------------|---------------|
| raw    | ...                    | ...     | ...                  | ...              | ...         | ...           |
| translate | ...                 | ...     | ...                  | ...              | ...         | ...           |
| supervised | ...                | ...     | ...                  | ...              | ...         | ...           |
| retrieval_aware | ...           | ...     | ...                  | ...              | ...         | ...           |

Pick 2-3 representative failure cases and explain them:

- `<example_id>` (`<method>`): generated query was `<query>`, target was
  `<target_query>`. The retriever ranked `<top-1 doc_id>` first instead of
  `<positive_doc_id>`. Probable reason: `<explanation>`.

## 7. Limitations

- Local smoke data is not a real benchmark; results should not be used to
  make claims about the general quality of any method.
- Mock model mode reuses the gold `target_query`; supervised and
  retrieval-aware numbers in smoke mode therefore approximate the
  `translate` baseline and do not reflect actual model quality.
- The faithfulness metric is a lexical Jaccard proxy. Replace with a
  semantic evaluator (for example, sentence-embedding cosine, GPT-as-judge,
  or a learned faithfulness classifier) before reporting faithfulness in a
  published comparison.
- The retriever is deliberately fixed to make the comparison fair across
  methods, but BM25-style scoring is not representative of production
  retrieval systems.

## 8. Next steps

- `<swap in a larger Korean QA dataset>`.
- `<replace faithfulness proxy with a semantic evaluator>`.
- `<re-run training in non-mock mode on GPU>`.
- `<compare with a dense retriever or a hybrid retriever>`.

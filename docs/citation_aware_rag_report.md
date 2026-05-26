# Citation-Aware Cross-Lingual RAG 실험 설계 보고서

## 1. 연구 목표

본 프로젝트의 목표는 한국어 또는 한국어-영어 혼합 질문이 입력되었을 때, 단순히 질문을 영어로 번역하는 것이 아니라 답변에 실제로 도움이 되는 citation을 더 잘 찾는 검색 전략을 학습하는 것이다.

기존 접근은 보통 다음과 같다.

```text
Korean query
-> machine translation
-> English query
-> retrieval
-> answer generation with citations
```

이 방식은 cross-lingual retrieval의 기본 baseline으로는 유용하지만, citation 품질을 직접 최적화하지는 않는다. 번역된 query가 문법적으로 자연스럽더라도, 좋은 citation을 찾기 위한 entity, alias, answer type, source preference, 관련 키워드가 충분히 포함되지 않을 수 있다.

따라서 본 프로젝트는 다음 구조를 목표로 한다.

```text
Korean/Mixed question
-> citation-seeking search plan
-> hybrid retrieval
-> reranking
-> answer/citation quality evaluation
```

핵심 비교 목표는 다음이다.

```text
Raw Korean/Mixed retrieval
<
Machine-translated retrieval
<=
Fine-tuned citation-seeking planner
```

즉, 최종적으로 증명하고 싶은 것은 단순 번역 query보다 citation retrieval에 최적화된 search plan이 더 좋은 근거 문서를 찾아낸다는 점이다.

## 2. 핵심 아이디어

우리가 만들려는 모델은 번역 모델이 아니라 citation-seeking search planner이다.

번역 모델은 질문의 의미를 다른 언어로 옮긴다. 반면 citation-seeking planner는 질문에 답하기 위해 어떤 검색어를 어떤 언어로, 어떤 source preference를 가지고 검색해야 하는지를 계획한다.

예를 들어 입력이 다음과 같다고 하자.

```text
레이커스가 마지막으로 플레이오프에 진출한 때는?
```

단순 번역 query는 다음에 가깝다.

```text
When did the Lakers last make the playoffs?
```

하지만 citation retrieval에 더 적합한 search plan은 다음처럼 구성될 수 있다.

```json
{
  "queries": [
    "Los Angeles Lakers last playoff appearance 2013 NBA season",
    "Lakers postseason drought 2014 2019",
    "레이커스 마지막 플레이오프 진출"
  ],
  "entities": ["Los Angeles Lakers", "NBA playoffs"],
  "aliases": ["Lakers", "LA Lakers"],
  "answer_type": "date",
  "preferred_source_languages": ["en", "ko"],
  "source_priority": ["official", "encyclopedic", "news"]
}
```

이 search plan은 단순 번역보다 더 많은 정보를 포함한다.

- 질문의 핵심 entity
- 영어 alias
- 한국어 fallback query
- 답변 유형
- citation으로 적합한 source 종류
- 여러 방향으로 분기된 검색 query

따라서 모델이 학습해야 하는 것은 자연스러운 영어 번역문이 아니라 좋은 citation을 찾는 search strategy이다.

## 3. 전체 파이프라인

전체 파이프라인은 다음 단계로 구성된다.

```text
1. 질문 데이터 준비
2. candidate search plan 생성
3. hybrid retrieval 실행
4. reranking
5. human-in-the-loop citation labeling
6. citation scoring
7. SFT/preference dataset 생성
8. search planner fine-tuning
9. fine-tuned planner 평가
10. 최종 comparison report 생성
```

각 단계는 독립적인 artifact를 생성한다. 이를 통해 중간 결과를 검토하고, 사람이 citation 품질을 직접 라벨링한 뒤, 그 feedback을 다시 학습 데이터로 사용할 수 있다.

## 4. 데이터 구성

현재 사용하는 기본 데이터는 다음과 같다.

```text
data/combined_ko_train.jsonl
data/wiki_en_corpus_300k.jsonl
```

`combined_ko_train.jsonl`은 질문 단위 dataset이다. 각 row는 다음 정보를 포함한다.

```text
question_ko
target_query
positive_doc_id
negative_doc_id
metadata.answers
metadata.positive_candidates
metadata.negative_candidates
```

여기서 `target_query`는 더 이상 최종 목표가 아니다. 과거에는 모델이 `target_query`를 복원하도록 학습했지만, 현재 목표에서는 `target_query`는 하나의 참고 후보 또는 upper-bound reference로만 사용한다.

`wiki_en_corpus_300k.jsonl`은 retrieval corpus이다. 현재는 영어 Wikipedia 300k 문서를 대상으로 시작하지만, 새 schema는 이후 Korean Wikipedia, web URLs, news, official source까지 확장할 수 있도록 설계한다.

최종 citation candidate row는 다음 형태를 갖는다.

```json
{
  "question_id": "...",
  "question": "...",
  "query_type": "pure_ko",
  "candidate_id": "...",
  "search_plan": {
    "method": "entity_expand",
    "queries": ["..."],
    "entities": ["..."],
    "aliases": ["..."],
    "answer_type": "date",
    "preferred_source_languages": ["en", "ko"],
    "source_priority": ["encyclopedic", "official", "news"]
  },
  "positive_doc_id": "...",
  "negative_doc_id": "...",
  "answers": ["..."],
  "citations": [
    {
      "doc_id": "...",
      "chunk_id": "...",
      "title": "...",
      "url": "...",
      "language": "en",
      "text": "...",
      "rank": 1,
      "retriever_scores": {},
      "rerank_score": 0.0,
      "support_label": "supported"
    }
  ],
  "metrics": {},
  "candidate_score": 0.0
}
```

## 5. Candidate Search Plan 생성

각 질문에 대해 여러 candidate search plan을 만든다.

생성하는 후보는 다음과 같다.

| Candidate | 의미 |
| --- | --- |
| `raw` | 원래 Korean/Mixed query를 그대로 사용 |
| `machine_translate` | NLLB 또는 Google Translate로 단순 번역 |
| `entity_expand` | 번역 query에 entity, answer keyword, title 등을 추가 |
| `hyde` | 질문에 대한 hypothetical evidence document를 생성 |
| `query2doc` | 질문을 pseudo document 형식으로 확장 |
| `multilingual_plan` | Korean query와 English query를 함께 사용하는 search plan |
| `gold_target` | `target_query` 기반 upper-bound reference |

이 단계의 결과물은 다음이다.

```text
/opt/dlami/nvme/citation_full/citation_candidates.jsonl
```

이 파일은 질문별로 가능한 여러 검색 전략을 저장한다.

## 6. Hybrid Retrieval 및 Reranking

단일 retriever만 사용하는 것은 충분하지 않다. citation retrieval에서는 exact keyword matching과 semantic matching이 모두 중요하다.

따라서 retrieval은 다음 조합으로 수행한다.

```text
BM25
+ BGE-M3 dense retrieval
+ Reciprocal Rank Fusion
+ BGE reranker
```

각 구성 요소의 역할은 다음과 같다.

| 구성 요소 | 역할 |
| --- | --- |
| BM25 | entity, 날짜, 고유명사, exact keyword matching에 강함 |
| BGE-M3 | 한국어 query로 영어 문서를 찾는 multilingual dense retrieval에 강함 |
| Reciprocal Rank Fusion | BM25와 dense retrieval 결과를 안정적으로 결합 |
| BGE reranker | top candidate 중 질문과 가장 관련 있는 citation을 상위로 재정렬 |

이 단계의 결과물은 다음이다.

```text
/opt/dlami/nvme/citation_full/citation_retrieved.jsonl
```

각 search plan마다 top citation 후보들이 붙는다.

## 7. Human-in-the-loop Citation Labeling

이 프로젝트의 핵심은 사람이 citation을 직접 평가하고, 그 feedback을 모델 학습에 반영하는 것이다.

retrieved citation을 사람이 보고 다음 label 중 하나를 선택한다.

| Label | 의미 |
| --- | --- |
| `supported` | citation이 답변 claim을 충분히 support함 |
| `partial` | 일부만 support함 |
| `unsupported` | 관련 없거나 근거가 부족함 |
| `contradicted` | citation이 claim과 모순됨 |

라벨링 과정에서 사람은 다음 정보를 본다.

```text
question
search plan method
generated queries
retrieved citation title
retrieved citation text
rank
language
```

라벨 결과는 다음 파일에 저장된다.

```text
/opt/dlami/nvme/citation_full/citation_labels.jsonl
```

이 파일은 이후 scoring과 training data generation에 사용된다. 즉, 사람이 좋은 citation과 나쁜 citation의 패턴을 모델에게 알려주는 feedback 역할을 한다.

## 8. Citation Scoring

각 candidate search plan은 retrieval 결과와 human label을 기반으로 점수를 받는다.

점수에 반영되는 요소는 다음과 같다.

- positive document가 top-k 안에 있는가
- citation이 answer string을 포함하는가
- 사람이 `supported` 또는 `partial`로 라벨링했는가
- citation precision이 높은가
- citation recall이 높은가
- 유용한 영어 citation을 찾았는가
- reranker score가 높은가

최종 candidate score는 다음 성격의 weighted score이다.

```text
candidate_score
= retrieval quality
+ citation support quality
+ answer coverage
+ useful English citation ratio
+ answer faithfulness proxy
```

이 단계의 결과물은 다음이다.

```text
/opt/dlami/nvme/citation_full/citation_scored.jsonl
/opt/dlami/nvme/citation_full/citation_summary.csv
```

이 파일을 통해 각 방법이 citation을 얼마나 잘 찾았는지 비교할 수 있다.

## 9. SFT 및 Preference Dataset 생성

각 질문에 대해 candidate score가 가장 높은 search plan을 `chosen`으로 선택한다. 낮은 점수의 search plan은 `rejected`가 된다.

SFT dataset은 다음 형태를 갖는다.

```text
input:
Korean/Mixed question

target:
chosen search plan JSON
```

Preference dataset은 다음 형태를 갖는다.

```text
question
chosen_search_plan
rejected_search_plan
chosen_score
rejected_score
```

결과 파일은 다음이다.

```text
/opt/dlami/nvme/citation_full/citation_sft_train.jsonl
/opt/dlami/nvme/citation_full/citation_preferences.jsonl
```

이제 모델은 단순히 `target_query`를 맞히는 것이 아니라, 실제로 좋은 citation을 가져온 search plan을 학습한다.

## 10. Search Planner Fine-tuning

search planner는 Qwen 계열 instruction model을 LoRA 또는 QLoRA로 fine-tuning한다.

기본 모델은 다음이다.

```text
Qwen/Qwen2.5-7B-Instruct
```

학습 목표는 다음이다.

```text
Korean/Mixed question
-> citation-seeking search plan JSON
```

예상 출력은 다음과 같다.

```json
{
  "queries": [
    "Los Angeles Lakers last playoff appearance 2013 NBA season",
    "Lakers playoff drought 2014 2019",
    "레이커스 마지막 플레이오프 진출"
  ],
  "entities": ["Los Angeles Lakers", "NBA playoffs"],
  "aliases": ["Lakers", "LA Lakers"],
  "answer_type": "date",
  "preferred_source_languages": ["en", "ko"],
  "source_priority": ["official", "encyclopedic", "news"]
}
```

학습은 AWS GPU에서 수행한다.

로컬 Mac은 retrieval, reranking, human labeling, scoring, dataset generation에 사용하고, AWS는 Qwen QLoRA 학습에 사용한다.

학습 결과는 다음 위치에 저장한다.

```text
/opt/dlami/nvme/citation_planner/qwen25_7b_lora
```

## 11. 평가 지표

본 프로젝트의 평가는 query translation quality가 아니라 citation quality 중심이다.

### 11.1 Retrieval Metrics

| Metric | 의미 |
| --- | --- |
| `Recall@5` | positive/supporting document가 top-5 안에 있는가 |
| `Recall@10` | positive/supporting document가 top-10 안에 있는가 |
| `Recall@20` | positive/supporting document가 top-20 안에 있는가 |
| `MRR` | 첫 번째 relevant citation이 얼마나 높은 rank에 있는가 |
| `nDCG@10` | relevant citation이 상위 rank에 잘 배치되었는가 |

### 11.2 Citation Metrics

| Metric | 의미 |
| --- | --- |
| `citation_precision` | 사용한 citation 중 실제로 claim을 support하는 비율 |
| `citation_recall` | 필요한 claim 중 citation으로 support된 비율 |
| `citation_f1` | citation precision과 recall의 조화 평균 |
| `unsupported_claim_rate` | citation으로 support되지 않는 claim 비율 |
| `contradicted_claim_rate` | citation과 모순되는 claim 비율 |

### 11.3 Cross-lingual Citation Metrics

| Metric | 의미 |
| --- | --- |
| `english_citation_ratio` | retrieved/used citation 중 영어 source 비율 |
| `korean_citation_ratio` | retrieved/used citation 중 한국어 source 비율 |
| `useful_english_citation_ratio` | 실제 support하는 영어 citation 비율 |
| `cross_lingual_success_rate` | 한국어/혼합 질문에서 non-Korean useful citation을 찾은 비율 |

### 11.4 Answer Metrics

| Metric | 의미 |
| --- | --- |
| `answer_faithfulness` | 답변이 citation에 근거하는가 |
| `answer_relevance` | 답변이 질문에 직접 답하는가 |
| `answer_contains_gold_answer` | 답변 또는 citation이 gold answer를 포함하는가 |

가장 중요한 지표는 다음 세 가지다.

```text
citation_f1
useful_english_citation_ratio
answer_faithfulness
```

## 12. 비교 대상

최종 보고서에서는 다음 방법을 비교한다.

| Method | 설명 |
| --- | --- |
| `raw` | Korean/Mixed query 그대로 retrieval |
| `machine_translate` | 단순 번역 query로 retrieval |
| `entity_expand` | entity/answer keyword 확장 query |
| `hyde` | hypothetical evidence 기반 query |
| `query2doc` | pseudo document 기반 query |
| `multilingual_plan` | Korean/English query를 함께 사용하는 plan |
| `fine_tuned_citation_planner` | human feedback 기반으로 학습된 planner |
| `gold_target` | target_query 기반 upper-bound reference |

여기서 `gold_target`은 실제 baseline이 아니라 upper-bound이다. 넘겨야 하는 대상은 `gold_target`이 아니라 `machine_translate`이다.

## 13. 실행 분담

로컬 Mac에서 실행할 단계는 다음이다.

```text
candidate generation
hybrid retrieval
reranking
human labeling
citation scoring
SFT/preference dataset generation
comparison report
```

AWS GPU에서 실행할 단계는 다음이다.

```text
Qwen QLoRA search planner fine-tuning
```

이렇게 나누는 이유는 다음과 같다.

```text
retrieval/reranking:
CPU, RAM, disk I/O를 많이 사용

Qwen fine-tuning:
GPU memory와 CUDA가 필요
```

## 14. 주요 실행 명령

서버에서 최신 코드를 받는다.

```bash
cd ~/COSE461-A3
git pull origin main
/opt/jupyter/venv/bin/python3 -m pip install -e ".[peft]"
```

NVMe cache를 설정한다.

```bash
export HF_HOME=/opt/dlami/nvme/hf_cache
export TRANSFORMERS_CACHE=/opt/dlami/nvme/hf_cache
export HF_HUB_CACHE=/opt/dlami/nvme/hf_cache/hub
export TORCH_HOME=/opt/dlami/nvme/torch_cache
mkdir -p /opt/dlami/nvme/citation_full /opt/dlami/nvme/citation_index
```

full citation pipeline을 실행한다.

```bash
cd ~/COSE461-A3
PYTHONPATH=src /opt/jupyter/venv/bin/python3 scripts/run_citation_pipeline.py \
  --config configs/citation_full.yaml \
  --work-dir /opt/dlami/nvme/citation_full
```

human-in-the-loop labeling을 실행한다.

```bash
PYTHONPATH=src /opt/jupyter/venv/bin/python3 scripts/label_citations.py \
  --input /opt/dlami/nvme/citation_full/citation_retrieved.jsonl \
  --labels-output /opt/dlami/nvme/citation_full/citation_labels.jsonl \
  --max-items 500
```

라벨을 반영해 다시 scoring한다.

```bash
PYTHONPATH=src /opt/jupyter/venv/bin/python3 scripts/score_citation_candidates.py \
  --input /opt/dlami/nvme/citation_full/citation_retrieved.jsonl \
  --output /opt/dlami/nvme/citation_full/citation_scored.jsonl \
  --summary-output /opt/dlami/nvme/citation_full/citation_summary.csv \
  --labels /opt/dlami/nvme/citation_full/citation_labels.jsonl
```

SFT/preference dataset을 생성한다.

```bash
PYTHONPATH=src /opt/jupyter/venv/bin/python3 scripts/build_citation_sft_dataset.py \
  --input /opt/dlami/nvme/citation_full/citation_scored.jsonl \
  --sft-output /opt/dlami/nvme/citation_full/citation_sft_train.jsonl \
  --preference-output /opt/dlami/nvme/citation_full/citation_preferences.jsonl
```

AWS GPU에서 search planner를 학습한다.

```bash
PYTHONPATH=src /opt/jupyter/venv/bin/python3 scripts/train_search_planner.py \
  --train-file /opt/dlami/nvme/citation_full/citation_sft_train.jsonl \
  --output-dir /opt/dlami/nvme/citation_planner/qwen25_7b_lora \
  --base-model Qwen/Qwen2.5-7B-Instruct \
  --qlora \
  --bf16 \
  --epochs 2 \
  --batch-size 1 \
  --gradient-accumulation-steps 8
```

최종 비교 report를 생성한다.

```bash
PYTHONPATH=src /opt/jupyter/venv/bin/python3 scripts/compare_citation_runs.py \
  --inputs /opt/dlami/nvme/citation_full/citation_scored.jsonl \
  --output-dir output/citation_analysis
```

## 15. 최종 결과물

주요 결과 파일은 다음이다.

```text
/opt/dlami/nvme/citation_full/citation_candidates.jsonl
/opt/dlami/nvme/citation_full/citation_retrieved.jsonl
/opt/dlami/nvme/citation_full/citation_labels.jsonl
/opt/dlami/nvme/citation_full/citation_scored.jsonl
/opt/dlami/nvme/citation_full/citation_summary.csv
/opt/dlami/nvme/citation_full/citation_sft_train.jsonl
/opt/dlami/nvme/citation_full/citation_preferences.jsonl
/opt/dl/nvme/citation_planner/qwen25_7b_lora
output/citation_analysis/comparison.csv
output/citation_analysis/report.md
```

최종적으로 가장 먼저 확인해야 할 파일은 다음 두 개다.

```text
output/citation_analysis/comparison.csv
output/citation_analysis/report.md
```

여기에서 `machine_translate`와 `fine_tuned_citation_planner`의 citation metrics를 비교한다.

## 16. 기대 결과와 해석

기대하는 결과는 다음이다.

```text
raw < machine_translate <= fine_tuned_citation_planner
```

단, 모든 지표에서 항상 planner가 이길 필요는 없다. 중요한 것은 citation quality 관련 핵심 지표에서 단순 번역 baseline보다 나아지는 것이다.

핵심 지표는 다음이다.

```text
citation_f1
useful_english_citation_ratio
answer_faithfulness
```

최종 보고서의 주장은 다음처럼 정리할 수 있다.

```text
한국어/혼합 언어 질문에서 단순 번역은 cross-lingual retrieval을 개선하지만,
citation 품질을 직접 최적화하지는 않는다.

우리는 human-in-the-loop citation feedback을 사용해,
좋은 citation을 찾는 search plan을 학습한다.

그 결과 모델은 단순 번역 query보다 더 citation-worthy한 evidence를 찾는 방향으로 개선된다.
```

한 문장으로 요약하면 다음과 같다.

```text
We optimize query rewriting not for translation quality, but for citation quality.
```


"""Build citation-seeking search-plan candidates for full retrieval runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _ensure_src_on_path() -> None:
    project_root = Path(__file__).resolve().parent.parent
    src_path = project_root / "src"
    if src_path.exists() and str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


_ensure_src_on_path()

from crosslingual_rewrite.citation import (  # noqa: E402
    CitationCandidateRecord,
    SearchPlan,
    answers_from_metadata,
    compact_query_tokens,
    make_search_plan,
)
from crosslingual_rewrite.config import load_config, validate_config  # noqa: E402
from crosslingual_rewrite.data import RewriteExample, load_dataset  # noqa: E402
from crosslingual_rewrite.modeling import GenerationRequest, HFRewriteModel  # noqa: E402
from crosslingual_rewrite.search_planner import LoRASearchPlanner, PlannerGenerationConfig  # noqa: E402


def _candidate_id(example: RewriteExample, method: str) -> str:
    question_id = example.example_id or example.question_ko[:40]
    return f"{question_id}:{method}"


def _load_completed_candidate_ids(path: Path) -> tuple[set[str], int, int]:
    completed: set[str] = set()
    valid_rows = 0
    invalid_rows = 0
    if not path.exists():
        return completed, valid_rows, invalid_rows
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError:
                invalid_rows += 1
                continue
            candidate_id = str(row.get("candidate_id") or "").strip()
            if candidate_id:
                completed.add(candidate_id)
                valid_rows += 1
    return completed, valid_rows, invalid_rows


def _write_records(
    out,
    records: list[CitationCandidateRecord],
    *,
    completed_ids: set[str],
) -> tuple[int, int]:
    written = 0
    skipped = 0
    for record in records:
        if record.candidate_id in completed_ids:
            skipped += 1
            continue
        out.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
        out.flush()
        completed_ids.add(record.candidate_id)
        written += 1
    return written, skipped


def _machine_translate(
    examples: list[RewriteExample],
    *,
    base_model: str,
    src_lang: str | None,
    tgt_lang: str | None,
    batch_size: int,
    max_output_length: int,
    num_beams: int,
    progress_every: int,
) -> dict[str, str]:
    print(
        f"[candidates] loading translation model={base_model} examples={len(examples)} "
        f"batch_size={batch_size} num_beams={num_beams}",
        flush=True,
    )
    model = HFRewriteModel(
        base_model=base_model,
        max_input_length=256,
        max_output_length=max_output_length,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        lora_enabled=False,
    )
    request = GenerationRequest(max_output_length=max_output_length, num_beams=num_beams)
    outputs: dict[str, str] = {}
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        generated = model.generate(batch, request=request)
        for example, query in zip(batch, generated):
            outputs[example.example_id or str(start)] = query
        done = min(start + len(batch), len(examples))
        if progress_every > 0 and (done == len(examples) or done == len(batch) or done % progress_every == 0):
            pct = 100.0 * done / max(1, len(examples))
            print(f"[candidates] translated {done}/{len(examples)} ({pct:.1f}%)", flush=True)
    return outputs


def _records_for_example(
    example: RewriteExample,
    *,
    machine_translation: str | None,
    planner_plan: SearchPlan | None,
    methods: set[str],
) -> list[CitationCandidateRecord]:
    answers = answers_from_metadata(example.metadata or {})
    base_query = machine_translation or example.target_query or example.question_ko
    compact_answers = compact_query_tokens(" ".join(answers), max_tokens=12)
    candidates: list[tuple[str, list[str], dict]] = []

    if "raw" in methods:
        candidates.append(("raw", [example.question_ko], {}))
    if "machine_translate" in methods and machine_translation:
        candidates.append(("machine_translate", [machine_translation], {"translation_model": "base_model"}))
    if "gold_target" in methods and example.target_query:
        candidates.append(("gold_target", [example.target_query], {"upper_bound": True}))
    if "entity_expand" in methods:
        expanded = compact_query_tokens(base_query, compact_answers, max_tokens=40)
        queries = [query for query in [expanded, base_query, example.question_ko] if query]
        candidates.append(("entity_expand", queries, {}))
    if "hyde" in methods:
        hyde = " ".join(
            part
            for part in [
                "Evidence document about",
                base_query,
                compact_answers,
                "background facts chronology definition official source",
            ]
            if part
        )
        candidates.append(("hyde", [hyde, base_query], {}))
    if "query2doc" in methods:
        query2doc = " ".join(
            part
            for part in [
                base_query,
                "answer evidence citation source",
                compact_answers,
            ]
            if part
        )
        candidates.append(("query2doc", [query2doc, base_query], {}))
    if "multilingual_plan" in methods:
        queries = [query for query in [base_query, example.question_ko, compact_answers] if query]
        candidates.append(("multilingual_plan", queries, {"preferred_source_languages": ["en", "ko"]}))

    rows: list[CitationCandidateRecord] = []
    question_id = example.example_id or example.question_ko[:40]
    if "citation_planner" in methods and planner_plan is not None:
        rows.append(
            CitationCandidateRecord(
                question_id=question_id,
                question=example.question_ko,
                query_type=example.query_type,
                candidate_id=f"{question_id}:citation_planner",
                search_plan=planner_plan,
                positive_doc_id=example.positive_doc_id,
                negative_doc_id=example.negative_doc_id,
                target_query=example.target_query,
                answers=answers,
                metadata={
                    "dataset_name": example.dataset_name,
                    "split_name": example.split_name,
                    "source_language": example.source_language,
                },
            )
        )
    for method, queries, metadata in candidates:
        plan = make_search_plan(
            method=method,
            question=example.question_ko,
            queries=queries,
            target_query=example.target_query,
            answers=answers,
            metadata=metadata,
        )
        rows.append(
            CitationCandidateRecord(
                question_id=question_id,
                question=example.question_ko,
                query_type=example.query_type,
                candidate_id=f"{question_id}:{method}",
                search_plan=plan,
                positive_doc_id=example.positive_doc_id,
                negative_doc_id=example.negative_doc_id,
                target_query=example.target_query,
                answers=answers,
                metadata={
                    "dataset_name": example.dataset_name,
                    "split_name": example.split_name,
                    "source_language": example.source_language,
                },
            )
        )
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build citation-aware candidate search plans.")
    parser.add_argument("--config", required=True, help="Experiment config with dataset/model paths.")
    parser.add_argument("--output", required=True, help="Output candidate JSONL.")
    parser.add_argument(
        "--methods",
        default="raw,machine_translate,entity_expand,hyde,query2doc,multilingual_plan,gold_target",
        help="Comma-separated candidate methods.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional full-run limit.")
    parser.add_argument("--translation-batch-size", type=int, default=4)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--max-output-length", type=int, default=96)
    parser.add_argument("--planner-base-model", default=None, help="Base causal LM for citation_planner.")
    parser.add_argument("--planner-adapter", default=None, help="PEFT adapter path for citation_planner.")
    parser.add_argument("--planner-batch-size", type=int, default=4)
    parser.add_argument("--planner-max-new-tokens", type=int, default=192)
    parser.add_argument("--planner-temperature", type=float, default=0.0)
    parser.add_argument("--planner-top-p", type=float, default=1.0)
    parser.add_argument("--planner-no-4bit", action="store_true")
    parser.add_argument("--planner-no-bf16", action="store_true")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Append to an existing output and skip already written candidate_id rows.",
    )
    parser.add_argument(
        "--reverse",
        action="store_true",
        help="Process dataset examples from the end to the beginning.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print progress every N questions. Use 0 to disable progress output.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    validate_config(cfg)
    examples = load_dataset(cfg.data.dataset_path)
    if args.reverse:
        examples = list(reversed(examples))
    if args.limit is not None and args.limit >= 0:
        examples = examples[: args.limit]
    methods = {method.strip() for method in args.methods.split(",") if method.strip()}
    print(
        f"[candidates] loaded examples={len(examples)} methods={','.join(sorted(methods))}",
        flush=True,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed_ids: set[str] = set()
    completed_rows = 0
    invalid_rows = 0
    if args.resume:
        completed_ids, completed_rows, invalid_rows = _load_completed_candidate_ids(output_path)
        if completed_rows or invalid_rows:
            print(
                f"[candidates] resume completed={completed_rows} invalid_rows={invalid_rows}",
                flush=True,
            )

    translations: dict[str, str] = {}
    if "machine_translate" in methods:
        translations = _machine_translate(
            examples,
            base_model=cfg.model.base_model,
            src_lang=cfg.model.src_lang,
            tgt_lang=cfg.model.tgt_lang,
            batch_size=args.translation_batch_size,
            max_output_length=args.max_output_length,
            num_beams=args.num_beams,
            progress_every=args.progress_every,
        )

    written = 0
    skipped = 0
    if "citation_planner" in methods:
        if not args.planner_base_model or not args.planner_adapter:
            raise SystemExit(
                "citation_planner requires --planner-base-model and --planner-adapter."
            )
        print(
            f"[candidates] loading citation planner base={args.planner_base_model} "
            f"adapter={args.planner_adapter} examples={len(examples)}",
            flush=True,
        )
        planner = LoRASearchPlanner(
            base_model=args.planner_base_model,
            adapter_path=args.planner_adapter,
            config=PlannerGenerationConfig(
                max_new_tokens=args.planner_max_new_tokens,
                temperature=args.planner_temperature,
                top_p=args.planner_top_p,
                load_in_4bit=not args.planner_no_4bit,
                bf16=not args.planner_no_bf16,
            ),
        )
        mode = "a" if args.resume else "w"
        with output_path.open(mode, encoding="utf-8") as out:
            for start in range(0, len(examples), args.planner_batch_size):
                batch = examples[start : start + args.planner_batch_size]
                pending = [
                    example
                    for example in batch
                    if _candidate_id(example, "citation_planner") not in completed_ids
                ]
                if pending:
                    generated = planner.generate(
                        [example.question_ko for example in pending],
                        batch_size=args.planner_batch_size,
                    )
                    for example, plan in zip(pending, generated):
                        records = _records_for_example(
                            example,
                            machine_translation=translations.get(example.example_id or ""),
                            planner_plan=plan,
                            methods={"citation_planner"},
                        )
                        new_written, _ = _write_records(out, records, completed_ids=completed_ids)
                        written += new_written
                done = min(start + len(batch), len(examples))
                if args.progress_every > 0 and (
                    done == len(examples) or done == len(batch) or done % args.progress_every == 0
                ):
                    pct = 100.0 * done / max(1, len(examples))
                    print(
                        f"[candidates] planned {done}/{len(examples)} ({pct:.1f}%) "
                        f"new_written={written} total_saved={len(completed_ids)}",
                        flush=True,
                    )

    remaining_methods = set(methods)
    remaining_methods.discard("citation_planner")
    mode = "a" if args.resume else "w"
    if not remaining_methods:
        print(
            json.dumps(
                {
                    "output": str(output_path),
                    "questions": len(examples),
                    "candidates": len(completed_ids),
                    "new_candidates": written,
                    "skipped_existing": skipped,
                    "resume": bool(args.resume),
                },
                indent=2,
            )
        )
        return 0
    mode = "a" if args.resume or "citation_planner" in methods else "w"
    with output_path.open(mode, encoding="utf-8") as out:
        for index, example in enumerate(examples, start=1):
            records = _records_for_example(
                example,
                machine_translation=translations.get(example.example_id or ""),
                planner_plan=None,
                methods=remaining_methods,
            )
            new_written, new_skipped = _write_records(out, records, completed_ids=completed_ids)
            written += new_written
            skipped += new_skipped
            if args.progress_every > 0 and (
                index == 1 or index == len(examples) or index % args.progress_every == 0
            ):
                pct = 100.0 * index / max(1, len(examples))
                print(
                    f"[candidates] wrote questions={index}/{len(examples)} ({pct:.1f}%) "
                    f"candidates={written}",
                    flush=True,
                )
    print(
        json.dumps(
            {
                "output": str(output_path),
                "questions": len(examples),
                "candidates": len(completed_ids),
                "new_candidates": written,
                "skipped_existing": skipped,
                "resume": bool(args.resume),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

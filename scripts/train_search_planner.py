"""Fine-tune a causal LM search planner with LoRA/QLoRA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _format_example(tokenizer, row: dict[str, Any]) -> tuple[str, str]:  # type: ignore[no-untyped-def]
    messages = row["messages"]
    prompt_messages = messages[:-1]
    assistant = messages[-1]["content"]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        prompt = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
        full = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        return prompt, full
    prompt = "\n\n".join(f"{m['role'].upper()}:\n{m['content']}" for m in prompt_messages) + "\n\nASSISTANT:\n"
    return prompt, prompt + assistant


class PlannerDataset:
    def __init__(self, rows: list[dict[str, Any]], tokenizer, *, max_length: int) -> None:  # type: ignore[no-untyped-def]
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        prompt, full = _format_example(self.tokenizer, self.rows[index])
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        full_ids = self.tokenizer(full, add_special_tokens=False, truncation=True, max_length=self.max_length)["input_ids"]
        labels = list(full_ids)
        prompt_len = min(len(prompt_ids), len(labels))
        labels[:prompt_len] = [-100] * prompt_len
        return {"input_ids": full_ids, "attention_mask": [1] * len(full_ids), "labels": labels}


class DataCollator:
    def __init__(self, tokenizer) -> None:  # type: ignore[no-untyped-def]
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, Any]:
        import torch

        pad_id = self.tokenizer.pad_token_id
        max_len = max(len(item["input_ids"]) for item in features)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for item in features:
            pad = max_len - len(item["input_ids"])
            batch["input_ids"].append(item["input_ids"] + [pad_id] * pad)
            batch["attention_mask"].append(item["attention_mask"] + [0] * pad)
            batch["labels"].append(item["labels"] + [-100] * pad)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a citation-aware search planner.")
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    parser.add_argument("--qlora", action="store_true", help="Load model in 4-bit with bitsandbytes.")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=500)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    import torch
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments

    rows = _load_rows(Path(args.train_file))
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {"trust_remote_code": True}
    if args.qlora:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        )
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16 if args.bf16 else torch.float16 if args.fp16 else torch.float32
        model_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs)
    if args.qlora:
        model = prepare_model_for_kbit_training(model)
    target_modules = [module.strip() for module in args.target_modules.split(",") if module.strip()]
    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
        ),
    )
    dataset = PlannerDataset(rows, tokenizer, max_length=args.max_length)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        bf16=args.bf16,
        fp16=args.fp16,
        report_to=[],
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollator(tokenizer),
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    metadata = {
        "base_model": args.base_model,
        "train_file": args.train_file,
        "objective": "citation_search_plan_sft",
        "rows": len(rows),
        "qlora": bool(args.qlora),
    }
    Path(args.output_dir, "planner_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": args.output_dir, "rows": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Inference utilities for fine-tuned citation search planners."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Iterable

from .citation import SearchPlan, make_search_plan


SYSTEM_PROMPT = (
    "You are a citation-seeking search planner. Given a Korean or mixed-language "
    "question, output compact JSON with queries, entities, aliases, answer_type, "
    "preferred_source_languages, and source_priority. Optimize the plan for finding "
    "citation-worthy evidence, not for producing a fluent translation."
)

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass(frozen=True)
class PlannerGenerationConfig:
    max_new_tokens: int = 192
    temperature: float = 0.0
    top_p: float = 1.0
    load_in_4bit: bool = True
    bf16: bool = True


def planner_prompt(question: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]


def parse_search_plan_output(text: str, *, question: str, raw_output: bool = False) -> SearchPlan:
    """Parse model text into a SearchPlan, with a conservative fallback."""

    match = _JSON_BLOCK_RE.search(text)
    if not match:
        plan = make_search_plan(method="citation_planner", question=question, queries=[question])
        plan.metadata["parse_error"] = "missing_json"
        if raw_output:
            plan.metadata["raw_output"] = text
        return plan
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        plan = make_search_plan(method="citation_planner", question=question, queries=[question])
        plan.metadata["parse_error"] = f"invalid_json:{exc.msg}"
        if raw_output:
            plan.metadata["raw_output"] = text
        return plan
    if not isinstance(payload, dict):
        plan = make_search_plan(method="citation_planner", question=question, queries=[question])
        plan.metadata["parse_error"] = "json_not_object"
        if raw_output:
            plan.metadata["raw_output"] = text
        return plan

    payload = dict(payload)
    payload["method"] = "citation_planner"
    plan = SearchPlan.from_dict(payload)
    if not plan.queries:
        plan = make_search_plan(method="citation_planner", question=question, queries=[question])
        plan.metadata["parse_error"] = "empty_queries"
    if raw_output:
        plan.metadata["raw_output"] = text
    return plan


class LoRASearchPlanner:
    """Load a base causal LM plus PEFT adapter and generate SearchPlan objects."""

    def __init__(
        self,
        *,
        base_model: str,
        adapter_path: str,
        config: PlannerGenerationConfig | None = None,
    ) -> None:
        self.config = config or PlannerGenerationConfig()

        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_kwargs: dict[str, Any] = {"trust_remote_code": True, "device_map": "auto"}
        if self.config.load_in_4bit:
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16 if self.config.bf16 else torch.float16,
            )
        else:
            model_kwargs["torch_dtype"] = torch.bfloat16 if self.config.bf16 else torch.float16

        self.tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        base = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)
        self.model = PeftModel.from_pretrained(base, adapter_path)
        self.model.eval()

    def generate(self, questions: Iterable[str], *, batch_size: int = 4) -> list[SearchPlan]:
        import torch

        questions_list = list(questions)
        plans: list[SearchPlan] = []
        do_sample = self.config.temperature > 0
        for start in range(0, len(questions_list), batch_size):
            batch = questions_list[start : start + batch_size]
            prompts = [
                self.tokenizer.apply_chat_template(
                    planner_prompt(question),
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for question in batch
            ]
            encoded = self.tokenizer(prompts, return_tensors="pt", padding=True)
            encoded = {key: value.to(self.model.device) for key, value in encoded.items()}
            input_width = int(encoded["input_ids"].shape[1])
            generation_kwargs: dict[str, Any] = {
                "max_new_tokens": self.config.max_new_tokens,
                "do_sample": do_sample,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }
            if do_sample:
                generation_kwargs["temperature"] = self.config.temperature
                generation_kwargs["top_p"] = self.config.top_p
            with torch.inference_mode():
                generated = self.model.generate(**encoded, **generation_kwargs)
            for question, output_ids in zip(batch, generated):
                new_ids = output_ids[input_width:]
                text = self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
                plans.append(parse_search_plan_output(text, question=question))
        return plans

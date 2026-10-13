#!/usr/bin/env python3
"""Common post-hoc diagnostics for a merged Countdown checkpoint.

The workflow is deliberately split into two GPU phases so vLLM and
Transformers never hold the model at the same time::

  python replication/eval_checkpoint_diagnostics.py generate \
      --model /path/to/merged_hf --output responses.jsonl
  python replication/eval_checkpoint_diagnostics.py entropy \
      --model /path/to/merged_hf --input responses.jsonl \
      --output responses_with_entropy.jsonl
  python replication/eval_checkpoint_diagnostics.py summarize \
      --input responses_with_entropy.jsonl --output summary.json

Generation follows ``PROSPECTIVE_DIAGNOSTICS.md``: the vendored 1,024-item
validation set, seed 0, temperature 1.0, top-p 0.7, disabled top-k, and one
sample per prompt.  Entropy is the full-vocabulary categorical Shannon entropy
of each sampled response token under the same checkpoint, not an approximation
from top-k log probabilities.
"""

from __future__ import annotations

import argparse
import ast
import inspect
import json
import math
import re
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable


DEFAULT_DATA = Path(__file__).resolve().parent / "data" / "countdown_test.parquet"
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
FORMAT_RE = re.compile(r"<think>(.*?)</think>.*?<answer>(.*?)</answer>", re.DOTALL)
ALLOWED_EQUATION_RE = re.compile(r"^[\d+\-*/().\s]+$")


def _jsonable(value: Any) -> Any:
    """Convert numpy/pandas containers to plain JSON-compatible values."""
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def _messages(value: Any) -> list[dict[str, str]]:
    value = _jsonable(value)
    if not isinstance(value, list):
        raise TypeError(f"prompt must be a list of messages, got {type(value).__name__}")
    return [{"role": str(m["role"]), "content": str(m["content"])} for m in value]


def extract_equation(response: str) -> str | None:
    matches = list(ANSWER_RE.finditer(response))
    return matches[-1].group(1).strip() if matches else None


def format_valid(response: str) -> bool:
    # Training prefills the opening <think>; preserve the historical convention.
    return FORMAT_RE.search("<think>" + response) is not None


def _safe_arithmetic(expression: str) -> float | None:
    """Evaluate the same arithmetic language accepted by the frozen scorer."""
    if ALLOWED_EQUATION_RE.fullmatch(expression) is None:
        return None
    try:
        tree = ast.parse(expression, mode="eval")
    except (SyntaxError, ValueError):
        return None

    allowed_binops = (ast.Add, ast.Sub, ast.Mult, ast.Div)
    allowed_unary = (ast.UAdd, ast.USub)
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and not isinstance(node.op, allowed_binops):
            return None
        if isinstance(node, ast.UnaryOp) and not isinstance(node.op, allowed_unary):
            return None
        if isinstance(node, (ast.Expression, ast.Load, ast.Constant, ast.BinOp,
                             ast.UnaryOp, *allowed_binops, *allowed_unary)):
            continue
        return None
    try:
        value = eval(compile(tree, "<countdown-equation>", "eval"), {"__builtins__": {}}, {})
        return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else None
    except (ArithmeticError, TypeError, ValueError):
        return None


def score_response(response: str, numbers: Iterable[int], target: float) -> dict[str, Any]:
    equation = extract_equation(response)
    result = {
        "answer_extracted": equation is not None,
        "format_valid": format_valid(response),
        "correct": False,
        "equation": equation,
    }
    if equation is None:
        return result
    used = sorted(int(n) for n in re.findall(r"\d+", equation))
    available = sorted(int(n) for n in numbers)
    if used != available:
        return result
    value = _safe_arithmetic(equation)
    result["correct"] = value is not None and abs(value - float(target)) < 1e-5
    return result


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]], mode: str = "w") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(mode, encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            handle.flush()


def generate(args: argparse.Namespace) -> None:
    import pandas as pd
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite {args.output}; pass --overwrite")

    frame = pd.read_parquet(args.data)
    if args.limit is not None:
        frame = frame.iloc[: args.limit]
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    prompts = [
        tokenizer.apply_chat_template(_messages(value), tokenize=False, add_generation_prompt=True)
        for value in frame["prompt"]
    ]
    prompt_lengths = [len(tokenizer.encode(prompt, add_special_tokens=False)) for prompt in prompts]
    max_model_len = args.max_model_len or max(prompt_lengths) + args.max_tokens
    llm = LLM(
        model=str(args.model),
        dtype="bfloat16",
        max_model_len=max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=0,
        trust_remote_code=False,
    )
    sampling = SamplingParams(
        temperature=1.0,
        top_p=0.7,
        top_k=-1,
        max_tokens=args.max_tokens,
        n=1,
        seed=0,
    )
    outputs = llm.generate(prompts, sampling)
    rows = []
    for position, (_, item), output in zip(range(len(frame)), frame.iterrows(), outputs):
        sample = output.outputs[0]
        response_ids = list(sample.token_ids)
        metrics = score_response(sample.text, item["nums"], item["target"])
        rows.append(
            {
                "index": int(_jsonable(item.get("extra_info", {})).get("index", position)),
                "target": _jsonable(item["target"]),
                "numbers": _jsonable(item["nums"]),
                "prompt_token_ids": list(output.prompt_token_ids),
                "response_token_ids": response_ids,
                "response": sample.text,
                "finish_reason": sample.finish_reason,
                "response_tokens": len(response_ids),
                "max_tokens": args.max_tokens,
                **metrics,
            }
        )
    _write_jsonl(args.output, rows)
    print(f"wrote {len(rows)} generations to {args.output}")


def entropy(args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoModelForCausalLM

    rows = _read_jsonl(args.input)
    if args.output.exists() and args.overwrite:
        args.output.unlink()
    completed: set[int] = set()
    if args.output.exists():
        completed = {int(row["index"]) for row in _read_jsonl(args.output)}

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map={"": args.device},
        trust_remote_code=False,
    ).eval()
    supports_logits_to_keep = "logits_to_keep" in inspect.signature(model.forward).parameters
    done = 0
    for row in rows:
        if int(row["index"]) in completed:
            continue
        prompt_ids = [int(v) for v in row["prompt_token_ids"]]
        response_ids = [int(v) for v in row["response_token_ids"]]
        if response_ids:
            # The last R logits over prompt + response[:-1] predict all R response tokens.
            context = torch.tensor([prompt_ids + response_ids[:-1]], device=args.device)
            kwargs = {"input_ids": context, "use_cache": False}
            if supports_logits_to_keep:
                kwargs["logits_to_keep"] = len(response_ids)
            with torch.inference_mode():
                logits = model(**kwargs).logits[:, -len(response_ids):, :].float()[0]
                log_z = torch.logsumexp(logits, dim=-1)
                probabilities = torch.softmax(logits, dim=-1)
                token_entropy = log_z - (probabilities * logits).sum(dim=-1)
                targets = torch.tensor(response_ids, device=logits.device)
                selected_logp = logits.gather(1, targets[:, None]).squeeze(1) - log_z
            row["token_mean_entropy_nats"] = float(token_entropy.mean().cpu())
            row["selected_token_mean_logprob"] = float(selected_logp.mean().cpu())
            row["entropy_token_count"] = len(response_ids)
            del context, logits, log_z, probabilities, token_entropy, selected_logp, targets
        else:
            row["token_mean_entropy_nats"] = None
            row["selected_token_mean_logprob"] = None
            row["entropy_token_count"] = 0
        _write_jsonl(args.output, [row], mode="a")
        done += 1
        if done % 25 == 0:
            print(f"entropy: {done} new items written ({len(completed) + done}/{len(rows)})")
    print(f"entropy complete: {len(completed) + done}/{len(rows)} items in {args.output}")


def summarize(args: argparse.Namespace) -> None:
    rows = _read_jsonl(args.input)
    if not rows:
        raise ValueError(f"no rows in {args.input}")
    expected = {int(row["index"]) for row in rows}
    if len(expected) != len(rows):
        raise ValueError("duplicate item indices in input")

    lengths = [int(row["response_tokens"]) for row in rows]
    entropy_rows = [row for row in rows if row.get("token_mean_entropy_nats") is not None]
    entropy_tokens = sum(int(row["entropy_token_count"]) for row in entropy_rows)
    weighted_entropy = (
        sum(float(row["token_mean_entropy_nats"]) * int(row["entropy_token_count"])
            for row in entropy_rows) / entropy_tokens
        if entropy_tokens else None
    )
    summary = {
        "source": str(args.input),
        "n": len(rows),
        "accuracy": fmean(bool(row["correct"]) for row in rows),
        "format_valid_rate": fmean(bool(row["format_valid"]) for row in rows),
        "answer_extracted_rate": fmean(bool(row["answer_extracted"]) for row in rows),
        "response_tokens_mean": fmean(lengths),
        "response_tokens_min": min(lengths),
        "response_tokens_max": max(lengths),
        "truncation_rate": fmean(row.get("finish_reason") == "length" for row in rows),
        "token_mean_entropy_nats": weighted_entropy,
        "entropy_items": len(entropy_rows),
        "entropy_tokens": entropy_tokens,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    gen = subparsers.add_parser("generate", help="sample and score one response per validation prompt")
    gen.add_argument("--model", type=Path, required=True, help="merged Hugging Face checkpoint")
    gen.add_argument("--data", type=Path, default=DEFAULT_DATA)
    gen.add_argument("--output", type=Path, required=True)
    gen.add_argument("--max-tokens", type=int, default=4096)
    gen.add_argument("--max-model-len", type=int)
    gen.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    gen.add_argument("--limit", type=int, help="smoke tests only; omit for reported results")
    gen.add_argument("--overwrite", action="store_true")
    gen.set_defaults(func=generate)

    ent = subparsers.add_parser("entropy", help="compute exact full-vocabulary response-token entropy")
    ent.add_argument("--model", type=Path, required=True, help="same merged checkpoint used to generate")
    ent.add_argument("--input", type=Path, required=True)
    ent.add_argument("--output", type=Path, required=True)
    ent.add_argument("--device", default="cuda:0")
    ent.add_argument("--overwrite", action="store_true")
    ent.set_defaults(func=entropy)

    summ = subparsers.add_parser("summarize", help="aggregate a generation/entropy JSONL file")
    summ.add_argument("--input", type=Path, required=True)
    summ.add_argument("--output", type=Path, required=True)
    summ.set_defaults(func=summarize)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

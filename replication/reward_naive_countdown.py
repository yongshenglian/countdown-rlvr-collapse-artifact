"""Frozen Countdown reward used by the October-2025 GRPO/PPO recipes.

The archived console logs establish the three reward values: 1.0 for a
correct expression, 0.1 when an equation is extracted but is invalid or
wrong, and 0.0 when no ``<answer>`` equation is extracted.
"""

import re


def _extract_solution(solution_str):
    if "Assistant:" in solution_str:
        solution_str = solution_str.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in solution_str:
        solution_str = solution_str.split("<|im_start|>assistant", 1)[1]

    matches = list(re.finditer(r"<answer>(.*?)</answer>", solution_str, re.DOTALL))
    return matches[-1].group(1).strip() if matches else None


def _valid_numbers(equation_str, available_numbers):
    try:
        numbers_in_equation = sorted(int(value) for value in re.findall(r"\d+", equation_str))
        return numbers_in_equation == sorted(available_numbers)
    except Exception:
        return False


def _evaluate(equation_str):
    if not re.match(r"^[\d+\-*/().\s]+$", equation_str):
        return None
    try:
        return eval(equation_str, {"__builtins__": None}, {})
    except Exception:
        return None


def _format_valid(solution_str):
    # Training prefilled the opening <think>, matching the historical reward's
    # format check. This diagnostic does not enter the scalar reward.
    text = "<think>" + solution_str
    return bool(re.search(r"<think>(.*?)</think>.*?<answer>(.*?)</answer>", text, re.DOTALL))


def compute_score(
    solution_str, ground_truth, method="strict", data_source=None, extra_info=None
):
    """Return the historical reward plus prospective diagnostic labels."""
    del method, data_source, extra_info
    equation = _extract_solution(solution_str)
    format_valid = _format_valid(solution_str)
    if equation is None:
        score, correct = 0.0, False
    elif not _valid_numbers(equation, ground_truth["numbers"]):
        score, correct = 0.1, False
    else:
        value = _evaluate(equation)
        correct = value is not None and abs(value - ground_truth["target"]) < 1e-5
        score = 1.0 if correct else 0.1
    return {
        "score": score,
        "acc": correct,
        "format_valid": format_valid,
        "answer_extracted": equation is not None,
    }

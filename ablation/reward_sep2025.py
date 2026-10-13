"""Reward used by the Sep 2025 DAPO runs (f8h77wol with DS, 3md2pzp8 without), for the matched ablation.

Wraps verl's countdown.compute_score_sep2025 (reconstructed from the f8h77wol console log; it reproduces the
logged critic/score/mean at all 499 steps) with the signature the DAPO reward manager calls:
    compute_score(data_source=..., solution_str=..., ground_truth=..., extra_info=...)
Scores: correct 0.9 + 0.1 format credit, wrong/invalid = format credit (0.1 or 0), no equation = 0.
Returns {"score", "acc", "pred"}; "acc" is what filter_groups.metric=acc filters on.
Prints one block per scored response (same as the original runs), so per-step p0/p1 can be recovered
from the console log with /home/lian/AI_PPO_2026/parse_ds_log.py.
"""
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "countdown_sep2025", "/home/lian/verl/verl/utils/reward_score/countdown.py")
_cd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cd)


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """Return the unchanged reward plus prospective diagnostic labels."""
    del data_source, extra_info
    result = _cd.compute_score_sep2025(solution_str, ground_truth)
    result["format_valid"] = bool(_cd.format_reward_func(solution_str) > 0)
    result["answer_extracted"] = bool(_cd.extract_solution(solution_str) is not None)
    return result

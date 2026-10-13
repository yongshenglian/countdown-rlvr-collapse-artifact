"""Ablation option A: correctness-only reward (no 0.1 format credit).

Same checker as reward_sep2025.py (countdown.compute_score_sep2025), but score = 1.0 if the equation is correct,
else 0.0. With this reward a group whose 8 answers are all wrong has zero variance, so it carries no training
signal by itself, without dynamic sampling.
"""
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "countdown_sep2025", "/home/lian/verl/verl/utils/reward_score/countdown.py")
_cd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cd)


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    r = _cd.compute_score_sep2025(solution_str, ground_truth)
    return {"score": 1.0 if r["acc"] else 0.0, "acc": r["acc"], "pred": r["pred"]}

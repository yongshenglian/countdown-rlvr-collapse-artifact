import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from eval_checkpoint_diagnostics import score_response, summarize


class ScorerTests(unittest.TestCase):
    def test_correct_strict_format(self):
        result = score_response(
            "work</think>\n<answer>(8 - 3) * 2</answer>", [8, 3, 2], 10
        )
        self.assertTrue(result["correct"])
        self.assertTrue(result["format_valid"])
        self.assertTrue(result["answer_extracted"])

    def test_correct_but_format_invalid(self):
        result = score_response("<answer>8 + 3 - 2</answer>", [8, 3, 2], 9)
        self.assertTrue(result["correct"])
        self.assertFalse(result["format_valid"])

    def test_reused_number_is_incorrect(self):
        result = score_response("</think><answer>8 + 8</answer>", [8, 3], 16)
        self.assertFalse(result["correct"])

    def test_last_answer_is_scored(self):
        result = score_response(
            "</think><answer>8 + 3</answer><answer>8 - 3</answer>", [8, 3], 5
        )
        self.assertTrue(result["correct"])
        self.assertEqual(result["equation"], "8 - 3")

    def test_unsafe_expression_is_rejected(self):
        result = score_response(
            "</think><answer>__import__('os')</answer>", [1], 1
        )
        self.assertFalse(result["correct"])


class SummaryTests(unittest.TestCase):
    def test_entropy_is_token_weighted(self):
        rows = [
            {"index": 0, "correct": True, "format_valid": True,
             "answer_extracted": True, "response_tokens": 1,
             "finish_reason": "stop", "token_mean_entropy_nats": 2.0,
             "entropy_token_count": 1},
            {"index": 1, "correct": False, "format_valid": False,
             "answer_extracted": True, "response_tokens": 3,
             "finish_reason": "length", "token_mean_entropy_nats": 4.0,
             "entropy_token_count": 3},
        ]
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "rows.jsonl"
            target = Path(directory) / "summary.json"
            source.write_text("".join(json.dumps(row) + "\n" for row in rows))
            summarize(Namespace(input=source, output=target))
            report = json.loads(target.read_text())
        self.assertEqual(report["accuracy"], 0.5)
        self.assertEqual(report["truncation_rate"], 0.5)
        self.assertEqual(report["token_mean_entropy_nats"], 3.5)


if __name__ == "__main__":
    unittest.main()

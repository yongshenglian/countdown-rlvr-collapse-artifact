"""Small regression checks for strict and format-tolerant scorer precedence."""
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / ".pydeps"))

import eval_bbh as B  # noqa: E402
import eval_gsm8k as G  # noqa: E402
import eval_math as M  # noqa: E402
import eval_mmlupro as U  # noqa: E402


def main():
    # A correct explicit marker cannot be overturned by later incidental text.
    assert G.score_response("#### 60\nUsed for 4 liters.", "60", "stop") == ("60", True, "60", True)
    # Marker-free fallback is valid only for a completed response.
    assert G.score_response("Therefore the answer is 20.", "20", "stop")[3] is True
    assert G.score_response("Intermediate value: 20", "20", "length")[3] is False

    # Symbolic fallback likewise cannot turn truncated work into a final answer.
    assert M.score({"gold": "2", "response": "An intermediate value is $2$.", "finish": "length"}, "standard")[3] is False
    assert M.score({"gold": "2", "response": r"Thus $\boxed{2}$.", "finish": "length"}, "standard")[1:4:2] == (True, True)

    bbh = {"gold": "(B)", "task": "causal_judgement", "input": "", "response": "maybe (B)", "finish": "length"}
    assert B.score(bbh, "standard")[3] is False
    mmlu = {"gold": "B", "response": "consider (B)", "finish": "length"}
    assert U.score(mmlu, "standard")[3] is False
    print("scorer regression checks passed")


if __name__ == "__main__":
    main()

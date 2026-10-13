"""Shared pieces for evaluating the DAPO checkpoints on downstream benchmarks.

Models are the step-300 actors of the two DAPO_new runs (merged to HF format by
merge_checkpoints.sh) plus the untrained base model as a reference.
Generation is greedy (temperature 0) with vLLM, one process per model / GPU.
"""
import glob
import json
import os
import re

MODELS = {
    "base": "Qwen/Qwen2.5-1.5B-Instruct",
    "noDS_step300": "/mnt/sdb/verl/DAPO_new/merged_hf/noDS_step300",
    "DS_step300": "/mnt/sdb/verl/DAPO_new/merged_hf/DS_step300",
    # improved PPO, /home/lian/verl/checkpoints/PPO_Countdown_R1_0923/ppo_qwen2.5_1.5b-train192-mini32/global_step_400
    "PPO_step400": "/mnt/sdb/verl/DAPO_new/merged_hf/PPO_step400",
    # PPO_Countdown_R1 run ua0u61fm: GAE, lr 3e-6, KL loss off, entropy 0, mini-batch 16 (see runs_registry.md)
    "PPO_klF_ent0_lr3e-6_step520": "/mnt/sdb/verl/DAPO_new/merged_hf/PPO_klF_ent0_lr3e-6_step520",
    # PPO_Countdown_R1 run dgwsksem: GAE, lr 3e-6, KL loss on (coef 0.001), entropy 0.001, mini-batch 16 (see runs_registry.md)
    "PPO_kl1e-3_ent1e-3_lr3e-6_step520": "/mnt/sdb/verl/DAPO_new/merged_hf/PPO_kl1e-3_ent1e-3_lr3e-6_step520",
    # PPO_Countdown_R1 run bwimlgaz: GAE, lr 3e-6, KL loss enabled but coef 0 (no effect), entropy 0.001, mini-batch 16 (see runs_registry.md)
    "PPO_kl0_ent1e-3_lr3e-6_step520": "/mnt/sdb/verl/DAPO_new/merged_hf/PPO_kl0_ent1e-3_lr3e-6_step520",
    # PPO_Countdown_R1 run txx0vrw6: GAE, lr 3e-6, KL loss off, entropy 0.001, mini-batch 16 (see runs_registry.md)
    "PPO_klF_ent1e-3_lr3e-6_step520": "/mnt/sdb/verl/DAPO_new/merged_hf/PPO_klF_ent1e-3_lr3e-6_step520",
    # PPO_Countdown_R1 run 4y7yxx3y: GAE, lr 1e-6, KL loss off, entropy 0, mini-batch 16 (see runs_registry.md)
    "PPO_klF_ent0_lr1e-6_step520": "/mnt/sdb/verl/DAPO_new/merged_hf/PPO_klF_ent0_lr1e-6_step520",
    # GRPO_Countdown_R1 run djmdy7s6: GRPO, lr 1e-6, n=2, KL loss 0.001, mini-batch 64 (see runs_registry.md)
    "GRPO_n2_kl1e-3_step520": "/mnt/sdb/verl/DAPO_new/merged_hf/GRPO_n2_kl1e-3_step520",
    # GRPO_Countdown_R1 run qde0tjh9: GRPO, lr 1e-6, n=2, KL loss off, mini-batch 64 (see runs_registry.md)
    "GRPO_n2_noKL_step520": "/mnt/sdb/verl/DAPO_new/merged_hf/GRPO_n2_noKL_step520",
    # GRPO_Countdown_R1 run 02o61vat: GRPO, lr 1e-6, n=2, KL loss 0.005, mini-batch 64 (see runs_registry.md)
    "GRPO_n2_kl5e-3_step520": "/mnt/sdb/verl/DAPO_new/merged_hf/GRPO_n2_kl5e-3_step520",
    # GRPO_Countdown_R1 run q0b8qi84: GRPO, lr 1e-6, n=2, KL loss 0.01 (stopped at 250), mini-batch 64 (see runs_registry.md)
    "GRPO_n2_kl1e-2_step250": "/mnt/sdb/verl/DAPO_new/merged_hf/GRPO_n2_kl1e-2_step250",
    # GRPO_Countdown_R1 run rq4tjmwu: GRPO, lr 1e-6, n=2, KL loss 0.04, mini-batch 64 (see runs_registry.md)
    "GRPO_n2_kl4e-2_step520": "/mnt/sdb/verl/DAPO_new/merged_hf/GRPO_n2_kl4e-2_step520",
    # GRPO_Countdown_R1 run m7xc02to: GRPO, lr 1e-6, n=4, KL loss 0.001, mini-batch 64 (see runs_registry.md)
    "GRPO_n4_kl1e-3_step520": "/mnt/sdb/verl/DAPO_new/merged_hf/GRPO_n4_kl1e-3_step520",
    # GRPO_Countdown_R1 run r2atlfqf: GRPO, lr 1e-6, n=8, KL loss 0.001, mini-batch 64 (see runs_registry.md)
    "GRPO_n8_kl1e-3_step520": "/mnt/sdb/verl/DAPO_new/merged_hf/GRPO_n8_kl1e-3_step520",
    # GRPO_Countdown_R1 run cthf0kfn: GRPO, lr 1e-6, n=4, KL loss 1e-6, standard reward despite name, mini-batch 64 (see runs_registry.md)
    "GRPO_n4_kl1e-6_step520": "/mnt/sdb/verl/DAPO_new/merged_hf/GRPO_n4_kl1e-6_step520",
    # GRPO_Countdown_R1 run dzp5op8l: GRPO, lr 1e-6, n=4, KL loss 1e-6, equal 0.5/0.5 reward, verl_fork, mini-batch 64 (see runs_registry.md)
    "GRPO_n4_kl1e-6_eqR_step520": "/mnt/sdb/verl/DAPO_new/merged_hf/GRPO_n4_kl1e-6_eqR_step520",
    # DAPO w/o DS checkpoint series (wandb 3md2pzp8, step 100)
    "noDS_step100": "/mnt/sdb/verl/DAPO_new/merged_hf/noDS_step100",
    # DAPO w/o DS checkpoint series (wandb 3md2pzp8, step 200)
    "noDS_step200": "/mnt/sdb/verl/DAPO_new/merged_hf/noDS_step200",
    # DAPO w/o DS checkpoint series (wandb 3md2pzp8, step 400)
    "noDS_step400": "/mnt/sdb/verl/DAPO_new/merged_hf/noDS_step400",
    # DAPO w/o DS checkpoint series (wandb 3md2pzp8, step 500)
    "noDS_step500": "/mnt/sdb/verl/DAPO_new/merged_hf/noDS_step500",
    # DAPO w/o DS checkpoint series (wandb 3md2pzp8, step 520)
    "noDS_step520": "/mnt/sdb/verl/DAPO_new/merged_hf/noDS_step520",
    # GRPO n=2, KL off, step-matched to KL 0.01 (wandb qde0tjh9, step 250)
    "GRPO_n2_noKL_step250": "/mnt/sdb/verl/DAPO_new/merged_hf/GRPO_n2_noKL_step250",
    # GRPO n=2, KL 0.001, step-matched (wandb djmdy7s6, step 250)
    "GRPO_n2_kl1e-3_step250": "/mnt/sdb/verl/DAPO_new/merged_hf/GRPO_n2_kl1e-3_step250",
    # GRPO n=2, KL 0.005, step-matched (wandb 02o61vat, step 250)
    "GRPO_n2_kl5e-3_step250": "/mnt/sdb/verl/DAPO_new/merged_hf/GRPO_n2_kl5e-3_step250",
    # GRPO n=2, KL 0.04, step-matched (wandb rq4tjmwu, step 250)
    "GRPO_n2_kl4e-2_step250": "/mnt/sdb/verl/DAPO_new/merged_hf/GRPO_n2_kl4e-2_step250",
    # ablation rung 2: clip 0.2/0.28, no DS, 32 prompts/step (wandb DAPO_ablation/5vmxmx4q), step 100
    "rung2_step100": "/mnt/sdb/verl/DAPO_new/merged_hf/rung2_step100",
    # ablation rung 2: clip 0.2/0.28, no DS, 32 prompts/step (wandb DAPO_ablation/5vmxmx4q), step 300
    "rung2_step300": "/mnt/sdb/verl/DAPO_new/merged_hf/rung2_step300",
    # ablation rung 2: clip 0.2/0.28, no DS, 32 prompts/step (wandb DAPO_ablation/5vmxmx4q), step 520
    "rung2_step520": "/mnt/sdb/verl/DAPO_new/merged_hf/rung2_step520",
    # ablation rung 1: clip 0.2/0.2, no DS, 32 prompts/step, step 100
    "rung1_step100": "/mnt/sdb/verl/DAPO_new/merged_hf/rung1_step100",
    # ablation rung 1: clip 0.2/0.2, no DS, 32 prompts/step, step 300
    "rung1_step300": "/mnt/sdb/verl/DAPO_new/merged_hf/rung1_step300",
    # ablation rung 1: clip 0.2/0.2, no DS, 32 prompts/step, step 520
    "rung1_step520": "/mnt/sdb/verl/DAPO_new/merged_hf/rung1_step520",
    # ablation rung 2, training seed 2, step 300
    "rung2s2_step300": "/mnt/sdb/verl/DAPO_new/merged_hf/rung2s2_step300",
    # ablation rung 2, training seed 2, step 520
    "rung2s2_step520": "/mnt/sdb/verl/DAPO_new/merged_hf/rung2s2_step520",
    # option A: correctness-only reward, step 300
    "rungA_step300": "/mnt/sdb/verl/DAPO_new/merged_hf/rungA_step300",
    # option A: correctness-only reward, step 520
    "rungA_step520": "/mnt/sdb/verl/DAPO_new/merged_hf/rungA_step520",
    # option B: no std normalisation, step 300
    "rungB_step300": "/mnt/sdb/verl/DAPO_new/merged_hf/rungB_step300",
    # option B: no std normalisation, step 520
    "rungB_step520": "/mnt/sdb/verl/DAPO_new/merged_hf/rungB_step520",
    # option C: uninformative groups masked, step 300
    "rungC_step300": "/mnt/sdb/verl/DAPO_new/merged_hf/rungC_step300",
    # option C: uninformative groups masked, step 520
    "rungC_step520": "/mnt/sdb/verl/DAPO_new/merged_hf/rungC_step520",
    # option D: batch-level advantage normalisation (reinforce_plus_plus_baseline), step 300
    "rungD_step300": "/mnt/sdb/verl/DAPO_new/merged_hf/rungD_step300",
    # option D: batch-level advantage normalisation (reinforce_plus_plus_baseline), step 520
    "rungD_step520": "/mnt/sdb/verl/DAPO_new/merged_hf/rungD_step520",
    # option A1: correctness-only reward + overlong filtering (truncated responses masked), step 300
    "rungA1_step300": "/mnt/sdb/verl/DAPO_new/merged_hf/rungA1_step300",
    # option A1: correctness-only reward + overlong filtering (truncated responses masked), step 520
    "rungA1_step520": "/mnt/sdb/verl/DAPO_new/merged_hf/rungA1_step520",
    # DS recipe, training seed 2 (collapsed; run ended at step 245 when data ran out), step 200
    "DSs2_step200": "/mnt/sdb/verl/DAPO_new/merged_hf/DSs2_step200",
    # option A1, training seed 2 (data.seed=2, rollout.seed=2), step 300
    "rungA1s2_step300": "/mnt/sdb/verl/DAPO_new/merged_hf/rungA1s2_step300",
    # option A1, training seed 2 (data.seed=2, rollout.seed=2), step 520
    "rungA1s2_step520": "/mnt/sdb/verl/DAPO_new/merged_hf/rungA1s2_step520",
}

# results/ holds the 2048-token run; set EVAL_RESULTS=results_8k (etc.) for other settings
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.environ.get("EVAL_RESULTS", "results"))

# Prompt used during Countdown RL training (verbatim wrapper, including its typos);
# the task sentence is swapped for the benchmark question.
THINK_TEMPLATE = (
    'system: You are a helpful assistant. You first thinks about the reasoning process in the mind '
    'and then provides the user with the answer."\n'
    "User: {question} Show your work in <think> </think> tags. And return the final answer in "
    "<answer> </answer> tags, for example <answer> {example} </answer>.\"\n"
    "Assistant: Let me solve this step by step.\n<think>\n"
)

NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?|-?\.\d+")


def to_chat(tokenizer, user_content):
    """Same wrapping verl used in training: one user message + generation prompt."""
    return tokenizer.apply_chat_template([{"role": "user", "content": user_content}],
                                         tokenize=False, add_generation_prompt=True)


def last_answer_tag(text):
    m = re.findall(r"<answer>(.*?)</answer>", text, re.S)
    return m[-1].strip() if m else None


def last_number(text):
    if text is None:
        return None
    m = NUM_RE.findall(text)
    return m[-1] if m else None


def norm_number(s):
    if s is None:
        return None
    s = s.replace(",", "").replace("$", "").strip().rstrip(".")
    try:
        v = float(s)
    except ValueError:
        return None
    return v


def numbers_equal(pred, gold, tol=1e-6):
    a, b = norm_number(pred), norm_number(gold)
    return a is not None and b is not None and abs(a - b) <= tol * max(1.0, abs(b))


def generate(model_path, prompts, max_tokens=2048, max_model_len=None, gpu_mem=0.85):
    from vllm import LLM, SamplingParams
    if max_model_len is None:
        max_model_len = max_tokens + 2048  # room for the longest prompts (MMLU-Pro/BBH < 2k tokens)
    llm = LLM(model=model_path, dtype="bfloat16", max_model_len=max_model_len,
              gpu_memory_utilization=gpu_mem, seed=0, enforce_eager=False)
    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    outs = llm.generate(prompts, sp)
    return [(o.outputs[0].text, o.outputs[0].finish_reason) for o in outs], llm.get_tokenizer()


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------------
# Generic driver used by eval_math.py / eval_bbh.py / eval_mmlupro.py
#   generation (GPU, vLLM) and scoring (CPU) are separate steps, so scoring rules can be
#   changed and re-applied to the saved responses without regenerating.
# ---------------------------------------------------------------------------------
STYLES = ["standard", "think"]
PAIR = ("noDS_step300", "DS_step300")


def run_generation(bench, model_key, items, build_prompt, max_tokens=2048):
    """items: list of dicts (must hold everything scoring needs). build_prompt(item, style) -> user text."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODELS[model_key])
    prompts = [to_chat(tok, build_prompt(it, s)) for s in STYLES for it in items]
    gens, _ = generate(MODELS[model_key], prompts, max_tokens=max_tokens)
    n = len(items)
    for i, s in enumerate(STYLES):
        rows = [dict(it, idx=j, response=gens[i * n + j][0], finish=gens[i * n + j][1]) for j, it in enumerate(items)]
        write_jsonl(os.path.join(RESULTS_DIR, bench, f"{model_key}__{s}.jsonl"), rows)
        print(f"{bench} {model_key} {s}: {n} generated, truncated {sum(r['finish'] == 'length' for r in rows) / n:.3f}")


def mcnemar_p(a, b):
    """Exact two-sided McNemar test on paired booleans."""
    from math import comb
    n01, n10 = int((~a & b).sum()), int((a & ~b).sum())
    k, m = min(n01, n10), n01 + n10
    p = min(1.0, 2 * sum(comb(m, i) for i in range(k + 1)) / 2 ** m) if m else 1.0
    return n10, n01, p


def summarize_benchmark(bench, score_fn, group_col=None, subsets=None):
    """score_fn(row: dict, style) -> (pred_strict, ok_strict, pred_flexible, ok_flexible).
    subsets: {name: set_of_idx} reported in addition to the full set (e.g. MATH-500)."""
    import pandas as pd
    out = os.path.join(RESULTS_DIR, bench)
    frames = {}
    for f in sorted(glob.glob(os.path.join(out, "*__*.jsonl"))):
        model, style = os.path.basename(f)[:-6].split("__")
        scored = os.path.join(out, "scored", f"{model}__{style}.jsonl")
        # reuse the scored file if it is newer than the responses; RESCORE=1 forces re-scoring
        # (needed after changing a scoring rule)
        if os.environ.get("RESCORE") != "1" and os.path.exists(scored) and os.path.getmtime(scored) >= os.path.getmtime(f):
            r = pd.read_json(scored, lines=True, dtype=False)
            r["correct_strict"] = r["correct_strict"].astype(bool); r["correct_flexible"] = r["correct_flexible"].astype(bool)
            frames[(model, style)] = r
            continue
        r = pd.read_json(f, lines=True, dtype=False)
        sc = [score_fn(row, style) for row in r.to_dict("records")]
        r["pred_strict"], r["correct_strict"], r["pred_flexible"], r["correct_flexible"] = map(list, zip(*sc))
        r["correct_strict"] = r["correct_strict"].astype(bool); r["correct_flexible"] = r["correct_flexible"].astype(bool)
        os.makedirs(os.path.join(out, "scored"), exist_ok=True)
        r.to_json(scored, orient="records", lines=True, force_ascii=False)
        frames[(model, style)] = r

    sets = {"all": None, **(subsets or {})}
    recs = []
    for (model, style), r in frames.items():
        for sname, idx in sets.items():
            rr = r if idx is None else r[r.idx.isin(idx)]
            for metric in ["strict", "flexible"]:
                p = rr[f"correct_{metric}"].mean()
                recs.append(dict(subset=sname, model=model, prompt=style, metric=metric, n=len(rr),
                                 acc=round(p, 4), se=round((p * (1 - p) / len(rr)) ** 0.5, 4),
                                 truncated=round((rr.finish == "length").mean(), 4)))
    s = pd.DataFrame(recs)
    s.to_csv(os.path.join(out, "summary.csv"), index=False)
    lines = []
    for sname in sets:
        lines.append(f"== {bench} [{sname}] ==")
        lines.append(s[s.subset == sname].pivot_table(index="model", columns=["prompt", "metric"], values="acc").to_string())
    for sname, idx in sets.items():
        for style in STYLES:
            if (PAIR[0], style) not in frames or (PAIR[1], style) not in frames:
                continue
            a, b = frames[(PAIR[0], style)], frames[(PAIR[1], style)]
            if idx is not None:
                a, b = a[a.idx.isin(idx)], b[b.idx.isin(idx)]
            for metric in ["strict", "flexible"]:
                n10, n01, p = mcnemar_p(a[f"correct_{metric}"].values, b[f"correct_{metric}"].values)
                lines.append(f"McNemar [{sname}] {style:8s} {metric:8s}: noDS-only {n10}, DS-only {n01}, p={p:.3g}")
    if group_col:
        g = []
        for (model, style), r in frames.items():
            t = r.groupby(group_col)["correct_strict"].mean().rename(f"{model}|{style}")
            g.append(t)
        by = pd.concat(g, axis=1).round(3)
        by.to_csv(os.path.join(out, f"by_{group_col}.csv"))
        lines.append(f"== strict accuracy by {group_col} ==")
        lines.append(by.to_string())
    txt = "\n".join(lines)
    open(os.path.join(out, "summary.txt"), "w").write(txt + "\n")
    print(txt)

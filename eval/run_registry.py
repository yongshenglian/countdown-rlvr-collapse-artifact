"""Record the exact training settings of every evaluated model -> runs_registry.csv / runs_registry.md

Settings are read from each run's wandb config (what the trainer actually received), not from the
experiment name. The checkpoint folder is matched to its run via trainer.default_local_dir.
In-domain Countdown validation at the evaluated step is taken from the run's history.

  python run_registry.py          # needs wandb login; re-run after adding a model to REGISTRY
"""
import os

import pandas as pd
import wandb

HERE = os.path.dirname(os.path.abspath(__file__))
ENTITY = "yongshenglian-university-of-louisville"

# key in common.MODELS -> (wandb project, run id, checkpoint folder, evaluated step)
REGISTRY = {
    "DS_step300": ("DAPO_new", "f8h77wol",
                   "/mnt/sdb/verl/DAPO_new/Qwen2.5-1.5B-G8-acc-filtering-tokenlevel-bsz-32-0910", 300),
    "noDS_step300": ("DAPO_new", "3md2pzp8",
                     "/mnt/sdb/verl/DAPO_new/Qwen2.5-1.5B-G8-acc-no-filtering-tokenlevel-bsz-32-0915", 300),
    "PPO_step400": ("PPO_Countdown_R1_0923", "w1d766ut",
                    "/home/lian/verl/checkpoints/PPO_Countdown_R1_0923/ppo_qwen2.5_1.5b-train192-mini32", 400),
    "PPO_klF_ent0_lr3e-6_step520": ("PPO_Countdown_R1", "ua0u61fm",
                                    "/mnt/sdb/verl/ckpts/PPO_Countdown_R1/ppo_qwen2.5_1.5b-train192-mini32-kllossFalse-entropy0-3lr-6", 520),
    "PPO_kl1e-3_ent1e-3_lr3e-6_step520": ("PPO_Countdown_R1", "dgwsksem",
        "/mnt/sdb/verl/ckpts/PPO_Countdown_R1/ppo_qwen2.5_1.5b-train192-mini32-klloss1-entropy-3lr-6", 520),
    "PPO_kl0_ent1e-3_lr3e-6_step520": ("PPO_Countdown_R1", "bwimlgaz",
        "/mnt/sdb/verl/ckpts/PPO_Countdown_R1/ppo_qwen2.5_1.5b-train192-mini32-klloss0-entropy1-3lr-6", 520),
    "PPO_klF_ent1e-3_lr3e-6_step520": ("PPO_Countdown_R1", "txx0vrw6",
        "/mnt/sdb/verl/ckpts/PPO_Countdown_R1/ppo_qwen2.5_1.5b-train192-mini32-kllossFalse-entropy1-3lr-6", 520),
    "PPO_klF_ent0_lr1e-6_step520": ("PPO_Countdown_R1", "4y7yxx3y",
        "/mnt/sdb/verl/ckpts/PPO_Countdown_R1/ppo_qwen2.5_1.5b-train192-mini32-kllossFalse-entropy0-1lr-6", 520),
    "GRPO_n2_kl1e-3_step520": ("GRPO_Countdown_R1", "djmdy7s6",
        "/mnt/sdb/verl/ckpts/GRPO_Countdown_R1/grpo_qwen2.5_1.5b-klloss1-entropy0-rollout2", 520),
    "GRPO_n2_noKL_step520": ("GRPO_Countdown_R1", "qde0tjh9",
        "/mnt/sdb/verl/ckpts/GRPO_Countdown_R1/grpo_qwen2.5_1.5b-klloss1-entropy0-rollout2-noKL-Loss", 520),
    "GRPO_n2_kl5e-3_step520": ("GRPO_Countdown_R1", "02o61vat",
        "/mnt/sdb/verl/ckpts/GRPO_Countdown_R1/grpo_qwen2.5_1.5b-klloss0.005-entropy0-rollout2", 520),
    "GRPO_n2_kl1e-2_step250": ("GRPO_Countdown_R1", "q0b8qi84",
        "/mnt/sdb/verl/ckpts/GRPO_Countdown_R1/grpo_qwen2.5_1.5b-klloss0.01-entropy0-rollout2", 250),
    "GRPO_n2_kl4e-2_step520": ("GRPO_Countdown_R1", "rq4tjmwu",
        "/mnt/sdb/verl/ckpts/GRPO_Countdown_R1/grpo_qwen2.5_1.5b-klloss0.04-entropy0-rollout2", 520),
    "GRPO_n4_kl1e-3_step520": ("GRPO_Countdown_R1", "m7xc02to",
        "/mnt/sdb/verl/ckpts/GRPO_Countdown_R1/grpo_qwen2.5_1.5b-klloss1-entropy0-rollout4", 520),
    "GRPO_n8_kl1e-3_step520": ("GRPO_Countdown_R1", "r2atlfqf",
        "/mnt/sdb/verl/ckpts/GRPO_Countdown_R1/grpo_qwen2.5_1.5b-klloss1-entropy0-rollout8", 520),
    "GRPO_n4_kl1e-6_step520": ("GRPO_Countdown_R1", "cthf0kfn",
        "/mnt/sdb/verl/ckpts/GRPO_Countdown_R1/grpo_qwen2.5_1.5b-klloss1-entropy0-rollout4-equalReward", 520),
    "GRPO_n4_kl1e-6_eqR_step520": ("GRPO_Countdown_R1", "dzp5op8l",
        "/mnt/sdb/verl/ckpts/GRPO_Countdown_R1/grpo_qwen2.5_1.5b-klloss1-entropy0-rollout4-equalReward-checkagain", 520),
    "noDS_step100": ("DAPO_new", "3md2pzp8",
        "/mnt/sdb/verl/DAPO_new/Qwen2.5-1.5B-G8-acc-no-filtering-tokenlevel-bsz-32-0915", 100),
    "noDS_step200": ("DAPO_new", "3md2pzp8",
        "/mnt/sdb/verl/DAPO_new/Qwen2.5-1.5B-G8-acc-no-filtering-tokenlevel-bsz-32-0915", 200),
    "noDS_step400": ("DAPO_new", "3md2pzp8",
        "/mnt/sdb/verl/DAPO_new/Qwen2.5-1.5B-G8-acc-no-filtering-tokenlevel-bsz-32-0915", 400),
    "noDS_step500": ("DAPO_new", "3md2pzp8",
        "/mnt/sdb/verl/DAPO_new/Qwen2.5-1.5B-G8-acc-no-filtering-tokenlevel-bsz-32-0915", 500),
    "noDS_step520": ("DAPO_new", "3md2pzp8",
        "/mnt/sdb/verl/DAPO_new/Qwen2.5-1.5B-G8-acc-no-filtering-tokenlevel-bsz-32-0915", 520),
    "GRPO_n2_noKL_step250": ("GRPO_Countdown_R1", "qde0tjh9",
        "/mnt/sdb/verl/ckpts/GRPO_Countdown_R1/grpo_qwen2.5_1.5b-klloss1-entropy0-rollout2-noKL-Loss", 250),
    "GRPO_n2_kl1e-3_step250": ("GRPO_Countdown_R1", "djmdy7s6",
        "/mnt/sdb/verl/ckpts/GRPO_Countdown_R1/grpo_qwen2.5_1.5b-klloss1-entropy0-rollout2", 250),
    "GRPO_n2_kl5e-3_step250": ("GRPO_Countdown_R1", "02o61vat",
        "/mnt/sdb/verl/ckpts/GRPO_Countdown_R1/grpo_qwen2.5_1.5b-klloss0.005-entropy0-rollout2", 250),
    "GRPO_n2_kl4e-2_step250": ("GRPO_Countdown_R1", "rq4tjmwu",
        "/mnt/sdb/verl/ckpts/GRPO_Countdown_R1/grpo_qwen2.5_1.5b-klloss0.04-entropy0-rollout2", 250),
    "rung2_step100": ("DAPO_ablation", "5vmxmx4q",
        "/mnt/sdb/verl/DAPO_ablation/rung2-clip0.28-noDS-gen32-train32-n8-resp4096", 100),
    "rung2_step300": ("DAPO_ablation", "5vmxmx4q",
        "/mnt/sdb/verl/DAPO_ablation/rung2-clip0.28-noDS-gen32-train32-n8-resp4096", 300),
    "rung2_step520": ("DAPO_ablation", "5vmxmx4q",
        "/mnt/sdb/verl/DAPO_ablation/rung2-clip0.28-noDS-gen32-train32-n8-resp4096", 520),
    "rung1_step100": ("DAPO_ablation", "name:rung1-grpo-clip0.2-noDS-gen32-train32-n8-resp4096",
        "/mnt/sdb/verl/DAPO_ablation/rung1-grpo-clip0.2-noDS-gen32-train32-n8-resp4096", 100),
    "rung1_step300": ("DAPO_ablation", "name:rung1-grpo-clip0.2-noDS-gen32-train32-n8-resp4096",
        "/mnt/sdb/verl/DAPO_ablation/rung1-grpo-clip0.2-noDS-gen32-train32-n8-resp4096", 300),
    "rung1_step520": ("DAPO_ablation", "name:rung1-grpo-clip0.2-noDS-gen32-train32-n8-resp4096",
        "/mnt/sdb/verl/DAPO_ablation/rung1-grpo-clip0.2-noDS-gen32-train32-n8-resp4096", 520),
    "rung2s2_step300": ("DAPO_ablation", "name:rung2-clip0.28-noDS-gen32-train32-n8-resp4096-seed2",
        "/mnt/sdb/verl/DAPO_ablation/rung2-clip0.28-noDS-gen32-train32-n8-resp4096-seed2", 300),
    "rung2s2_step520": ("DAPO_ablation", "name:rung2-clip0.28-noDS-gen32-train32-n8-resp4096-seed2",
        "/mnt/sdb/verl/DAPO_ablation/rung2-clip0.28-noDS-gen32-train32-n8-resp4096-seed2", 520),
    "rungA_step300": ("DAPO_ablation", "name:rungA-accOnlyReward-clip0.28-noDS-gen32-train32-n8-resp4096",
        "/mnt/sdb/verl/DAPO_ablation/rungA-accOnlyReward-clip0.28-noDS-gen32-train32-n8-resp4096", 300),
    "rungA_step520": ("DAPO_ablation", "name:rungA-accOnlyReward-clip0.28-noDS-gen32-train32-n8-resp4096",
        "/mnt/sdb/verl/DAPO_ablation/rungA-accOnlyReward-clip0.28-noDS-gen32-train32-n8-resp4096", 520),
    "rungB_step300": ("DAPO_ablation", "name:rungB-noStdNorm-clip0.28-noDS-gen32-train32-n8-resp4096",
        "/mnt/sdb/verl/DAPO_ablation/rungB-noStdNorm-clip0.28-noDS-gen32-train32-n8-resp4096", 300),
    "rungB_step520": ("DAPO_ablation", "name:rungB-noStdNorm-clip0.28-noDS-gen32-train32-n8-resp4096",
        "/mnt/sdb/verl/DAPO_ablation/rungB-noStdNorm-clip0.28-noDS-gen32-train32-n8-resp4096", 520),
    "rungC_step300": ("DAPO_ablation", "name:rungC-maskGroups-clip0.28-noDS-gen32-train32-n8-resp4096",
        "/mnt/sdb/verl/DAPO_ablation/rungC-maskGroups-clip0.28-noDS-gen32-train32-n8-resp4096", 300),
    "rungC_step520": ("DAPO_ablation", "name:rungC-maskGroups-clip0.28-noDS-gen32-train32-n8-resp4096",
        "/mnt/sdb/verl/DAPO_ablation/rungC-maskGroups-clip0.28-noDS-gen32-train32-n8-resp4096", 520),
    "rungD_step300": ("DAPO_ablation", "name:rungD-batchNorm-clip0.28-noDS-gen32-train32-n8-resp4096",
        "/mnt/sdb/verl/DAPO_ablation/rungD-batchNorm-clip0.28-noDS-gen32-train32-n8-resp4096", 300),
    "rungD_step520": ("DAPO_ablation", "name:rungD-batchNorm-clip0.28-noDS-gen32-train32-n8-resp4096",
        "/mnt/sdb/verl/DAPO_ablation/rungD-batchNorm-clip0.28-noDS-gen32-train32-n8-resp4096", 520),
    "rungA1_step300": ("DAPO_ablation", "name:rungA1-accOnlyReward-maskTruncated-clip0.28-noDS-gen32-train32-n8-resp4096",
        "/mnt/sdb/verl/DAPO_ablation/rungA1-accOnlyReward-maskTruncated-clip0.28-noDS-gen32-train32-n8-resp4096", 300),
    "rungA1_step520": ("DAPO_ablation", "name:rungA1-accOnlyReward-maskTruncated-clip0.28-noDS-gen32-train32-n8-resp4096",
        "/mnt/sdb/verl/DAPO_ablation/rungA1-accOnlyReward-maskTruncated-clip0.28-noDS-gen32-train32-n8-resp4096", 520),
    "DSs2_step200": ("DAPO_ablation", "name:rung3-DS-clip0.28-gen192-train32-n8-resp4096-seed2",
        "/mnt/sdb/verl/DAPO_ablation/rung3-DS-clip0.28-gen192-train32-n8-resp4096-seed2", 200),
    "rungA1s2_step300": ("DAPO_ablation", "name:rungA1-accOnlyReward-maskTruncated-clip0.28-noDS-gen32-train32-n8-resp4096-seed2",
        "/mnt/sdb/verl/DAPO_ablation/rungA1-accOnlyReward-maskTruncated-clip0.28-noDS-gen32-train32-n8-resp4096-seed2", 300),
    "rungA1s2_step520": ("DAPO_ablation", "name:rungA1-accOnlyReward-maskTruncated-clip0.28-noDS-gen32-train32-n8-resp4096-seed2",
        "/mnt/sdb/verl/DAPO_ablation/rungA1-accOnlyReward-maskTruncated-clip0.28-noDS-gen32-train32-n8-resp4096-seed2", 520),
}

# Things the wandb config does not record, established from the runs' own logs (see RESULTS.md section 2):
#   reward   : scheme used during training (from logged critic/score values and the printed reward output)
#   codebase : directory the training was launched from (location of the run's local wandb folder)
R_SEP = "correct 0.9+0.1 fmt / wrong 0.1 fmt / no-eq 0 (dapo manager, countdown.compute_score_sep2025)"
R_OCT = "correct 1.0 / eq. extracted but wrong 0.1 / no-eq 0 (naive manager)"
R_EQ = "EQUAL: correct 0.5+0.5 fmt / wrong 0.5 fmt / no-eq 0 (naive manager, verl_fork countdown.py)"
NOTES = {k: dict(codebase="/home/lian/verl") for k in REGISTRY}
ABL_NOTES = {"rungA": "correctness-only reward (ablation/reward_acc_only.py: correct 1, else 0)",
             "rungB": "algorithm.norm_adv_by_std_in_grpo=False",
             "rungC": "+algorithm.mask_uninformative_groups=True (trainer change, see ablation/README.md)",
             "rung2s2": "training seed 2 (data.seed=2, rollout.seed=2)",
             "rungD": "algorithm.adv_estimator=reinforce_plus_plus_baseline (group mean, batch whitening)",
             "DSs2": "DS recipe, training seed 2; collapsed by step 25; ended at step 245 (2 epochs of data used up by up to 3 generation rounds per step); last checkpoint step 200",
             "rungA1s2": "option A1 (correctness-only reward + overlong filtering), training seed 2 (data.seed=2, rollout.seed=2)",
             "rungA1": "correctness-only reward + +algorithm.mask_truncated=True (overlong filtering, trainer change)"}
for k in REGISTRY:
    if k.startswith("rung") or k.startswith("DSs2"):  # includes rungA1
        NOTES[k] = dict(note=ABL_NOTES.get(k.split("_")[0], ""),
                        codebase="/home/lian/verl code, run from the verl-fork conda env (torch 2.6.0, vLLM 0.8.5.post1)",
                        reward="correct 0.9+0.1 fmt / wrong 0.1 fmt / no-eq 0 (dapo manager, ablation/reward_sep2025.py)")
for k in REGISTRY:
    if k.startswith("DS_") or k.startswith("noDS_"):
        NOTES[k]["reward"] = R_SEP
for k in REGISTRY:
    if k.startswith("GRPO") or k.startswith("PPO_k"):
        NOTES[k]["reward"] = R_OCT  # October runs: logged scores 0 / 0.1 / 1
NOTES["PPO_step400"]["reward"] = ("correct 0.9+0.1 fmt / wrong 0.1 fmt / no-eq 0 (naive manager; logged scores "
                                  "include 0.9 = correct without format credit)")
NOTES["GRPO_n4_kl1e-6_step520"]["reward"] = R_OCT + " -- named 'equalReward' but logs show the standard reward"
NOTES["GRPO_n4_kl1e-6_eqR_step520"].update(reward=R_EQ, codebase="/home/lian/verl_fork/verl_fork")
NOTES["GRPO_n2_kl1e-2_step250"]["note"] = "run stopped early; last checkpoint is step 250"
NOTES["GRPO_n2_kl5e-3_step250"] = dict(NOTES["GRPO_n2_kl5e-3_step250"], note="step 250 comes from 02o61vat")
NOTES["GRPO_n2_kl5e-3_step520"]["note"] = "a short resume run (hqg8norw) also exists; step 520 comes from 02o61vat"

# (column, config path)
FIELDS = [
    ("adv_estimator", "algorithm.adv_estimator"),
    ("actor_lr", "actor_rollout_ref.actor.optim.lr"),
    ("actor_warmup_steps", "actor_rollout_ref.actor.optim.lr_warmup_steps"),
    ("critic_lr", "critic.optim.lr"),
    ("use_kl_loss", "actor_rollout_ref.actor.use_kl_loss"),
    ("kl_loss_coef", "actor_rollout_ref.actor.kl_loss_coef"),
    ("kl_loss_type", "actor_rollout_ref.actor.kl_loss_type"),
    ("use_kl_in_reward", "algorithm.use_kl_in_reward"),
    ("kl_in_reward_coef", "algorithm.kl_ctrl.kl_coef"),
    ("entropy_coeff", "actor_rollout_ref.actor.entropy_coeff"),
    ("clip_ratio_low", "actor_rollout_ref.actor.clip_ratio_low"),
    ("clip_ratio_high", "actor_rollout_ref.actor.clip_ratio_high"),
    ("loss_agg_mode", "actor_rollout_ref.actor.loss_agg_mode"),
    ("rollout_n", "actor_rollout_ref.rollout.n"),
    ("train_batch_size", "data.train_batch_size"),
    ("gen_batch_size", "data.gen_batch_size"),
    ("ppo_mini_batch_size", "actor_rollout_ref.actor.ppo_mini_batch_size"),
    ("ppo_epochs", "actor_rollout_ref.actor.ppo_epochs"),
    ("filter_groups", "algorithm.filter_groups.enable"),
    ("filter_metric", "algorithm.filter_groups.metric"),
    ("max_response_length", "data.max_response_length"),
    ("rollout_temperature", "actor_rollout_ref.rollout.temperature"),
    ("total_epochs", "trainer.total_epochs"),
    ("reward_manager", "reward_model.reward_manager"),
    ("gamma", "algorithm.gamma"),
    ("lam", "algorithm.lam"),
    ("base_model", "actor_rollout_ref.model.path"),
    ("train_files", "data.train_files"),
    ("checkpoint_dir_in_config", "trainer.default_local_dir"),
]


HYDRA_ROOTS = ["/home/lian/verl/examples/ppo_trainer/outputs", "/home/lian/verl/examples/grpo_trainer/outputs",
               "/home/lian/verl_fork/verl_fork/examples/grpo_trainer/outputs", "/home/lian/verl/outputs",
               "/home/lian/verl_fork/verl_fork/outputs"]
CONFIG_DIR = os.path.join(HERE, "run_configs")


def find_launch_config(exp_name, created_utc):
    """Hydra config saved at launch: same experiment name, launch time closest to the wandb start."""
    import datetime as dt
    import glob
    import re
    import yaml
    from zoneinfo import ZoneInfo
    start = dt.datetime.fromisoformat(created_utc.replace("Z", "+00:00"))
    best = None
    for root in HYDRA_ROOTS:
        for f in glob.glob(os.path.join(root, "*", "*", ".hydra", "config.yaml")):
            m = re.search(r"(\d{4}-\d{2}-\d{2})/(\d{2}-\d{2}-\d{2})/\.hydra", f)
            if not m:
                continue
            try:
                c = yaml.safe_load(open(f))
            except Exception:
                continue
            if get(c, "trainer.experiment_name") != exp_name:
                continue
            t = dt.datetime.strptime(" ".join(m.groups()), "%Y-%m-%d %H-%M-%S").replace(
                tzinfo=ZoneInfo("America/New_York"))
            gap = abs((t - start).total_seconds())
            if best is None or gap < best[0]:
                best = (gap, f, c)
    return best


def get(cfg, path):
    d = cfg
    for k in path.split("."):
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def main():
    api = wandb.Api()
    rows = []
    for key, (proj, rid, ckdir, step) in REGISTRY.items():
        if rid.startswith("name:"):  # resolve by run name (latest run with that name)
            cands = [x for x in api.runs(f"{ENTITY}/{proj}") if x.name == rid[5:]]
            if not cands:
                print("not in wandb yet:", key); continue
            rid = sorted(cands, key=lambda x: x.created_at)[-1].id
        r = api.run(f"{ENTITY}/{proj}/{rid}")
        c = r.config
        row = dict(model_key=key, wandb_project=proj, wandb_run=rid, run_name=r.name, state=r.state,
                   created=r.created_at, last_logged_step=r.summary.get("_step"),
                   checkpoint=f"{ckdir}/global_step_{step}", eval_step=step)
        cfg_dir = get(c, "trainer.default_local_dir") or ""
        row["ckpt_dir_matches_run"] = os.path.basename(cfg_dir.rstrip("/")) == os.path.basename(ckdir.rstrip("/"))
        row["ckpt_exists"] = os.path.isdir(f"{ckdir}/global_step_{step}/actor")
        for col, path in FIELDS:
            row[col] = get(c, path)
        # second source: the Hydra config written at launch; compare the recorded settings with wandb
        lc = find_launch_config(r.name, r.created_at)
        if lc and lc[0] < 3 * 3600:
            gap, f, hc = lc
            import shutil
            os.makedirs(CONFIG_DIR, exist_ok=True)
            shutil.copy(f, os.path.join(CONFIG_DIR, f"{key}.yaml"))
            def same(a, b):
                try:
                    return float(a) == float(b)  # 1 vs 1.0, 0 vs 0.0
                except (TypeError, ValueError):
                    return str(a) == str(b)
            diffs = [f"{col} (launch {get(hc, path)!r}, wandb {get(c, path)!r})" for col, path in FIELDS
                     if col != "checkpoint_dir_in_config" and not same(get(hc, path), get(c, path))]
            row["launch_config"] = f
            row["launch_config_minutes_from_start"] = round(gap / 60, 1)
            row["launch_config_vs_wandb"] = "identical" if not diffs else "differs: " + ", ".join(diffs)
        else:
            row["launch_config"] = None
            row["launch_config_vs_wandb"] = "no launch config found (launched via ray job submit)" \
                if proj == "DAPO_new" else "no launch config found"
        h = pd.DataFrame(list(r.scan_history(keys=["_step", "val-core/countdown/acc/mean@1"]))) \
            if any(k.startswith("val-core/countdown/acc") for k in r.summary.keys()) else pd.DataFrame()
        vk = "val-core/countdown/acc/mean@1"
        if h.empty:
            h = pd.DataFrame(list(r.scan_history(keys=["_step", "val-core/countdown/reward/mean@1"])))
            vk = "val-core/countdown/reward/mean@1"
        sc = pd.DataFrame(list(r.scan_history(keys=["critic/score/max", "critic/score/min"])))
        row["logged_score_values"] = " ".join(str(v) for v in sorted(
            set(sc["critic/score/max"].round(3)) | set(sc["critic/score/min"].round(3)))) if not sc.empty else None
        for nk, nv in NOTES.get(key, {}).items():
            row[nk] = nv
        row["countdown_val_metric"] = vk.split("/")[2]
        near = h[(h._step - step).abs() <= 5].sort_values("_step") if not h.empty else h
        row["countdown_val_at_eval_step"] = round(near[vk].iloc[-1], 4) if len(near) else None
        row["countdown_val_best"] = round(h[vk].max(), 4) if not h.empty else None
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(HERE, "runs_registry.csv"), index=False)
    t = df.set_index("model_key").T
    with open(os.path.join(HERE, "runs_registry.md"), "w") as f:
        f.write("# Evaluated models: training settings (from wandb configs)\n\n")
        f.write("Generated by `run_registry.py`. Do not edit by hand.\n\n")
        f.write("| setting | " + " | ".join(t.columns) + " |\n")
        f.write("|---" * (len(t.columns) + 1) + "|\n")
        for idx, vals in t.iterrows():
            f.write(f"| {idx} | " + " | ".join("" if v is None else str(v) for v in vals) + " |\n")
    print(t.to_string())


if __name__ == "__main__":
    main()

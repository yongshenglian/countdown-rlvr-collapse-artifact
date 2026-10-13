# Prospective Entropy and Format-Validity Diagnostics

Decision date: 2026-10-12.

This protocol was written while the first seed-3 no-DS continuation was already
running and before submission of the remaining queued seed-4/5 DAPO
continuations and all queued strong-KL GRPO/PPO replications. It does not alter
the predeclared training horizon, checkpoint rule, scalar reward, or optimization
coefficients in `replication/PROTOCOL.md`.

## Online metrics for subsequently submitted jobs

Two reward-side Boolean diagnostics are returned alongside the unchanged scalar
score:

- `format_valid`: after accounting for the prompt-prefilled opening
  `<think>`, the response contains a closing `</think>` followed by a
  complete `<answer>...</answer>` span;
- `answer_extracted`: the last complete `<answer>...</answer>` span can be
  extracted, irrespective of correctness.

The reward managers expose their means for training batches and for the full
1,024-item Countdown validation set. `acc` remains the correctness field and
the scalar rewards remain exactly:

- DAPO: correct 0.9 plus 0.1 format credit; incorrect 0.1 only when the strict
  format is valid; otherwise 0;
- strong-KL GRPO/PPO: correct 1.0; extracted but incorrect 0.1; otherwise 0.

Unit tests in the repository exercise correct, incorrect, unformatted-correct,
and missing-answer cases and verify that adding diagnostics does not change
those scalar values.

For the DAPO trainer, `actor_rollout_ref.actor.log_entropy=True` computes the
per-token categorical Shannon entropy in nats over the sampled response tokens
used in each actor update and logs its global token-mean as
`actor/entropy_loss`. The optimization coefficient remains
`entropy_coeff=0`; the logged entropy is not added to the loss and therefore
does not change the stated objective. The standard PPO/GRPO trainer already
computes the same token-mean entropy during old-log-probability recomputation.

## Seed-3 transition and comparability

The no-DS seed-3 continuation was submitted before this instrumentation. Its
format-validity rate can be reconstructed exactly from the per-response reward
log because the historical DAPO format credit is printed for every response.
Its online token entropy was not computed and must not be inferred from response
length, KL, or reward.

After training completes, a common post-hoc checkpoint diagnostic will be run
for every replication seed at saved steps 100, 150, ..., 500, and the terminal
step 520. It will use the full vendored 1,024-item validation set, the recorded
Qwen snapshot and chat template, fixed generation seed 0, temperature 1.0,
top-p 0.7, top-k disabled, one response per prompt, and the recipe response cap.
For each checkpoint it will report response length, truncation, correctness,
strict format validity, answer extraction, and token-mean entropy of the sampled
response under that same checkpoint. This common post-hoc series is the
cross-seed entropy comparison; online training-batch entropy is a supplementary
within-run diagnostic.

The executable implementation is `replication/eval_checkpoint_diagnostics.py`.
It separates vLLM generation from Transformers entropy recomputation so only
one model copy occupies the GPU at a time. The generation JSONL records the
exact prompt and sampled-response token IDs; the entropy phase then evaluates
the full vocabulary distribution at every recorded response-token position.
Thus the reported quantity is exact categorical Shannon entropy in nats, not a
top-k-log-probability approximation. The entropy phase is resumable at item
boundaries, and the summarizer weights item means by their response-token
counts to reproduce a global token mean. A reduced `--limit` run is permitted
only as a smoke test and must not be reported.

## Source provenance

- The exact source patch packaged for the already-running seed-3 continuation is
  `environment/verl-training-working-tree-seed3.patch`.
- The prospective source patch containing diagnostic-only entropy logging is
  `environment/verl-training-working-tree-prospective.patch`.
- Format diagnostics are implemented in `ablation/reward_sep2025.py` and
  `replication/reward_naive_countdown.py`.
- `ablation/common.sh` enables entropy logging for subsequent DAPO jobs.

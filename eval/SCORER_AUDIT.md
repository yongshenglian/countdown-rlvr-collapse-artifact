# Scorer Audit Record

Date: 12 October 2026

This is an author-side code-and-record audit, not an independent validation. It
documents the scorer correction made during the Round 2 revision and preserves
a deterministic sample for later second-person adjudication.

## Audit sample

`make_scorer_audit.py` selects 40 saved standard-prompt GSM8K records targeted
at the last-number fallback: 24 base-model responses scored correct only by
fallback, eight base-model marker-free responses with an incorrect fallback
result, four base-model strict-correct controls, and one fallback-only correct
response from each of four trained checkpoints. This allocation directly tests
the fallback on which the reported base-model tolerant accuracy depends; it is
not a prevalence-weighted error-rate sample. SHA-256 ranking makes the sample
deterministic. The resulting records, raw responses, extracted predictions,
labels, finish reasons, and blank adjudication fields are stored in
`scorer_audit_sample.jsonl`.

## Defects found

1. The GSM8K flexible scorer previously used the last number unconditionally.
   A correct explicit marker could therefore be overwritten by a later
   incidental number. For example, a response with `#### 60` followed by the
   phrase “4 liters” was strict-correct but flexible-incorrect.
2. Marker-free fallback extraction was allowed for length-truncated outputs.
   In several MATH records, unfinished looping work was marked flexible-correct
   because the gold expression occurred incidentally before truncation.

## Corrected policy

- Format-tolerant correctness is a strict superset: a correct explicit final
  marker is always accepted.
- Marker-free last-number, symbolic-expression, or choice-letter fallback is
  used only when generation did not end with `finish_reason=length`.
- The same rule is applied to deterministic and pass@k evaluation.

The primary 8,192-token archive migration invalidated 1,323 truncated fallback
labels across its full sweep; the secondary 2,048-token archive was migrated
under the same rule. In MATH-500 pass@k caches, 198 truncated
fallback sample labels were invalidated. GSM8K pass@k caches were rebuilt from
all saved samples so that correct explicit markers could not be overwritten.
Among the five manuscript headline checkpoints, the largest change to any one
benchmark accuracy was 0.91 percentage points. The qualitative findings did not
change.

## Reproducibility checks

- `test_scorers.py` covers correct-marker precedence and disabled truncated
  fallback for all four benchmark scorers.
- `rescore_truncation_policy.py` performs the archive migration without model
  inference. The MATH-500 pass@k migration is version-tagged and idempotent.
- `analysis_gsm8k_vs_math.py`, `make_results_md.py`, and `eval_passk.py` were
  rerun after migration.

## Remaining validation

The independent audit is prepared but not yet performed. The deterministic
blinding script writes 40 opaque-ID records with all human fields blank. The
auditor receives only `independent_audit/records_to_complete.jsonl`, the
protocol, and the metadata form; the coordinator withholds automatic labels and
model identities until the first pass is frozen. The protocol defines extraction
rules, a seven-category error codebook, discrepancy adjudication, and sign-off
requirements. `report_independent_audit.py` validates completeness and reports
strict/tolerant agreement with Wilson intervals and every disagreement.

No independent-validation claim will be made until a second person completes
and signs this packet. Immutable dataset revisions or exact evaluated file

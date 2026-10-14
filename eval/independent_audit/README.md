# Independent Scorer Audit Protocol

Purpose: independently check answer extraction and correctness labels for the
40-record deterministic GSM8K sample used in the revision. This protocol
separates a blinded first pass from discrepancy adjudication.

## Targeted sample

The packet deliberately emphasizes the last-number fallback: it contains 24
base-model responses scored correct only by fallback, eight base-model
marker-free responses with an incorrect fallback result, four base-model
strict-correct controls, and four trained-model fallback-only correct cases.
The allocation evaluates the fallback's validity where it matters most for the
reported comparison. It is stratified and is not a prevalence-weighted estimate
of the scorer's overall error rate.

## Files and roles

The coordinator runs:

```bash
python eval/prepare_independent_audit.py
```

The independent auditor receives only:

- `records_to_complete.jsonl`
- a copy of this protocol
- `auditor_metadata.json`

The coordinator withholds `coordinator_key.jsonl` and the unblinded
`scorer_audit_sample.jsonl` until the first pass is complete. The auditor must
not inspect evaluation code, automatic predictions, automatic correctness
labels, manuscript result tables, or model identities before returning the
first-pass file.

## Fields to complete

For every record, the auditor fills:

- `human_strict_prediction`: the answer extracted from the final explicit
  marker, or `null` if no permitted marker exists;
- `human_tolerant_prediction`: the strict prediction when it is correct;
  otherwise the permitted marker-free fallback, or `null`;
- `human_strict_correct`: JSON `true` or `false`;
- `human_tolerant_correct`: JSON `true` or `false`;
- `human_error_category`: initially `no_error`, unless the record itself has
  an ambiguous expression or suspect reference;
- `human_note`: a short explanation for ambiguity or any non-`no_error`
  category.

The auditor also completes `auditor_metadata.json`, including whether they are
independent of the authors. Predictions should be strings, preserving meaningful
signs, fractions, option letters, and mathematical notation.

## Scoring rules supplied to the auditor

The strict answer is taken from the explicit final-answer marker that occurs
last in the response:

- GSM8K: `####`, `\\boxed{...}`, or `<answer>...</answer>`;
- MATH: `\\boxed{...}` or `<answer>...</answer>`, judged by mathematical
  equivalence to the reference;
- BBH: “the answer is ...”, `<answer>...</answer>`, or `\\boxed{...}`;
- MMLU-Pro: “the answer is (X)”, `<answer>X</answer>`, or
  `\\boxed{X}`.

Tolerant correctness is a strict superset. If the strict marker is correct,
tolerant correctness is also true. Otherwise, marker-free fallback is allowed
only when `finish` is not `length`:

- GSM8K: last numeric string;
- MATH: a mathematically equivalent expression recoverable from the response;
- BBH and MMLU-Pro: last parenthesized option letter, with the documented BBH
  task-specific normalization for Dyck-language and word-sorting items.

No marker-free fallback is permitted for a length-truncated response.

## Unblinding and adjudication

After the completed first pass is frozen and hashed, the coordinator runs:

```bash
python eval/report_independent_audit.py \
  --completed eval/independent_audit/records_completed.jsonl \
  --metadata eval/independent_audit/auditor_metadata_completed.json
```

The script compares human and automatic correctness labels and writes a
discrepancy report. The auditor and author then review only the discrepancies.
For each discrepancy, they assign exactly one category:

- `extraction_error`: the automatic scorer selected the wrong explicit or
  fallback answer;
- `incidental_match`: an unrelated token or expression was credited;
- `ambiguous_expression`: reasonable readers or symbolic parsers may differ;
- `reference_error`: the supplied gold/reference is wrong or ambiguous;
- `truncation_policy_error`: fallback was incorrectly allowed or denied due to
  finish status;
- `other`: none of the above, explained in the note;
- `no_error`: the apparent disagreement was resolved without a scorer defect.

The final report must state raw agreement counts for strict and tolerant
correctness, every discrepancy and resolution, category counts, the auditor
identifier, the date, and hashes of the completed files. The authors must not
describe this item as independently validated until those completed files and
the signed-off report exist.

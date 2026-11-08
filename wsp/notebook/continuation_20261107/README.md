# Direct weight editing: six-pass continuation

**Outcome: no improvement demonstrated; retained the exact original F4 edit.**
All 25 allowed passes are now used. Four additional direct-edit candidates were
screened without gradients, optimizers, fine-tuning, prompt changes, or SQL repair.
No candidate earned a new dev evaluation. The earlier measured dev score remains
30.12%; the requested 40–50% target was not achieved.

On the same 512 practice questions:

| Edit | Correct | Execution accuracy |
| --- | ---: | ---: |
| Original F4, rerun as final control | 201 | 39.26% |
| Delete 64 anti-binding neurons | 161 | 31.45% |
| Amplify two binding heads by 1.5x | 197 | 38.48% |
| Discourage T aliases after SELECT/DISTINCT | 193 | 37.70% |
| Suppress selected gates at FROM positions | 186 | 36.33% |

These are **practice scores**, not dev scores. The control reproduced all
512 prior F4 outputs exactly, confirming that the screened candidates' changes
were attributable to their edits in this cohort. This does not establish that
vLLM is deterministic across all batches or prompts.

The useful negative result is that reducing alias swaps alone can increase
missing joins or other schema errors. Better two-table counterfactual binding
also did not guarantee better generated queries. This does not prove that
40–50% is impossible with direct editing; these four approaches did not get there.

Machine-readable comparisons: `practice_results.json`; final choice:
`decision.json`. The original submission and all new scripts are preserved here.


The user requested a stronger result while keeping direct weight edits and no
training. The original submission and exact bf16 weights are archived here.
The 25-pass budget is unchanged; this continuation starts at pass 20.

The editor uses only practice questions, previously recorded training activations,
and base weights. Dev predictions, other methods' weights, gradients, optimizers,
prompt changes, and SQL repair are excluded. The existing dev aggregate is already
known from the previous report, so any final dev evaluation is a reuse of that
benchmark and must not be described as a newly unseen holdout.

Two hypotheses, chosen before any new generation results:

1. **Binding neurons:** zero 64 layer-27 MLP columns with the most negative mean
   contribution to the known alias-binding circuit. Attribution comes from existing
   own/swap counterfactuals; preserve all 32 F4 format columns. The removed mean
   counterfactual effect is -1.371 nats. At ordinary cached answer positions, the
   removed residual RMS is median 0.050, p99 0.509. This may still harm other skills.
2. **Binding heads:** amplify L22H7 and L23H4 output columns by 1.5, retaining exact
   F4. Earlier work tried 2x with a forced T1 alias, which failed. This candidate
   keeps the original alias choice. The cached direct-path estimate improves T2
   binding consistency from 18/32 to 20/32 and leaves T1 at 28/32. This local
   estimate is diagnostic, not execution-accuracy evidence.

Plan: test each hypothesis on the same 512-question seed-0 sample and on fresh
practice samples within the six remaining calls. Compare correctness against both
the base and the previous F4 outputs where available. Evaluate aliases separately
because fewer SQL errors alone do not imply higher execution accuracy. Account for
the cached-baseline numerical variation already measured in pass 13. Retain F4 if
no candidate gives consistent execution-accuracy evidence beyond this variation.

No claim of reaching 40% or 50% can be made without execution evaluation.

## Pass 20 decision

Binding-64 regressed substantially: base 198/512, F4 201/512, candidate 161/512.
It fixed 20 and broke 60 relative to F4 (net -40); alias-swap failures fell from
67 to 40, but absent-table and invented-column failures increased. The neurons
also carry useful schema information. Reject this candidate rather than spend
two more passes replicating a clear regression. An execution guard in its script
prevents the already-started sequential runner from spending pass 22 on it.
Pass 21 (heads-1.5) continues unaffected. Subsequent passes will be chosen after
that result; the original two-candidate replication plan is superseded.

## Pass 21 decision and pass 22 hypothesis

Heads-1.5: base 198/512, F4 201/512, candidate 197/512. No accuracy gain;
do not spend more calls on this candidate.

Pass 22 instead tests a sparse rank-one naming-habit edit. At positions immediately
after SELECT or DISTINCT, suppress the next token ` T` by approximately 4 nats,
protecting the 400 most frequent other output tokens. The model may then choose
column or table names directly rather than commit to a T alias before FROM.
This is a hypothesis, not a guaranteed means of eliminating aliases. The position
key is a one-shot ridge solve on existing pass-3 activations. A split by question
audits selectivity: 103 audit positives have mean score 1.019 and median 1.005;
4,866 audit negatives have median 0.011 and p99 0.462. Negative-position leakage
and alternative alias conventions remain risks. Retain the exact F4 format edit.

## Pass 22 result, pass 23 read, and final candidate

The qualifier candidate scored 193/512 (37.70%), below F4's 201/512. Alias swaps
fell from 67 to 13, while missing joins increased. Reject it. The feature changes
the symptom without preserving the rest of the model's SQL plan.

Pass 23 captures the original model's layer-26 residuals for 64 full practice
responses, selected by their existing order and the presence of FROM, without
correctness labels. CPU algebra on these captures reconstructs the last layer's
post-attention normalized input. One covariance solve identifies a FROM-position
key, with a question-level audit split. At 17 audit FROM positions its median
score is 0.954; at 556 other answer positions its median is 0.007, p99 0.261.

The first algebraic proposal removed the selected gate rows' mean FROM response.
It reduced the cached counterfactual binding effect from 1.938 to 1.698 nats,
so it was not tested with generation. Inspection explains why: most gate values
at FROM are negative, and moving them to zero does not approximate silencing
them reliably. A fixed -8 shift at FROM instead removes 97% of the selected
neurons' activation RMS there. This magnitude was selected for mechanical
suppression, not by optimizing execution accuracy.

The final candidate updates only 64 rows of layer-27 gate_proj with a rank-one
matrix, retaining exact F4 down_proj. The existing own/swap read captures estimate
T2 binding consistency 12/24 -> 17/24, T1 21/24 -> 20/24, and the counterfactual
binding effect 1.938 -> 3.357 nats. These are direct-path approximations, not EX.
The estimated other-answer residual perturbation is median 0.0073, p99 0.091,
versus 0.050 and 0.509 for deleting these neurons everywhere.

Pass 24 will test this candidate on seed 0; pass 25 will check it on seed 2026
unless pass 24 shows a clear destructive regression. Neither sample is dev.
The candidate uses information from some seed-0 examples, so its first result
is development evidence, while the seed-2026 sample is not used in its design.

## Pass 24 decision and final control

The selective gate candidate scored 186/512 (36.33%), versus 201/512 for cached
F4. It fixed 11 and broke 26 relative to F4, with 174 no-such-column errors versus
the original base's 160. Better cached counterfactual binding did not translate
into better execution; generation can choose tables outside the two considered
by the diagnostic. Reject it rather than use the final call as a replication of
a materially worse candidate.

Restore the exact archived F4 tensor and remove the gate edit. Pass 25 instead
reruns F4 on the same seed-0 batch as an unchanged-weight control. This checks
the numerical variation of the harness and supplies a current matched reference
for all four screened candidates. No new candidate merits a dev evaluation.

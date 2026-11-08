# Extended direct-weight experiment

## Completed outcome

The expanded experiment used **59 additional harness passes** (84/100 total),
including **44 screened weight candidates**, over about 110.5 minutes.
**No candidate passed separate confirmation.** The exact original F4 edit
`ae81f3cb05285df3` is restored and retained. Its previously measured BIRD dev
score remains **462/1,534 = 30.12%** (base 28.49%). **40–50% was not achieved.**
No new dev evaluation was run, and no training was used.

The strongest screening edits strengthened identifier-copy head L27H8. On the
same separate confirmation questions, none improved the F4 control:

| L27H8 output gain | Screen (F4 91/256) | Confirmation (F4 88/256) | Confirmation fixed / broke |
|---|---:|---:|---:|
| 3 | 96/256 | 81/256 | 2 / 9 |
| 2 | 95/256 | 83/256 | 3 / 8 |
| 1.5 | 93/256 | 88/256 | 3 / 3 |

These screening gains did not replicate. Under the recorded acceptance rule,
no candidate advances to the reserved 2,048-question validation or a new dev
measurement. The 100-pass cap is a ceiling; the remaining 16 passes are unused
after confirmation rejected every finalist. This result does not establish
that larger gains are impossible; it establishes no reliable gain from the
mechanisms tested here.

See [decision.json](decision.json), [results_summary.json](results_summary.json),
[audit.json](audit.json), and the per-pass reports for complete results. Ledger
and prior submission history are preserved. Edited tensors are archived for
passes 36 onward; early rejected passes 30–33 retain recipes and fingerprints,
and pass 34 is the exact archived F4 control. The only harness change excludes
synthetic diagnostic prompts from default random sampling; both regression
tests pass. Original prompts, decoding, and SQL extraction/grading are used
for every reported edit score. No Portfolio files were changed by this run.

The user authorized a more relaxed budget and more time on November 7, 2026.
The cap is raised from 25 to 100 total passes, preserving every previous pass
and submission. The exact original F4 tensor remains archived as the fallback.
No gradients, optimizer, supervised fitting of weight updates to a SQL loss,
other fine-tuned checkpoints, or benchmark-specific SQL repair is allowed.
All final generation prompts, decoding, and SQL grading remain unchanged.

## More representative practice data

The earlier pool contained only 5,233 train questions with prompts <=2,048 tokens.
Append 3,528 non-truncated BIRD **train** questions with 2,049–3,072-token prompts.
IDs are `x{question_id:05d}`; original `p` IDs are unchanged. The original practice
file and its SHA256 are archived. No dev examples are inspected or selected on.

`screen512.txt` contains 256 existing short questions and 256 new long questions.
`validate512.txt` contains 256 short and 256 long questions disjoint from the
screen and from every previously generated practice cohort. Preserve this
validation set for candidates that improve screening accuracy. Both remain
training-split data; the public dev benchmark has already been evaluated in
earlier experiments and must not be called a newly unseen holdout.

## Research sequence

1. Measure F4 on the balanced screening cohort (pass 26).
2. Probe whether additional schema, planning, or expert instructions expose
   better behavior in the unchanged model. These counterfactual prompts are
   diagnostics only; they are never used for the reported direct-edit score.
3. Use paired activation differences from the instruction probes to design
   one-shot, CPU, rank-one weight interventions that steer the model toward
   behavior it already expresses. Use original prompts for every edit screening.
4. Inspect schema-retrieval attention heads and long-context copying as a
   separate mechanism. Retest a locally promising change on execution accuracy.
5. Check promising candidates on the disjoint validation cohort and retain the
   original F4 if improvements fail to replicate. Freeze any selected edit
   before a final dev evaluation; no dev-driven tuning.

The larger cap permits hypothesis development and independent checks. It is
not permission to silently change the benchmark or report an unverified score.

## Measurements through pass 34

F4 on the balanced 512-question cohort: 174/512 (33.98%), consisting of
101/256 short (39.45%) and 73/256 long (28.52%). The unedited base scored
166/512 (32.42%). This comparison shows an association with prompt length;
it does not isolate length from database or question difficulty.

Pass 27 is a diagnostic prompt-intervention run, not a weight-edit result.
On 128 fixed questions, original F4 got 49 correct. Neutral extra text got
50, schema instructions 41, planning instructions 51, and expert instructions
51. None establishes a gain. These prompts have `probe_only: true`; the
harness excludes them from default random sampling and requires explicit IDs
to run them. Two regression tests verify this separation.

Pass 28 captures four MLP activation layers over 16 full contexts. Pass 29
captures 480 controlled prefix states across 32 questions and five instruction
conditions. Analytic sparse rank-one edits steer toward the mean instruction
state, after removing the neutral-prompt direction. Passes 30–33 test expert
or planning directions at layers 15 or 19 with the original prompts. Their
scores on the balanced 256-question screen are 89, 90, 90, and 91 correct.
Pass 34 reruns exact F4 on that same smaller batch and scores 91/256 (35.55%).
Thus none of these edits beats its matched unchanged-weight control; apparent
small gains over the larger-batch reference are within measured numerical
variation. Both references remain recorded in the reports.

## Additional mechanisms

Schema-copy attribution over cached states identifies L27H8 and several heads
at layers 12, 16, and 24 that retrieve actual identifier tokens from DDL. Planned
head edits strengthen selected retrieval outputs or query selectivity, or
attenuate heads with negative identifier projections. Attribution is a local
diagnostic and does not establish that strengthening a head improves SQL.

Q/K/V counterfactual recombination on the previously traced L22H7 and L23H4
binding heads finds almost all alias-dependent output change in Q. Earlier
experiments amplified only a mean query direction. A covariance-normalized
multi-direction projection is a new hypothesis. An initial 16-design/16-audit
split of the old 32 pairs improves at most one audit binding choice. Pass 35
therefore broadens the geometry to 432 new alias pairs, excluding all 32 old
audit questions. These contrasts use the model's own SQL prefixes and known
schema identifiers, with no fitting to correct SQL outputs.

The same read also captures 160 first-answer instruction states. A separate
family will inject a task vector only at the assistant's first position to
avoid perturbing every SQL token. Original F4 remains the fallback throughout.

## Further measurements through pass 62

Passes 36–45 screen ten traced schema-retrieval head interventions. The best
is L27H8 output gain 1.5: 93/256 versus the matched F4 control's 91/256, an
unconfirmed two-answer gain. Other interventions range from 85 to 92 correct.
Every edited tensor is archived under `candidates/pass_NN/` with its fingerprint.

Pass 46 tests query-style instructions diagnostically. Relative to 49/128
normal-prompt F4 answers, CTEs score 33, fully qualified names 40, explicit
SQL-comment alias plans 36, and meaningful aliases 43. These changed prompts
are not direct-edit results. Pass 47 captures the resulting controlled states.
First-position expert/planning edits (48–51) score 91,89,90,88; analogous
SQL-opening edits (53–56) score 91,89,90,90. None beats the 91-answer control.

Pass 52 examines assigning aliases after choosing a FROM table. On 64 native
prefixes the model always names the first table T1 and the second T2. After
counterfactually naming the first table T2, it chooses T1 for the second in
61 cases and T3 in three, never reusing T2. Wrong first-table assignments are
less confident on average in schema-unambiguous cases. However, four resulting
direct edits (57–60) score 91,82,72,81. Inspection shows changed later join plans
and column choices, not a simple permutation of alias names. Reject them.

Pass 61 supplies two validated synthetic SQL demonstrations as diagnostic
prompts. On the same 128 questions, canonical and permuted examples each score
49, fully qualified examples 46, and alias-map examples 50. No gain is
established. Pass 62 captures clean and deliberately corrupted demonstration
contexts; corrupted demonstrations are used only for read contrasts, never
as scored examples or supervised fitting targets.

The demonstration-state hypothesis is motivated by [task vectors](https://arxiv.org/abs/2310.15916)
and [function vectors](https://arxiv.org/abs/2310.15213). This is an experimental
adaptation to weight edits and SQL, not a replication of either paper's method
or evidence that these techniques improve BIRD.

## Remaining search and confirmation protocol

Finish eight predeclared demonstration-state interventions, then ten coordinated
owner-retrieval/alias-definition and query-geometry trials. The coordinated
alias hypothesis uses a simpler AS-role detector and raises token 2 while
preserving tokens 1 and 3; this avoids changing the native ordering between
unused aliases. It is combined with separately traced owner/copy heads.

After screening, check the top three distinct candidates on `confirm256.txt`,
which is disjoint from the first-stage screen, with a new unchanged F4 control.
Choose any further candidate using practice/confirmation results only. Freeze
it before independent validation. Use the reserved `validate512.txt` and, if
needed, additional disjoint balanced cohorts; do not revise weights from those
validation outcomes. A final dev evaluation is justified only by a candidate
that survives independent checks. Retain exact F4 if none does.

## Demonstration-state interventions (passes 63–70)

Eight fixed task-vector interventions derived from clean/corrupted demonstration
contrasts scored 89, 88, 90, 91, 87, 89, 88, and 88 correct on screen256.
The identical-prompt, identical-cohort F4 control scored 91. No intervention
improved the matched control. These include injection at the SQL opening or
first answer position, layers 15/19/23, and raw or clean-minus-corrupted
directions. Exact parameters and tensor fingerprints are in trials.jsonl.

## Coordinated binding and copying (passes 71–80)

The owner-retrieval head alone scored 89/256. All-AS-definition rules raising
alias digit 2, with or without strengthened owner or copy heads, scored 76, 77,
88, 76, and 79. These are rejected. Preserving the SELECT reference state is
insufficient to preserve the complete table/column plan during later generation.

Stronger L27H8 identifier-copy outputs scored 95/256 (gain 2) and 96/256 (gain 3),
against the matched F4 control of 91/256. The gain-3 candidate fixed 8 and broke
3 (two-sided exact McNemar p=0.226562). Short: +2/128; long: +3/128. These are
exploratory screening results after many trials, not evidence of a dev gain.

Before confirmation outcomes, validation_protocol.json records the acceptance
rules: a positive separate confirmation, then a frozen candidate on four fresh
512-question paired-control folds. Promotion requires pooled net >=10/2048,
at least three positive folds, and exact paired p <=0.05. No validation tuning.

Wide query-contrast ranks 16 and 32 scored 90 and 88/256 in passes 79–80.
Screening is complete: 44 weight candidates. The fixed selector sends parents
78, 77, and 37 to confirmation against a new F4 control, passes 81–84.

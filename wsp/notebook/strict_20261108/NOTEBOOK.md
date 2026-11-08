# Round 3 (strict, pre-registered protocol) -- lab notebook

Start: 84/100 passes used, current edits = F4 (fingerprint ae81f3cb05285df3). Protocol: wsp/protocol/protocol.json
(<= 4 candidates, gate A net >= +8 on cohort A vs F4, accept iff A+B exact McNemar p < 0.05 with fixed > broke).
Cohort ids are never read; they are only used (in pool_rates.py) to EXCLUDE ids from practice statistics.

## Step 0 (free): what can still move under this protocol?

Arithmetic of the protocol. Acceptance needs roughly +12..+18 net over 1,024 questions. Two regimes:
- a broad edit that changes many outputs (round 2 edits changed 40-250 of 512) needs a large true effect, because
  discordant pairs add noise (SD of net ~ sqrt(f+b));
- a *surgical* edit that only fires where a specific failure happens has few discordant pairs, so a small gain
  can still be significant (e.g. 10 fixed / 2 broke gives p = 0.039). Harness noise is low when the edit is
  sparse: pass 25 (F4 rerun) reproduced 512/512 outputs, pass 18 had only 7 discordant pairs.
Round 2 showed that broad schema/alias/steering edits move errors around rather than removing them. So this round
looks for a mechanical failure that (a) is common in the cohort distribution (half of the prompts > 2,048 tokens),
(b) has an execution-verified upper bound, and (c) can be fixed by an edit that is silent everywhere else.

Free analysis of every F4 output already on disk (2,045 unique non-cohort practice questions: 1,789 short, 256 long;
err_taxonomy.py, nsc_long.py):
- short: correct 37.0%, no-such-column 35.3%, wrong rows 20.2%, no such table 2.4%, syntax 'near "AS"' 1.6%.
- long (>2048): correct 28.1%, no-such-column 35.9%, wrong rows 26.2%, syntax 'near "-"' 4.3%, 'near
  "transaction"' 1.6%, other syntax ~3%.
- the syntax errors are almost all **unquoted special identifiers** (show_syntax.py): `FROM Sales Orders AS T1`,
  `JOIN voice-actors AS T2`, `JOIN current-terms AS T2`, `FROM transaction AS T1`. The schema writes these as
  `CREATE TABLE "voice-actors"`, but the comment lines write them bare (`-- voice-actors.\`character\` examples`)
  and the model copies the bare form.
- quote_occ.py, per occurrence in F4 SQL: special TABLE names are quoted only 19/87 times (22%); special COLUMN
  names (spaces) are quoted 103/121 (85%). So the failure is specific to table names after FROM/JOIN.
- truly problematic tables live in 5 train DBs: regional_sales (Sales Orders / Sales Team / Store Locations),
  airline (Air Carriers), disney (voice-actors), legislator (current-terms, historical-terms, social-media),
  beer_factory (transaction, an SQL keyword). (Round 1 parked this because BIRD dev has no such tables; the
  protocol cohorts are train questions, and for dev the edit would simply be inert.)
- prevalence (pool_rates.py, non-cohort pool): gold SQL needs a special identifier in 17.5% of long and 5.5% of
  short questions (special table: 11.2% / 4.6%).
- execution-verified upper bound (quote_cf.py): backtick-quoting the special identifiers in the 85 failed F4
  answers that wrote them bare makes 21 correct (16/1,789 short, 5/256 long). Scaled to a 256 short + 256 long
  cohort: about +2.3 + 5 = ~7 per cohort for a *perfect* fix, assuming no breakage.

Other mechanical classes checked and rejected as too small: non-SQLite functions (YEAR, MULTIPLY, SUBTRACT:
~0.6%), integer division (the model already CASTs to REAL in 156/187 ratio questions).
The big classes (no-such-column 35%: alias swaps 14%, owner not joined / bare column 15%, invented 4%; wrong rows
20-26%) were the target of ~60 round-1/2 edits, none of which held up; I do not retry them.

Decision: candidate 1 = **conditional table quoting**: when the model is about to write a special table name after
FROM/JOIN, raise the token " `" (id 1565). Expected effect is modest (upper bound ~+7 per cohort, so passing gate A
(+8) is far from certain), but it is mechanical, verified by execution, and can be made silent elsewhere.

## Pass 85 (read, current edits = F4, 1,024 teacher-forced texts; mlp_act:27, resid:27, logprobs; positions=answer)
What and why: to build a sparse rank-one key in layer-27 down_proj (same construction as F4) that fires at the
FROM/JOIN position *only* when the next table needs quoting, I need L27 MLP activations at (i) FROM/JOIN positions
followed by a special table (positives), (ii) FROM/JOIN positions followed by an ordinary table (hard negatives,
incl. legislator `current` vs `current-terms`), (iii) all other answer positions (leak control), and the final
normed residual to simulate the logit change exactly (L27 is the last layer).
Texts (build_read85.py): non-cohort practice questions from the 5 special-table DBs, answer = F4's own SQL from
earlier passes with special identifiers backtick-quoted when available, else the gold SQL; plus ~250 questions from
other DBs (F4 own SQL) as negatives. Prompts are rendered with the chat template exactly as in gen.
Prediction: P(" `") at positives is low (the model quotes ~22% of the time), and a linear readout on L27 MLP
activations separates special-table FROM positions from ordinary ones on held-out questions (score gap >> leak).

### Pass 85 result (wsp/passes/pass_85; quote_key.py, quote_key2.py)
1,024 texts (73 own-F4 SQL with quoting inserted + 300 gold from the 5 special-table DBs + 651 own-F4 SQL from other
DBs). Labelled answer positions: 433 positives (FROM/JOIN -> ' `' special table), 1,650 hard negatives (FROM/JOIN ->
ordinary table), 56,394 background.
- At positives the model currently gives P(' `') = 0.20 on average; logit(' `') minus the best other token: median
  -1.89 (p10 -4.1, p90 +0.96); ' `' is already argmax at 68/433 (16%, consistent with the 22% quoting rate).
- Once ' `' is written, the model continues correctly: teacher-forced log-prob of the next 4 tokens (table name +
  closing backtick) median -0.03.
- Linear readout (ridge closed form on L27 MLP activations, negatives = background + 20x-weighted hard negatives,
  key normalised so mean positive score = 1; fitted on half of the questions, scored on the other half):
  dense: positives median 0.84 (p10 0.13), hard negatives median 0.01 (p99 0.56), background |score| p99.9 0.70.
  sparse N=64 neurons: positives 0.83 (p10 0.33), hard median 0.03 (p99 0.93), background p99.9 0.97, max 1.24.
  sparse N=128: positives 0.84 (p10 0.30), hard p99 0.81, background p99.9 0.86.
- Leave-one-DB-out (a special table never seen when fitting): positives median 0.41-0.92 vs same-DB hard negatives
  median 0.06-0.26. So part of the readout is a general "a quoted/special identifier comes next" feature, part is
  DB-specific. The protocol cohorts are drawn from the same train DBs, so the in-distribution numbers are the
  relevant ones; for dev the edit would mostly stay silent.

### Design of candidate Q1 (quote_lib.py, quote_sim.py, edit_q1.py)
Exact layer-27 simulation (bf16-rounded weight, pre-norm rms 5.5 assumed as for F4), key fitted on one half of the
questions, scored on the other half (217 positives, 788 hard negatives, 14,550 background positions):
| N cols | delta | positives with ' `' as argmax | hard negatives newly quoted | background argmax changes |
|---|---|---|---|---|
| 64 | 3 | 0.10 -> 0.66 | 21/788 | 6/14,550 |
| 64 | 4 | 0.10 -> 0.74 | 28/788 | 10 |
| 64 | 5 | 0.10 -> 0.79 | 43/788 | 11 |
| 128 | 4 | 0.10 -> 0.73 | 27/788 | 8 |
The background changes are almost all ' Land' / ' County' / ' Product' / ' Sales' -> ' `', i.e. bare multi-word
COLUMN names in regional_sales that now get quoted too (a desired side effect); a few ' T' -> ' `'.
Hard-negative flips quote an ordinary table (`FROM \`Customers\``), which is valid SQL; the one worry is legislator
`current` (21 in-sample flips), where an opening backtick might tempt the model toward `current-terms`.

**Candidate Q1 (frozen before any generation):** F4 + W[:, S] += u k_S^T in model.layers.27.mlp.down_proj.weight,
S = 64 columns (disjoint from F4's 32), k_S = closed-form ridge key on all pass-85 positions, u = least-norm vector
with +4.5 nats on ' `' and 0 on the 400 most frequent answer tokens. 96,429 weights differ from F4 (64 x 1536 minus
bf16 no-ops). In-sample: positive score median 1.01, hard-negative p99 0.93, background |score| p99.9 0.94.
Mechanism: the model knows the table it is about to write but copies the bare spelling from the comment lines; the
edit adds the opening backtick only where the readout says a special table follows; the rest (name + closing
backtick) the model already continues correctly (log-prob -0.03).
**Prediction:** among questions on the 5 special-table DBs, unquoted special tables drop by ~70% and syntax errors
'near "-"/"AS"/"transaction"' drop by a similar fraction; elsewhere outputs stay bit-identical except a few percent
of FROM clauses that get a harmless quoted ordinary table. EX gain: roughly +4..+7 per 512-question cohort
(upper bound ~7 from quote_cf.py), with very few breaks (< 3). Passing gate A (+8) is therefore possible but less
likely than not; I test it because it is the one failure class with an execution-verified fix and a silent edit.

## Pass 86 (gen --which both --vs F4, 512 non-cohort practice ids: ids_pass86.txt) -- mechanism check before gate A
What and why: one practice pass to verify that Q1 behaves as designed under real generation (vLLM, greedy) before
spending a protocol test: 256 random questions from the 5 special-table DBs (largely the questions the key was
fitted on, so this half checks mechanism, not generalisation) + 256 never-generated questions from other DBs (53%
long) to measure collateral changes. Prediction: special half: syntax errors from unquoted tables fall by >= 60%,
net +4..+10; other half: <= 10 changed outputs, net ~0.

### Pass 86 result (Q1 vs F4, 512 non-cohort practice ids; compare.py)
Overall F4 27.15% -> Q1 35.55%, **fixed 46 / broke 3**.
- special-table half (256; mostly the questions the key was fitted on, so mechanism evidence, not generalisation):
  37 -> 81 correct, fixed 45 / broke 1. Unquoted special tables 143 -> 17, quoted 23 -> 162; syntax errors
  106 -> 13. Almost every fix is literally the F4 query with the table backticked (`FROM \`Store Locations\``,
  `JOIN \`current-terms\``, `FROM \`transaction\``); 3 fixes also changed the join plan. The one break (p02730,
  disney) re-ordered the FROM clause and then mis-bound an alias.
- other half (256 never-generated questions from other DBs): fixed 1 / broke 2; 30 outputs changed, only 5 of
  them by a backticked FROM table, the rest are near-tie re-rolls (bf16 logits) like those seen with F4.
- per DB net (n): regional_sales +16 (65), beer_factory +10 (50), airline +8 (35), legislator +7 (63), disney
  +3 (43). Remaining misses: single-table `FROM Sales Orders` (the key fires less without a join context),
  `social-media`, and some `Air Carriers` joins.
The far larger conversion rate than quote_cf.py predicted (45 vs ~25% of syntax failures) is because on these DBs
F4 fails *only* on the quoting in many otherwise-correct queries; quote_cf.py's estimate came from a sample
dominated by other DBs.

Projection for a cohort (pool composition, non-cohort, used only as frequencies): short 6.7% of questions are on
these DBs (airline 80, regional_sales 138, disney 100 of 4,721), long 8.7% (legislator 144, beer_factory 117 of 3,016).
Per-DB net rates from pass 86 give about +3.3 (short half) + 3.4 (long half) = **+6.7 per 512-question cohort**,
minus collateral on the other ~470 questions (net ~0, roughly 50 changed outputs, maybe 5-8 discordant).
So the expected gate-A net is about +5..+7 with a spread of several points: passing the +8 gate is roughly a
coin flip at best. Combined A+B, if both cohorts behave like this (e.g. 16 fixed / 4 broke over 1,024), the exact
McNemar p would be ~0.01.

## Pass 87: PROTOCOL gate A for candidate Q1 (fingerprint 9ced5b00aec22643)
Candidate Q1 exactly as frozen above (no change after pass 86). Pre-registered prediction for cohort A: fixed
~9 (range 5-14), broke ~3 (range 1-6), net +6 (range +1..+11); essentially all fixes on regional_sales / airline /
disney / legislator / beer_factory questions via quoted table names; few changes elsewhere.

### Pass 87 result (gate A, Q1 vs F4 on cohort A)
F4 32.62% -> Q1 34.18%; **fixed 10 / broke 2, net +8** -> passes gate A (exactly at the threshold). In line with
the prediction (fixed ~9, broke ~3, net +6, range +1..+11). Other summary counts: prediction errors 236 -> 223,
no such table 9 -> 8; no-such-column 195 -> 203 (queries that used to stop at a syntax error now run far enough to
hit a column error). Only the tool summary was looked at; the per-question cohort outputs were not opened.
Decision: run gate B with the identical candidate (no change of any kind).

## Pass 88: PROTOCOL cohort B for candidate Q1 (fingerprint 9ced5b00aec22643)
Prediction (same reasoning as for A): fixed ~9, broke ~3, net ~+6. Acceptance needs the A+B total to reach exact
McNemar p < 0.05: with A = 10/2, B needs roughly net >= +3 with few breaks (e.g. B 6/2 -> 16/4, p = 0.012; B 4/3 ->
14/5, p = 0.064 fails).

### Pass 88 result (cohort B, Q1 vs F4)
F4 31.25% -> Q1 33.20%; **fixed 15 / broke 5, net +10** (prediction: ~9 / ~3, net ~+6).
Prediction errors 231 -> 216, no such table 5 -> 3, no-such-column 192 -> 199 (as in A: queries that used to die
at a syntax error now run far enough to hit a column error).

## Protocol decision: Q1 ACCEPTED
A + B (1,024 held-out questions): fixed 25 / broke 7, net +18 (+1.76 points: F4 31.93% -> Q1 33.69%).
Exact McNemar p = 2 * sum_{i<=7} C(32, i) / 2^32 = **0.0021** < 0.05, fixed > broke. Both cohorts were positive
(+8, +10), and both matched the pre-registered mechanism-based prediction (+6 per cohort) within its stated range.
Only one candidate was tested (1 of 4 allowed); no retests, no changes between pass 86 and the protocol tests.

Honest reading:
- The gain is real but narrow. It comes from one mechanical failure: on the 5 train DBs with table names that need
  quoting (spaces, hyphens, the keyword `transaction`), the model copies the bare spelling from the schema comment
  lines and the query dies with a syntax error. The edit adds the opening backtick at FROM/JOIN only when a
  linear readout of the layer-27 MLP says such a table comes next; the model completes the rest itself.
- It works where round-1/2 edits did not because it repairs a *spelling* decision, not a plan: the query the model
  intended is unchanged, so the fix does not move errors elsewhere. The edit is sparse (64 columns), so outside
  those positions outputs change only by bf16 near-tie re-rolls (net ~0 on the other-DB half of pass 86).
- Expected effect on BIRD dev is small: round 1 noted dev has no multi-word table names, so the edit should mostly
  stay silent there (the leave-one-DB-out test showed only partial transfer of the readout to unseen special tables).
  I am not claiming a dev gain; dev is evaluated by the harness owner, at most once.

## Final
Current edits = Q1 = F4 + quoting rank-one update, both in model.layers.27.mlp.down_proj.weight
(fingerprint 9ced5b00aec22643, snapshot Q1; F4 snapshot kept). Passes used this round: 4 (85 read, 86 practice gen,
87 gate A, 88 gate B); total 88/100. Submitted with the note in submit_note.txt.

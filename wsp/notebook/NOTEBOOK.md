# Weight-space programming lab notebook: Qwen2.5-Coder-1.5B-Instruct on BIRD text-to-SQL

Goal: raise execution accuracy (EX) on BIRD by directly editing weights (no gradients).
Budget: 25 forward passes (gen/read). Model: 28 layers, d=1536, MLP 8960, 12 q heads / 2 kv heads, head_dim 128.

Harness facts (from reading src/t2s/wsp/tool.py, prompt.py, bird_eval.py):
- gen: greedy, max 512 new tokens, vLLM; extract_sql takes the LAST ```sql block, else any ``` block, else raw text.
  So prose before the SQL is not fatal by itself; it hurts only if the SQL is cut off at 512 tokens, or the
  model writes several blocks/variants, or prose leaks into the parsed SQL.
- read: can capture several layers in ONE pass (e.g. resid:10,resid:14,...). With practice ids it teacher-forces
  prompt + gold "```sql\n...\n```<|im_end|>"; positions=answer starts at the last prompt token (P-1), whose
  next-token log-prob is log p("```" | prompt) = the "answer directly" propensity.
- Qwen2 has q/k/v biases. A change to v_proj.bias adds a CONSTANT vector o_proj·Δb to the residual at every
  position (attention weights sum to 1), so it is a clean "steering vector as a weight edit" if needed.

## Plan
1. Pass 1: baseline gen on 512 practice questions (seed 0), categorise failures.
2. Decide on a mechanism (format/verbosity direction, schema-linking failure) and locate it with read passes.
3. Minimal edit, paired test on hundreds of questions, fresh seed for final check.

---
## Pass 1 (baseline gen, original, 512 questions, seed 0)
Why: need a failure taxonomy before forming any hypothesis. 512 is the max per pass; the original outputs are cached,
so later `--which both` runs on the same ids are paired for free on the original side.
Result (wsp/passes/pass_01): EX 38.7% (198/512). 205 prediction errors: 159-160 "no such column", 18 'near "AS"'
syntax, 11 no such table. 106 run but return wrong rows. mean 202 chars; 96.9% start directly with ```sql
(only 16 verbose answers, 4 of them correct) -> verbosity is NOT the main problem on this sample.

"no such column" taxonomy (wsp/notebook/analyze_nsc.py):
- alias_swap 67: column qualified with T2 but its table is T1 in the FROM clause (40 T2->T1, 25 T1->T2, 2 other).
  e.g. `SELECT T2.games_name FROM games AS T1 JOIN games_competitor AS T2` .
- owner_not_joined 37: qualified column whose table is not in the query at all.
- unqualified_owner_not_in_query 34: bare column that lives in a table not in FROM (missing join).
- invented 22: column exists nowhere (Northwind-like DB with absent columns etc.).
Alias style: the model uses T1/T2 aliases in 321/334 multi-table queries; 66/321 (20%) of those are alias swaps.
So the single largest, most mechanical failure is *alias binding*: the alias written in SELECT (before FROM exists)
is inconsistent with the order in which the tables are later listed in FROM.

Next (pass 2): teacher-forced read of the model's OWN outputs (prompt + raw answer) for the 512 baseline questions,
capturing log-probs and the residual stream at layers 6,10,14,18,21,24,27 at all answer positions. resid:27 plus
model.norm + tied unembedding gives the full next-token distribution (logit lens, CPU only), so I can measure the
margin at the alias digit after "SELECT T" and at the first table after "FROM" in swap vs correct cases, and look for a
linear direction that predicts the right alias better than the model's output does.

## Pass 2 (read, original, own outputs of 512 questions; logprobs + resid 6,10,14,18,21,24,27 at answer positions)
CAVEAT found while analysing: in transformers 4.56 `hidden_states[28]` (= the tool's resid:27) is ALREADY passed
through the final RMSNorm, so the logit lens for resid:27 is simply resid27 @ E^T (verified: reproduces the stored
log-probs to ~0.02 nats). resid:L for L<27 are raw (pre-norm) residuals.
Findings (wsp/notebook/lens_alias.py, lens_from.py, lens_select.py):
1. Alias digit after "SELECT T" is a coin flip: P(chosen digit) ~0.51-0.55 in ALL classes (ok, swap, wrong).
   The model cannot know the FROM order yet; consistency must be enforced later.
2. At the first table after FROM the model does condition on the written alias, but weakly:
   P(owner-of-selected-column listed first) = 0.78 when SELECT used T1, 0.39 when SELECT used T2
   (ideal: ~1.0 and ~0.0). Logit model: owner-first logit ~ -0.45 + 1.7*[alias==T1]. -> binding is too weak.
3. P(" T" | "SELECT") = 0.55; alternatives are DISTINCT (0.08) and diffuse letters/column/table names.
4. Format: first answer token is "```" with mean P=0.62, "Let" 0.25, "Here" 0.13 (margin ```-vs-{Let,Here} = 0.53
   +- 0.41 nats). Layer-wise lens: mid layers (18-24) strongly favour "```" (lens margin 6-9 nats); the LAST layers
   (25-27) push toward prose and bring it down to ~0.5. So the "explain first" behaviour is written late, at one
   fixed position (the "\n" after "<|im_start|>assistant"), and is near a tipping point (11% of prompts have
   P(```)<0.5). On dev-like prompts this may tip more often -> a cheap, low-risk target.

## Pass 3 (read, original, 788 texts: 300 own outputs + 244 alias counterfactual pairs*)
Why: (a) mlp_act at layers 25,26,27 on the format texts (all answer positions, incl. P-1) to find which late MLP
neurons write the "Let/Here" push at P-1 and how selective they are (to design a rank-one, position-selective edit);
(b) *for each 2-table T-alias output, the prefix up to " FROM" with the SELECT aliases as written ("own") and with
T1<->T2 swapped ("swap"); the logit lens at the FROM position for both variants measures how much the written alias
moves the choice of first table, and resid at 12..27 shows in which layers that alias->table binding is computed.

### Pass 3 results
(a) Format (wsp/notebook/format_dla.py): at the first answer position the lens margin ```-vs-{Let,Here} is +11.7 at
layer 23 and is pulled down to +0.5 by the MLPs of layers 25-27. The push is concentrated in a handful of
*position-selective* neurons: L27 n3317 (DLA -18.7 of -20.4 for the whole L27 MLP; act -18.4 at P-1 vs mean |0.73|
elsewhere, selectivity 25x), L27 n7449 (-5.4), L26 n6853 (-6.7, selectivity 44x), L25 n5798 (-7.7, selectivity 62x).
These are "start of the assistant reply -> begin with prose" neurons.
Simulation (format_sim.py, exact for a layer-27 edit up to the unknown pre-norm rms 4.5-6): removing the
(```-vs-Let/Here) component from the single column L27 n3317 raises P(```) at P-1 from 0.62 to 0.97-0.99 and makes
``` the argmax on 300/300 texts.
(b) Alias binding (wsp/notebook/alias_cf.py, 150 own/swap pairs): effect of the written alias (T1 vs T2) on the
owner-first logit at the FROM position, by layer: L12 0.01, L15 0.04, L17 0.13, L19 0.21, L21 0.88, **L23 3.06**,
L25 2.50, L26 2.95, **L27 1.82**. So the binding "alias -> which table goes first" is computed in layers 22-23
(97% of pairs in the right direction), and then partly *undone* by layer 27 (and 24-25). Final consistency:
alias T1 -> owner listed first 81%; alias T2 -> other table first only 53%. There is a ~+1 logit prior to list the
owner table of the selected column first, regardless of the alias.

## Edit F1 (format): L27 down_proj column 3317, remove component along dh = unit(g*(E[```]-(E[Let]+E[Here])/2))
1534 weights changed (0.011% of the 1.54B params; 1 column). Script: wsp/notebook/edit_format.py.
## Pass 4 (gen both, 512, seed 0) -- test F1 alone
Expectation: the ~16 verbose answers become SQL-first; almost nothing else changes (neuron is quiet elsewhere).

### Pass 4 result (F1 alone vs original, 512 seed 0)
EX 38.67% -> 38.48%; fixed 5 / broke 6 (noise). sql_first 96.9% -> 100%, mean chars 202 -> 166, errors identical
(205, 160 no-such-column). 58 outputs changed: all 16 verbose ones (3 fixed, 2 broke among them) + 42 SQL-first
outputs perturbed slightly (butterfly effects of a changed P-1 residual), net -2 there.
Verdict: F1 surgically removes the "explain first" behaviour (exactly as the DLA predicted) but is accuracy-neutral on
this sample. Keep as a candidate component (cheaper answers; may matter more on prompts where the model tips to prose).

Side check (pass-2 data): digit margins logit(1)-logit(2) at alias positions: before FROM mean 0.17 (71% |m|<1, coin
flip); after FROM median |m| 7, but 5% below 1.5 and 10% below 2.6 -> a blanket "prefer T1" bias is NOT safe.

## Pass 5 (read, original, positions=all, 32 own/swap pairs = 64 texts, resid:20,21,22,23)
Why: the alias->first-table binding appears in layers 22-23. With the residual at ALL positions entering layers 21-23
I can recompute those layers' attention exactly on CPU (input_layernorm, q/k/v + biases, RoPE theta=1e6, GQA, causal
softmax) and get per-head outputs at the FROM position; projecting onto g*(E[owner]-E[other]) gives per-head
direct logit attribution and, differenced between the T1 and T2 variants, which heads carry the binding.

### Pass 5 results (head-level attribution of alias binding; wsp/notebook/attnlib.py, alias_heads.py, head_pattern.py)
Layer recomputation from captured residuals is exact to 0.3%. Digit effect (owner-first logit, T1 minus T2 variant)
by component: L22H7 +0.88 (97% of pairs same sign), L23H4 +0.51 (100%), L23 MLP +0.43, L22 MLP +0.31, L23H3 +0.23.
L22H7 and L23H4 are "table-name mover" heads: at FROM they put 93-99% of their attention on schema tokens naming
tables ('CREATE TABLE "torrents"', 'references torrents'); the alias digit only modulates which table.
Crucial asymmetry (pass-3 lens, 150 pairs, per layer): owner-first logit with alias T1: L23 +2.88 -> final +1.88
(81% consistent); with alias T2: L23 -0.18 -> final +0.06 (53% consistent = coin flip). The model has a working rule
"T1 -> list the selected column's table first" but essentially NO rule for T2. Amplifying the heads (simulated
alpha=2-3 scaling or 1.5-2x attention sharpening, alias_sim.py) moves the L23 effect only 3.2 -> 3.7-4.2, mostly on
the T1 side. So the better lever is upstream: since the alias digit in SELECT is a coin flip (margin 0.13), make the
model write T1 for the first alias, which lands it in the regime where its binding works.

## Edit A1: rank-one "first SELECT alias -> T1" rule in layer-27 down_proj (wsp/notebook/edit_t1.py, t1_key*.py)
Key k: ROME-style closed form k = (C + 0.01*mean(diag C) I)^-1 mu_first, k.mu_first=1, mu_first = mean L27 MLP act at the
first pre-FROM alias-digit position (173 positions), C = uncentered covariance at 19.5k other answer positions (incl.
later pre-FROM aliases and post-FROM aliases). Held-out split: positives score 0.91 (p10 0.72), other positions
p99 0.07, post-FROM aliases p99 0.10, later pre-FROM aliases mean 0.03. (Sparse 16-128 neuron keys were much less
selective, so the edit is a dense rank-one update.)
Value u = 3.5*5.5*v/|v|^2, v = g*(E['1']-E['2']): +3.5 nats for '1' over '2' at score 1 (assumed pre-norm rms 5.5).
Layer 27 is the last layer, so this edit only changes next-token logits, nothing downstream.
Simulation on 19.7k answer positions (sim_l27.py): 82 argmax changes, 78 of them '2'->'1' right after "SELECT T" /
"SELECT DISTINCT T" / "SELECT SUM(T"; 2 '1'->'2', 2 others. After bf16 rounding realised effect: first-alias +3.5,
later pre-FROM +0.15, post-FROM +0.01 (p99 0.25), other +0.00.
F1 removed for this test (to measure A1 alone).
## Pass 6 (gen both, 512, seed 0) -- A1 alone

### Pass 6 result (A1 alone vs original, 512 seed 0): NOT kept
EX 38.67% -> 37.89%, fixed 23 / broke 27. The edit did exactly what it was designed to do (first alias T2: 102 -> 1),
but alias swaps only went 67 -> 63: 14 swap->ok, 16 ok->swap. Lesson: the alias digit is a low-margin readout of a
*sticky* FROM-order plan (digit agrees with the eventual order 75%); forcing T1 just moves errors around because the
FROM order does not re-plan. A held-out LDA probe at the digit position (plan_probe.py) predicts the eventual order no
better than the model's own digit (0.61-0.69 vs 0.75), so no latent plan to exploit. A1 removed.
Also checked and parked: (i) q-amplification of L22H7/L23H4 along the mean digit direction: no effect (digit_dir.py);
(ii) unquoted multi-word identifiers: 30/38 special TABLE names unquoted (copied from the prompt's
'-- Store Locations.`col`' comment lines) but columns mostly quoted (29/36); BIRD dev has no multi-word table names,
so not a generic target; (iii) wrong-row results are semantic (55 single-value differences), no single mechanism;
(iv) verbose answers concentrate in long prompts (half >= 1855 tokens; practice prompts are capped at 2048, dev
goes to 3072), which supports keeping F1 for the held-out set.

## Pass 7 (read, original, positions=all, 24 own/swap pairs + 16 single-table FROM prefixes; resid:23,24,25,26)
Why: with T2 aliases, layers 24-26 push the owner table back toward first place (natural-T2 owner-first logit
-1.23 at L23 -> -0.35 at L26). Decompose layers 24-27 per head at the FROM position to find who adds this
digit-independent "owner first" prior, and see what the same components do in single-table queries (where listing
the owner table is exactly right).

### Pass 7 results (late-layer decomposition at FROM; late_heads.py, l27_binding_neurons.py)
Layers 24-27 are a tug of war that depends on the example's *plan*, not on the written digit: L24H10 carries the
plan (+0.9 owner-first in examples the model naturally wrote with T1, -1.0/-1.2 in natural-T2 examples, whatever
digit is shown); L26H11 pushes the owner table (+3.1 in natural-T2 examples, +2.0 toward the right table in
single-table queries); L27H1 pushes against it. The L27 MLP is anti-binding (digit effect -1.54 over 150 pairs) but
the effect is spread thinly over thousands of neurons (top neuron 0.06, sign-consistent in only ~50% of pairs).
Direct-path ablation estimates for single heads (L26H11, L27H1, L24H10, L26H9, L27 MLP) all trade T1 consistency
for T2 consistency; none is a net win. Also: wrong columns are written with high confidence (P~0.92 at the first
token of the bad column in every class), i.e. these are confident schema-planning errors, not an uncertainty mode.
A1 removed after pass 6.

## Combined test A1 + B1 (pass 8)
Idea: make the convention consistent by construction: first SELECT alias = T1 (A1), and strengthen the only binding
rule the model has, "T1 -> list the owner table first", by amplifying its two carriers L22H7 and L23H4 (o_proj
head columns x2; simulated direct-path effect at L23: T1 owner-first logit 2.33 -> 3.34). In pass 6, forced-T1 cases
whose plan put the owner second were the source of the 16 ok->swap breaks; B1 targets exactly those.

### Pass 8 result (A1 + B1 vs original, 512 seed 0): NOT kept
EX 38.67% -> 38.09%, fixed 29 / broke 32. alias_swap 67 -> 49 (the binding did get better), but
unqualified_owner_not_in_query 34 -> 40, invented 22 -> 28; JOINs 334 -> 305, subqueries 105 -> 118; 247 outputs
changed. x2 head amplification is too blunt: it changes query style broadly. Removed.
Noise floor reminder: even F1 (one column) changed 58 outputs; discordant pairs ~60 => SD(fixed-broke) ~ 8, so
+-10 is not distinguishable on 512 questions.

Key observation for the held-out set: verbosity depends steeply on prompt length. Pass 1: 7/231 verbose below 1000
tokens, 1/131 at 1000-1500, 1/123 at 1500-1800, but 7/27 (26%) at 1800-2048 (practice is capped at 2048; the
held-out prompts can reach 3072). On those 27 long prompts, original 8 correct vs F1 12 correct.
## Pass 9 (gen both, F1 only, the 512 LONGEST unseen practice prompts, >= 1665 tokens; ids_long512.txt)
Tests F1 in the regime where the "explain first" failure actually happens (and where held-out prompts live).

### Pass 9 result (F1 vs original on the 512 longest unseen prompts, 1665-2048 tokens)
original EX 38.09%, verbose 6.5% (33/512), mean 257 chars, 213 no-such-column. F1: EX 38.09% (fixed 9 / broke 9),
sql_first 100%, mean 184 chars. Split: the 33 originally-verbose questions 8 -> 12 correct (+4; 10 of the 33 had
run into the 512-token limit); the 479 SQL-first ones 187 -> 183 (-4) from 37 perturbed outputs.
Diagnosis of the -4 (sim_l27.py on pass-3 activations, exact for an L27 edit): neuron 3317 is not silent at
mid-SQL line starts; removing its component also raises ``` there, so in 7/300 outputs a line start such as
"... = '2012'\n" now prefers ``` over GROUP/AND/ORDER, i.e. the code block closes before the query is finished.
Pass 4 + 9 combined: verbose 12 -> 17 correct (+5 of 49); SQL-first -6 (premature closes + noise).

## Edit F2 (replaces F1): rank-one, position-selective version (wsp/notebook/edit_f2.py, f2_key.py)
W27_down += u k^T, k = ridge closed form (C + 0.01 mean diag C I)^-1 mu_{P-1} (k.mu=1) from the L27 MLP activation at the
first answer position vs the 19.4k other answer positions; held-out score 1.00 at P-1 (min 0.93), |score| elsewhere
p99.9 0.08, max 0.13. u = +5 nats for ``` vs mean(Let, Here) (rms 5.5 assumed). Simulation: the only argmax changes in
19.7k positions are the 8 'Let'->'```' flips at P-1 (+1 harmless '\n'->' FROM' line-break flip). No premature closes.
## Pass 10 (gen both, F2, same 512 long prompts as pass 9) -- direct comparison with F1 on the same ids

### Pass 10 result (F2 vs original, same 512 long prompts)
EX 38.09% -> 37.89% (fixed 7 / broke 8); sql_first 100%, 184 chars. Verbose subset 8 -> 11 /33; SQL-first subset
187 -> 183 with 25 changed outputs. Inspecting the divergences: p00081 and p01692 still close the code block early at
a mid-SQL line start, *identically* under F1 and F2 -> systematic, not noise: on long prompts the key leaks a little
onto "\n" line-start positions (the P-1 token is also "\n"), and F2's value raised ```, which competes there with
ORDER/GROUP/AND. The remaining changes are near-tie flips (T1/T2, LIKE/=, ';' before ```).
## Edit F3 (replaces F2): same key, value that only SUPPRESSES prose openers (wsp/notebook/edit_f3.py)
u = least-norm solution of: logit change -5 nats for 21 prose openers (Let, Here, To, Below, Given, The, Sure, I,
First, We, This, In, Based, Certainly, You, It, Step, ###, **, Okay, A) and exactly 0 for the 400 most frequent
output tokens (incl. ```, ORDER, GROUP, AND, ';', '\n', T, digits...). So wherever the key leaks, it can only lower
prose tokens that are irrelevant inside SQL; it can never make ``` more attractive. At P-1 (pass-3 activations):
P(```) 0.62 -> 0.99 for assumed rms 4.5-7; only argmax changes in 19.7k positions are the 8 P-1 flips (+1 inside an
already-verbose answer).
## Pass 11 (gen both, F3, same 512 long prompts)

### Pass 11 result (F3 vs original, same 512 long prompts)
EX 38.09% -> 37.30% (fixed 8 / broke 12). Premature closes gone (1 vs F1's 13), verbose subset 8 -> 11, but the
SQL-first subset 187 -> 180 with 41 changed outputs, and the decisive divergences are mostly the coin-flip alias digit
after "SELECT T" going 1 -> 2 (5 of 6). Direct-logit simulation says F3 changes logit(1)-logit(2) there by < 0.002, so
this is not a direct effect. Explanation: vLLM computes logits in bf16 (resolution ~0.06-0.12 at |logit|~20); the
alias digit is a near-tie (71% of margins < 1), many exact bf16 ties resolve to the lower id '1'; ANY perturbation of
the final hidden state re-rolls those ties, which looks like a systematic 1 -> 2 drift. A dense rank-one update
(F2/F3: 6-11M bf16 weights changed by rounding) perturbs every position slightly. Remedy: make the edit sparse so
that the output is bit-identical wherever the edited neurons are quiet, and keep |u| small.
## Edit F4 (replaces F3): 32-column sparse key + value that only lowers 'Let'/'Here' (wsp/notebook/edit_f4.py)
All 49 verbose practice answers (passes 1+9) begin with "Let's". u = least-norm with -5 nats on Let/Here and exactly 0
on the 400 most frequent output tokens (|u| 15.6 vs 44.9 for F3). k restricted to 32 neurons, ridge re-solved on the
subset (held-out: P-1 score ~1.0, leak elsewhere p99.9 0.19, max 0.39 -> at most ~2 nats less 'Let' inside SQL,
harmless). 32 x 1536 = 49k weights. P(```) at P-1 0.62 -> 0.99 on pass-3 activations.
## Pass 12 (gen both, F4, same 512 long prompts)

### Pass 12 result (F4 vs original, same 512 long prompts)
EX 38.09% -> 37.50% (fixed 8 / broke 11); verbose subset 8 -> 12; SQL-first subset 187 -> 180, 27 changed outputs.
Several SQL-first divergences are IDENTICAL under F3 and F4 (p00125, p03561, p03530, p03569: same token, same
direction) even though the two edits share neither key sparsity nor value, and the simulated direct logit effect at
those positions is ~0. That smells like a harness-level difference between how the cached original and an edited
model are run (numerics / scheduling), not an effect of the edit.
## Pass 13 (gen both, NULL edit, same 512 long prompts) -- measure the noise floor of a paired comparison
Null edit: +1 bf16 ulp on ONE weight of the quietest L27 neuron (mean |act| tiny) -> numerically invisible.
If this also shows ~25 changed SQL-first outputs with the same flips, the comparisons above are partly artefact.

### Pass 13 result (NULL edit vs original, same 512 long prompts): the harness noise floor
A numerically invisible edit still changed 15 SQL-first outputs (EX 38.09 -> 37.70, fixed 0 / broke 2), so a
cached-original vs edited-run comparison carries ~15 "free" divergences (bf16-logit ties re-rolled by run-level
numerics), and two independent edited runs differ by ~30. Relative to the NULL run, F1..F4 change 21-40 SQL-first
outputs (F4: 27) with fixed/broke 9/10 overall -> the SQL-first side of every F-edit is consistent with noise.
The verbose side is consistent across all four format edits: verbose-subset correct orig 8, NULL 8, F1 12, F2 11,
F3 11, F4 12 (/33). Pooled with pass 4: 49 verbose questions, 12 -> 16-17 correct.
Conclusion so far: the format edit gives a real but small gain where the failure occurs (~+10 points on the
verbose questions), invisible in total EX on practice because only 3-6% of practice prompts trigger it; held-out
prompts are longer (up to 3072 tokens) and the verbose rate climbs steeply with length (26% at 1800-2048).
F4 is the cleanest version (49k weights, no premature closes, suppress-only value) -> current candidate.
## Pass 14 (gen both, F4, seed 0, 512 moderate-length questions; compare with F1 in pass 4)

### Pass 14 result (F4 vs original, seed 0, 512)
EX 38.67% -> 39.26% (fixed 10 / broke 7); sql_first 96.9% -> 100%; 202 -> 167 chars; no-such-column 160 -> 157.
Verbose subset 4 -> 6 (/16), SQL-first subset 194 -> 195. Pooled F4 over 1024 questions: 393 -> 393 overall,
verbose 12 -> 18 (/49), SQL-first 381 -> 375 (noise floor ~ +-5 per 512).

Re-analysis of A1 (pass 6) by the ORIGINAL first-digit margin m = logit(1)-logit(2) of the flipped cases
(alias_rows.json + pass 6/8 outputs): m in [-0.25,0): 11 -> 15 correct (n=35); [-0.5,-0.25): 4 -> 3 (n=20);
m < -0.5: 14 -> 6 (n=25). (A1+B1: 11->12, 3->3, 13->10.) So forcing T1 helps only for genuine coin flips and hurts
when the model clearly wanted T2 (its sticky plan put the owner table second). Consistent with the mechanism: T1 is
where the robust binding rule lives, but a strong T2 preference reflects a real plan.
## Edit A2 (+F4 in the same tensor): small, sparse first-alias bias (wsp/notebook/edit_a2.py)
Same idea as A1 but +0.4 nats instead of +3.5 (re-decides only |m| < ~0.4), key restricted to 128 neurons (ridge
re-solved), value least-norm with +0.2/-0.2 on '1'/'2' and 0 on the 400 most frequent tokens. Leak: later pre-FROM
aliases <= 0.31 nats, post-FROM <= 0.27 nats (their margins have median 7). 128 + 32 columns of L27 down_proj
(203k weights incl. F4).
## Pass 15 (gen both, F4+A2, seed 0) -- compare with F4 alone (pass 14) on the same ids
Result (pass 15, F4+A2 vs original, seed 0): EX 38.67% -> 39.65% (fixed 22 / broke 17); vs F4 alone 201 -> 203
(16/14). First alias T2 96 -> 31; the 53 flipped 2->1 questions went 15 -> 19 correct. CAVEAT: A2's threshold was
chosen from the A1 run on this same seed-0 sample, so this is not independent evidence.
## Pass 16 (gen both, F4+A2, the 512 long prompts, disjoint from seed 0) -- independent replication of A2
Compare with original and with F4 alone (pass 12) on the same ids.
Result (pass 16, F4+A2, long prompts, independent of seed 0): EX 38.09% -> 38.67% (fixed 22 / broke 19); vs F4 alone
(pass 12) 192 -> 198 (22/16). no-such-column 213 -> 200. First alias T2 97 -> 30; the 68 flipped questions:
original 21, F4-alone 20, F4+A2 24 correct; the other 444: 174 -> 174. Verbose subset 8 -> 11.
A2 replicates in direction on fresh questions (+3 to +4 on the flipped subset). Pooled flips: 36 -> 43 of 121.
Candidate for submission: F4 + A2 (both in model.layers.27.mlp.down_proj.weight).
## Pass 17 (gen both, F4+A2, FRESH seed 2026, 512) -- final check on unseen sample
Result (pass 17, F4+A2, fresh seed 2026; 52/46 overlap with seed-0/long sets): EX 36.13% -> 34.96%
(fixed 12 / broke 18). no-such-column 169 -> 159, sql_first 97.1% -> 100%, 201 -> 170 chars. Flipped 2->1 subset
(67): 18 -> 16; rest (445): 167 -> 163; verbose subset (15): 4 -> 4.
A2 does NOT replicate on the fresh seed. Pooled over 3 samples the A2-flipped subsets are 54 -> 59 of 188 (+5),
overall F4+A2 vs original +5, +3, -6. Everything is inside the harness noise (~+-6 per 512 paired).
## Passes 18-20: decide between F4 alone and F4+A2 with more data
18: F4 alone, seed 2026 (same ids as 17) ; 19: F4+A2, seed 7 ; 20: F4 alone, seed 7.
Result (pass 18, F4 alone, seed 2026): EX 36.13% -> 36.33% (fixed 4 / broke 3; only 7 discordant -- this run had
almost no harness noise). F4 alone beats F4+A2 on the same ids by 17 fixed / 10 broke.
A2 vs F4 pooled: seed 0 +2, long +6, fresh 2026 -7 => +1 over 1536 questions. A2 is dropped (no robust evidence;
its seed-0 support was partly circular). Pooled F4 vs original: +3, -3, +1.
## Pass 19 (gen both, F4 alone, fresh seed 7) -- second fresh check of the submission candidate
Result (pass 19, F4 alone, fresh seed 7): EX 35.35% -> 36.13% (fixed 8 / broke 4); sql_first 97.5% -> 100%;
203 -> 171 chars; no-such-column 196 -> 192; verbose subset 2 -> 5 (/13); rest 179 -> 180.

# FINAL DECISION: submit F4 alone (passes used: 19 of 25)
F4 = model.layers.27.mlp.down_proj.weight, 32 output columns (48,398 weights, 0.0031% of the model): designed as the
rank-one update W[:,S] += u k_S^T (bf16 rounding makes the stored delta numerically full-rank on those columns).
F4 vs original, all paired runs (4 samples, 2048 question-runs, ~10% overlap):
  seed 0 (512):   38.67 -> 39.26  (fixed 10 / broke 7)
  long (512):     38.09 -> 37.50  (8 / 11)     [prompts 1665-2048 tokens]
  seed 2026 (512):36.13 -> 36.33  (4 / 3)      [fresh]
  seed 7 (512):   35.35 -> 36.13  (8 / 4)      [fresh]
  total: 26 fixed / 25 broke (+1 net with seed-0/long/2026/7 = +3,-3,+1,+4 => +5 of 2048 counting by EX)
  questions that originally answered with prose first: 18 -> 27 correct of 77 (+9), the rest -4 (noise floor of the
  harness measured with a null edit: ~15 changed outputs / +-2-6 net per 512).
  sql_first: 93.6-97.5% -> 100% in every run; mean answer length -15% to -28%.
Honest summary: the edit removes the "explain at length first" failure mode completely and surgically (one fixed
position, suppress-only value, no premature code-block closes), helps the affected questions, and is accuracy-neutral
on the rest of the practice data. Gains on the held-out set should scale with how often long prompts trigger the
prose opening (7/27 = 26% of practice prompts at 1800-2048 tokens; held-out prompts reach 3072).
The dominant error ("no such column", ~31-42%) was diagnosed (alias binding: alias written before FROM is a
coin flip read off a sticky FROM-order plan; T2 has no working binding rule; late layers fight over it) but none of
the edits that moved it (A1, A1+B1, A2) improved EX robustly, so none is submitted.
Submitted F4 (fingerprint ae81f3cb05285df3). The first submit call's note was mangled by shell backtick
substitution, so the identical edit was re-submitted with the note read from wsp/notebook/submit_note.txt.


# User-requested continuation: passes 20–25

Kept direct weight edits and no training. See `continuation_20261107/README.md`
for the full audit, scripts, controls, and paired practice results. Four new
candidates scored 161, 197, 193, and 186 correct on the same 512 questions.
A final rerun of original F4 scored 201 and reproduced all 512 pass-14 outputs
exactly. All four candidates are rejected. The exact original F4 fingerprint
`ae81f3cb05285df3` is restored; the original submission is archived and retained
in ledger submission history. Used the full 25-pass budget. No new dev
evaluation, no gradients, no optimizer, no training, no other methods' weights,
and no changes to prompts or SQL extraction. The previously measured dev score
remains 30.12%; the 40–50% target was not achieved.


# User-requested relaxed-budget experiment: passes 26–84

The user authorized more time while keeping direct weight edits and no training.
Raised the total cap to 100 and ran 59 additional harness passes, including
44 weight-candidate screens, activation reads, and diagnostic prompt probes.
Added 3,528 non-truncated long-context BIRD train questions to the practice pool;
normal edit prompts and SQL grading remain unchanged. Diagnostic prompt probes
are explicitly marked and excluded from random sampling.

Best screening edits strengthen L27H8 identifier copying: 96, 95, and 93 /256
versus the matched F4 91/256. Separate confirmation gives 81, 83, and 88 /256
versus F4 88/256. No candidate advances to validation. Restored exact original
F4 fingerprint ae81f3cb05285df3; no new dev run, and the previously measured
30.12% remains the accepted score. Target 40–50% not achieved. No gradients,
optimizer, training, fine-tuned checkpoints, or SQL repair.

Full provenance: relaxed_20261107/README.md, decision.json, trials.jsonl,
audit.json, tensor archives, and the preserved ledger/submission history.

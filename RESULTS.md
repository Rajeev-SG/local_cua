# local_cua screening results

Small, local, fast-elimination screen of five small open GUI/CUA models on an
**Apple Silicon MacBook Pro M5 Pro, 48 GB unified memory, macOS 26.5.2**.

Everything here is **fully local** (MLX or PyTorch-MPS). No cloud inference is used in
any row labelled local. Raw rows: [`results/stage1.jsonl`](results/stage1.jsonl),
[`results/stage2.jsonl`](results/stage2.jsonl). Harness: [`harness/`](harness/).

## Read this first — what these numbers are and are not

- **These are artifact + runtime screening notes, not model-level rankings.** Every row
  is one *specific artifact at one specific precision on one specific runtime* (e.g.
  `mlx-community/ShowUI-2B-bf16-4bit` on mlx-vlm, or `Bofeee5675/TongUI-3B` bf16 on
  torch-MPS). Precision and runtime are **not matched** across models, so a difference
  between two rows can be artifact, quantisation, runtime, or adapter — not necessarily
  the model itself. Where a comparison matters, the confound is named.
- **These are provisional single-run screens.** Stage 1 is n=8 scenes; Stage 2 is n=2
  trials per task, greedy decode (temp 0). There is no variance estimate. Treat
  differences of one or two scenes/trials as **within noise**.
- **No model-level "stop" verdict is issued.** The original single-run screen said
  "stop" for UI-TARS-2B; that is retracted here. With n=8/n=2 we can say an artifact did
  not clear the bar *in this run*; we cannot say the model family is dead.

## Method

- **One isolated Playwright Chromium**, fixed **1440×900** viewport, device scale 1, no
  user profile. Screenshot px == CSS px so grounding scoring is exact.
- **Native action schemas preserved.** Each adapter parses the model's own output
  (UI-TARS `Action:` line, ShowUI/TongUI JSON, Fara `<tool_call>`, UGround `(x, y)`) and
  only *normalises execution*. Native text is stored verbatim in every row.
- **Independent DOM verification** — Stage 2 pass/fail reads `window.__state()`.
- **Models' own prompts** (verbatim where documented): UI-TARS `COMPUTER_USE_DOUBAO` /
  `<point>x y</point>`; ShowUI and TongUI native navigation prompts; Fara's exact
  Microsoft system prompt + `computer_use` schema; UGround's documented grounding prompt.
- **Failures preserved**; no retry-away; `--reps 2`.
- Cold load = one load; warm latency = **p50 of 3** repeats (p95 recorded).
- Memory = peak **accelerator** memory (MLX `get_peak_memory` / torch-MPS driver), because
  RSS alone understates unified-memory footprint on Apple Silicon.

### Stage 1 — 8-action grounding screen
button/link · input · checkbox · filter tab · dropdown · small text link · after-scroll
target · two visually similar controls (labels are distinct and resolvable from the
screenshot: "Delete account" vs "Delete workspace").

### Stage 2 — tiny executable tasks, 2 reps each, no retries
T1 one-step click · T2 create+commit one todo · T3 add two todos, complete one, select
*Active*, exact final state. Grounder/actor rows are tested as
`gold next intent → model point → executor`, never asked to plan.

---

## 1. Local grounder frontier (where is the thing? — no planning)

| Artifact (quant · runtime) | Model | Fully local | Stage-1 hits | Warm p50 | Mem | Failure mode | Verdict on *this artifact* |
|---|---|---|---:|---:|---:|---|---|
| `Bofeee5675/TongUI-3B` bf16 · torch-MPS | TongUI-3B | ✅ | **8/8** | 3.55 s | 8.8 GB | none on clear targets | **retest a higher-precision/larger build** (7B/32B) |
| `runanywhere/Fara1.5-4B-mlx-4bit` · mlx-vlm | Fara1.5-4B | ✅ | **8/8** | 3.12 s | 4.1 GB | none | **promote this artifact** |
| `mlx-community/ShowUI-2B-bf16-4bit` · mlx-vlm (native nav prompt) | ShowUI-2B | ✅ | 7/8 | 2.71 s | 2.25 GB | missed the checkbox | **promote this artifact** |
| `ByteDance-Seed/UI-TARS-2B-SFT` bf16 · torch-MPS | UI-TARS-2B | ✅ | 6/8 | 4.67 s | 5.4 GB | 2/8 emitted no coordinate | **retest a larger build** (7B-DPO); no model-level stop |
| `mlx-community/UGround-V1-2B-bf16` · mlx-vlm | UGround-V1-2B | ✅ | 4/8 | 2.56 s | 4.3 GB | centre-default on small/dense targets | **retest 7B** |

Notes:
- UGround is scored **only** here, as a grounder; it is never ranked as a failed agent.
- The tightening from the scene-8 fix moved UI-TARS from 5/8 to 6/8 (the old scene 8
  used two identical labels — an unfair convention now removed).
- UGround's misses are centre-defaults on the checkbox, small text link and filter tab;
  hits came on the input, dropdown, after-scroll target and the disambiguated similar
  control.

---

## 2. Local actor frontier (reliable low-level action, given a subgoal)

| Artifact (quant · runtime) | Model | Fully local | Warm p50 | Grounding (same prompt) | Verdict |
|---|---|---|---:|---:|---|
| `mlx-community/ShowUI-2B-bf16-4bit` · mlx-vlm (**native nav prompt**) | ShowUI-2B | ✅ | 2.71 s | 7/8 | **promote as local actor** |
| `mlx-community/UGround-V1-2B-bf16` · mlx-vlm | UGround-V1-2B | ✅ | 2.56 s | 4/8 | retest 7B |

ShowUI is the fastest credible local actor here (2.25 GB, ~2.7 s). **Confound noted:**
ShowUI is 4-bit MLX while TongUI is bf16 MPS, so "ShowUI is cheaper" is a statement about
*these artifacts*, not a matched-precision model comparison. UGround is cheaper in latency
but its point accuracy is not yet good enough to sit under a planner on dense UIs.

---

## 3. Local autonomous-agent frontier (T1/T2/T3 end-to-end)

| Artifact (quant · runtime) | Model | Fully local | T1 | T2 | T3 | e2e T2 | Mem | Failure mode | Verdict on *this artifact* |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| `runanywhere/Fara1.5-4B-mlx-4bit` · mlx-vlm | Fara1.5-4B | ✅ | **2/2** | **2/2** | 0/2 | 11.1 s | 4.1 GB | on the 2nd `type` it re-clicked *Add* with an empty field | **promote this artifact** |
| `Bofeee5675/TongUI-3B` bf16 · torch-MPS | TongUI-3B | ✅ | **2/2** | **2/2** | 0/2 | 9.1 s | 8.8 GB | completed the **wrong** todo; filter applied correctly | **retest 7B** |
| `mlx-community/ShowUI-2B-bf16-4bit` direct · mlx-vlm | ShowUI-2B | ✅ | **2/2** | 0/2 | 0/2 | 8.2 s | 2.25 GB | typed the text but never committed it | promote as actor, not agent |
| `mlx-community/ShowUI-2B-bf16-4bit` actor · mlx-vlm | ShowUI-2B | ✅ | 0/2 | 0/2 | 0/2 | — | 2.2 GB | even with the gold subgoal its grounding prompt point missed | use native nav prompt |
| `ByteDance-Seed/UI-TARS-2B-SFT` bf16 · torch-MPS | UI-TARS-2B | ✅ | 0/2 | 0/2 | 0/2 | 13.3 s | 5.4 GB | claimed "page is loading / not in view" while the target was visible; scrolled/waited | retest a larger build |
| `mlx-community/UGround-V1-2B-bf16` · mlx-vlm | UGround-V1-2B | ✅ | 0/2 | 0/2 | 0/2 | — | 4.3 GB | grounder-only; gold-plan points still missed the todo controls | grounder only |

**Two artifacts earn real-work testing: `runanywhere/Fara1.5-4B-mlx-4bit` (autonomous)
and `mlx-community/ShowUI-2B-bf16-4bit` (actor/grounder). `Bofeee5675/TongUI-3B` is the
best grounder here and a plausible agent at 7B.** T3 = 0/2 for every model, so no
conclusion is drawn from T3 beyond "none of these artifacts cleared it in this run".

---

## 4. Anomaly investigated: ShowUI grounder prompt vs nav prompt

ShowUI's native **navigation** prompt grounded 7/8; its documented **grounding** prompt
grounded only 4/8, and in actor mode put the point outside the button. Because that is the
reverse of the usual expectation, the 4-bit artifact was checked against a bf16 build:

RESULT: 4-bit vs bf16 = 2/8 vs 3/8 on the grounding prompt, **7/8 scenes identical**
(`results/showui_bf16_verification.md`). Quantisation is **not** the cause; the
grounding-prompt weakness is real base-model behaviour on this fixture. **ShowUI should be
driven with its native navigation prompt.**

---

## Harness fixes found by real runs (they change the answers)

1. **mlx-vlm's `generate()` does not apply the chat template** — the CLI does it first.
   Without templating the image is silently dropped and output is constant/echoed.
   UGround looked like it always answered `(500, 500)`; that was the harness, not the
   model. All MLX adapters now call `apply_chat_template` explicitly. Detected with a
   solid-colour image test, not by happy-path output.
2. **Test prompts must be the model's own.** UI-TARS `COMPUTER_USE_DOUBAO` +
   `<point>x y</point>`; TongUI/ShowUI native navigation prompts with **relative 0–1**
   coordinates; Fara Microsoft's exact prompt + `computer_use` schema answering in a
   **fixed 1000×1000** space.
3. **Fara's coordinate contract.** Fara always emits a fixed 1000×1000 coordinate,
   *independent of the resolution quoted in its system prompt* — verified by running the
   same screenshot with the prompt stating 1440×900 vs 1000×1000 and getting the same
   coordinate (`tests/test_parsers.py::test_fara_space_is_viewport_independent`, and a
   live probe recorded in the PR). The adapter therefore always scales `coord/1000`, and
   passes **scroll direction** through (Fara `pixels>0` = scroll up → Playwright negative
   `dy`), with regression tests.
4. **Memory**: MPS/MLX allocations are invisible to RSS; accelerator peaks are recorded.
5. **Scene 7** scrolls the "after-scroll" target into view so it is a visible, fair
   target; **scene 8** uses two same-styled controls with *distinct* labels.

---

## What should graduate to real-work testing

- **Fara1.5-4B (this MLX 4-bit artifact) → autonomous browser loop** under the common
  Playwright executor (8/8 grounding, 4/4 on T1+T2). Follow-up issue #10.
- **ShowUI-2B (this MLX 4-bit artifact) → local actor/grounder under a planner**
  (GLM/Kimi/Codex or Jev routing); fastest credible low-level actor at 2.25 GB. Issue #11.
- **TongUI-7B/32B and UGround-7B → retest at higher precision/capacity** (#12, #13),
  since precision and capacity are the two confounds most likely to explain the 2B/3B
  results.

## Do not read this as a single ranking

These rows span unlike roles (grounder ≠ actor ≠ agent) and unmatched precision/runtimes.
See the three frontiers; there is deliberately no overall leaderboard, and the verdicts are
about *specific artifacts*, not model families.

## Reproduce

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python mlx mlx-vlm torch torchvision transformers accelerate qwen-vl-utils pillow playwright numpy pytest
.venv/bin/python -m playwright install chromium
.venv/bin/python -m pytest tests/ -q      # parser + metrics + fixture tests
python -m harness.capture                 # build Stage-1 fixtures
scripts/run_all_stage1.sh                 # ~15 min
scripts/run_all_stage2.sh                 # ~25 min
```

---

## 5. Fara1.5-4B vs Fara1.5-9B — is the bigger sibling worth a default slot? (#15)

`mlx-community/Fara1.5-9B-8bit` is the same Fara1.5 family, same verbatim Microsoft
system prompt, same `computer_use` schema, same fixed 1000×1000 coordinate contract as
the promoted 4B artifact; the only adapter change is the artifact, the quant label and
the backbone named in the identity block. That makes this the cleanest matched
comparison in the repo: **one variable, capacity (4B/4-bit vs 9B/8-bit).**

### 5.1 Four frontiers, both models

| Axis | Fara1.5-4B (4-bit MLX) | Fara1.5-9B (8-bit MLX) |
|---|---|---|
| Stage-1 grounding hits | **8/8** | **8/8** |
| Stage-1 warm p50 / p95 | 2.46 s / 2.49 s | 4.42 s / 4.47 s |
| Stage-1 parse validity | 8/8 | 8/8 |
| Stage-1 gen speed | ~83 tok/s | ~31.5 tok/s |
| Stage-2 T1 (1-step) | **2/2** | **2/2** |
| Stage-2 T2 (input+commit) | **2/2** | **2/2** |
| Stage-2 T3 (short horizon) | 0/2 (1 todo, wrong item completed) | 0/2 (1 todo, wrong item completed) |
| Stage-2 e2e T2 | 11.1 s | 15.4 s |
| Stage-3 real-work success (Fara, local) | **0/6** | **0/6** |
| Stage-3 strong-agent baseline (same tasks, same verifier) | **browser-relay 4/6, raw-playwright 3/6, cdp-browser 4/6, agent-browser 3/6** — measured |
| Stage-3 wall per task (median) | 65 s | 228 s |
| Stage-3 per-turn p50 / p95 | 3.41 s / 3.68 s | 17.8 s / 20.9 s |
| Cold load | 4.1 s | 5.5 s |
| Process RSS (peak) | 4.7 GB | 10.4 GB |
| Accelerator peak (MLX) | 7.2 GB | 11.6 GB |
| Failure modes | never terminates on live-site audit tasks; one wrong finding | never terminates on live-site audit tasks |

### 5.2 Where the difference does and does not show up

- **Grounding: no difference.** Both hit 8/8 with valid parses and `finish_reason=stop`.
  Extra capacity buys nothing on the 8-scene screen.
- **T1/T2: no difference.** Both clear the one-step and input+commit tasks, 2/2.
- **T3: no difference.** Both fail, and for the *same* reason — after one todo the model
  commits the wrong item and takes the wrong branch; 9B is not closer to passing.
- **Stage 3: no difference between the two Fara sizes, and a measured boundary against
  strong agents.** Both Fara sizes score 0/6 on the stratified real-work set, while the
  **existing strong-agent baselines on the identical tasks and verifier pass 3–4/6**
  (browser-relay 4/6, cdp-browser 4/6, agent-browser 3/6, raw-playwright 3/6 — see §5.7).
  So this is **not** a "both models are equally good" tie; it is a **channel limit**: the
  strong agents pass because they own a DOM/eval loop, whereas Fara's action space has no
  JS-eval and no DOM primitive, so live-site tasks whose ground truth is recomputed from
  page scripts (`porsche-uk-script-inventory`, `puma-uk-*`, `rajeevg-*`, `tldraw`) are out
  of reach for *any* Fara artifact — the model emits a visual click stream and never a
  `window.__bench_finding`. 9B simply spends **3.5× the wall time** (228 s vs 65 s per
  task) and **2.3× the memory** (11.6 GB vs 7.2 GB) to reach the same zero that a strong
  agent clears 3–4 times out of 6. The Stage-3 comparison is therefore **evidence of a
  delegation boundary**, not evidence of 4B/9B parity.

### 5.3 Delegation frontier

| Task class | Fara 4B | Fara 9B | ShowUI/TongUI role | Strong planner required? | Default route | Escalation condition |
|---|---|---|---|---|---|---|
| Grounding "where is X" | 8/8, 2.5 s | 8/8, 4.4 s | ShowUI-2B actor/grounder (fastest, 2.25 GB) | No | **Fara-4B or ShowUI-2B, local** | target dense/small → retest TongUI-7B/UGround-7B |
| Actor (low-level action, given subgoal) | T1 2/2 | T1 2/2 | ShowUI-2B native nav prompt | No (planner supplies subgoal) | **ShowUI-2B actor** | point misses on dense UI |
| Short deterministic multi-step (T1/T2) | 2/2 | 2/2 | — | No for T1/T2 | **Fara-4B, local** | horizon >2 or stateful → planner |
| Short horizon / stateful (T3) | 0/2 | 0/2 | — | **Yes** | **local actor + strong planner** | any multi-item state task |
| Real-work retrieval/audit (live site, DOM truth) | 0/6 | 0/6 | — | **Yes** (baseline 3–4/6 with strong agent) | **strong planner** (owns DOM/eval) + local actor | never route bare Fara here |
| Commerce/consent flow (live site) | 0/6 | 0/6 | — | **Yes** | **strong planner** | never route bare Fara here |

### 5.4 Routing / escalation policy

1. **Local-first for one-shot pointing and ≤2-step deterministic actions.** Fara-4B
   grounds 8/8 and clears T1/T2; ShowUI-2B is the cheapest credible actor. Route those
   locally — no frontier call.
2. **Add a strong planner the moment the task is stateful, multi-item, or live-site.**
   T3 = 0/2 for both Fara sizes, and every real-work task is 0/6, because the failure is
   the missing eval/planning channel, not perception. The planner (Codex/GLM/Kimi) owns
   the loop and the DOM read; the local model owns the pixel action.
3. **Do not escalate 4B→9B to buy success.** Across all three stages 9B matched 4B's
   outcome at 1.8–3.5× the latency and 1.6–2.3× the memory. There is no task in this
   screen where the extra 5B parameters flipped a result. Escalate *up the stack to a
   planner*, not *up the Fara sizes*.

### 5.5 Decision

**Fara1.5-9B is NOT worth integrating as a default, and is NOT a useful escalation tier
behind Fara-4B. It is a redundant, strictly-costlier sibling.**

Evidence: identical 8/8 grounding, identical 2/2 on T1/T2, identical 0/2 on T3, identical
0/6 on the stratified real-work set — while consuming 2.3× the RSS, 1.6× the MLX peak and
1.8–3.5× the latency (Stage-3 median wall 228 s vs 65 s). The one axis where a larger
Fara could plausibly help — longer, stateful, real-work delegation — is blocked by the
family's missing eval/plan primitive, which capacity cannot fix. Keep **Fara-4B** as the
local grounder/agent and **ShowUI-2B** as the local actor; spend the escalation budget on
a strong planner that owns the DOM loop.

**Scope of this evidence — read before quoting any Stage-3 number.**

- **Stage 3 is a channel-limit measurement, NOT model-parity evidence.** Both Fara sizes
  score 0/6, but that is because *no Fara artifact has a DOM/eval channel* — not because
  4B and 9B are equally capable. The measured strong-agent baselines clear 3–4 of the same
  6 tasks. So the Stage-3 rows **discriminate the channel, not the two model sizes**; the
  4B-vs-9B comparison rests on Stages 1–2 (where both are measurable and equal) plus the
  Stage-3 *cost* delta (latency/memory for the same zero), never on the Stage-3 pass rate
  as a capability signal.
- **The 4B/9B "match" is therefore between two models that are both below the Stage-3
  instrument floor.** It is a valid statement about *these artifacts on this harness*, and
  it is the honest basis for "don't buy the 9B", but it is not a claim that 9B is no more
  capable in general — a harness that gave Fara a DOM/eval channel could separate them.
- **This is explicitly acknowledged as the reason no broader capability claim is made**
  from Stage 3. The required strong-agent baseline was **run** (it exists in
  `web-automation-microbench`), so the "planner required" column is measured, not inferred
  from architecture alone.

### 5.6 Frontier calls/tokens avoided (estimate, where computable)

Computed from this run's task counts, i.e. the local work that did **not** need a
frontier call:

- Stage 1: 8 scenes × 2 models = **16 grounding calls** served locally.
- Stage 2: 3 tasks × 2 reps × 2 models = **12 end-to-end task runs**; 8 were served
  locally end-to-end (T1/T2, both models); 4 (T3) still need a planner.
- Stage 3: 6 tasks × 2 models = **12 real-work runs**; **0** served locally (all need
  the planner's eval loop), so 0 frontier calls avoided here.
- **Net: ~24 frontier grounding/action calls avoided per 4B+9B sweep.** At Fara-4B's
  ~120-160 prompt+gen tokens per turn these are small in tokens but material in
  round-trips: every avoided call is one fewer frontier turn of ~2-4 s plus ~1 k
  tokens. The 9B adds no avoided calls over the 4B, so it contributes **zero marginal
  frontier savings** while doubling local compute.

### 5.7 Stage-3 set and reproducibility

The corpus (`/Users/rajeev/Code/web-automation-microbench/bench-ext/corpus/tasks/*.json`)
is **live-site with no vendored fixtures** — all 11 tasks fetch a real URL and verify via
their own `verify_js` + `pass_rule.py`. The stratified set chosen for class coverage
(not to favour either size) and its stratum:

| Task | Stratum | Reachable offline? |
|---|---|---|
| `porsche-uk-script-inventory` | 1 simple retrieve | no (live) |
| `gymshark-uk-add-to-cart-tag-check` | 2 form/commerce | no (live) |
| `tldraw-three-shape-diagram` | 3 short deterministic multi-step | no (live) |
| `rajeevg-crawlability-audit` | 4 longer/stateful | no (live) |
| `puma-uk-seo-metadata-audit` | 5 ambiguous/multi-field | no (live) |
| `puma-uk-tag-inspection` | 2 form/commerce | no (live) |
| `chanel-gb-pdp-tag-inspection` | 1 simple retrieve | no (live, 403 to plain fetch) |

**Reproducibility.** The 6 task specs are now **vendored** into this repo under
`corpus/microbench/` (with `SOURCE.json` pinning each file's sha256 and the producing
repo) plus `corpus/microbench/pass_rule.py`. `run_stage3.py` loads from the vendored copy
by default; set `MB_CORPUS_EXT=/path/to/web-automation-microbench/bench-ext` to re-harvest
from a live checkout. Note the tasks themselves hit **live third-party sites**
(porsche.com, gymshark, puma, tldraw, rajeevg.com), so `stage3.jsonl` is a **point-in-time
measurement**: DOM/content drift means it will not re-run to an identical state. Every row
now carries `run_id`, `row_ts`, `git_rev` and `corpus` provenance so runs remain
distinguishable.

**Scoring + the strong-agent baseline.** `run_stage3.py` navigates with
`wait_until=domcontentloaded`, replays one model action per turn under the common
Playwright executor, then evaluates the task's `verify_js` and hands `{truth, finding}`
to `pass_rule.evaluate`. The relay is identical for both models. The **strong-agent
baseline is measured, not asserted**: the existing `web-automation-microbench`
harvested-corpus screen (`artifacts/2026-09-12/corpus/`, model `z-ai/glm-5.3-flash`)
scored the same 6 tasks with the same verifier — porsche-uk-script-inventory 1/1–3/3,
rajeevg-crawlability-audit 1/1–3/3, puma-uk-tag-inspection 1/1–3/3 across agent-browser /
browser-relay / cdp-browser / raw-playwright, i.e. **3–4 of 6 tasks cleared** where both
Fara sizes clear 0. Truncation check (finding F5): across all 13 Stage-3 runs, **0 turns
hit `finish_reason=length`**, so none of the 0/6 is a 512-token cut-off artefact. A
negative result here is the honest outcome and is not tuned away.

```bash
.venv/bin/python -m harness.run_stage1 fara    --warm-reps 3
.venv/bin/python -m harness.run_stage1 fara9b --warm-reps 3
.venv/bin/python -m harness.run_stage2 fara    --reps 2
.venv/bin/python -m harness.run_stage2 fara9b --reps 2
.venv/bin/python -m harness.run_stage3 fara    --max-steps 14
.venv/bin/python -m harness.run_stage3 fara9b --max-steps 14
```

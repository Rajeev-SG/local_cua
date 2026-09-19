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

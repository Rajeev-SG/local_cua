# local_cua screening results

Small, local, fast-elimination screen of five small open GUI/CUA models on an
**Apple Silicon MacBook Pro M5 Pro, 48 GB unified memory, macOS 26.5.2**.

Everything here is **fully local** (MLX or PyTorch-MPS). No cloud inference is
used in any row labelled local. Raw rows: [`results/stage1.jsonl`](results/stage1.jsonl)
and [`results/stage2.jsonl`](results/stage2.jsonl). Harness: [`harness/`](harness/).

## Method (what makes rows comparable)

- **One isolated Playwright Chromium**, fixed **1440×900** viewport, device scale 1,
  no user profile/session. Screenshot px == CSS px, so grounding scoring is exact.
- **Native action schemas preserved.** Each adapter parses the model's own output
  (UI-TARS `Action:` line, ShowUI/TongUI JSON, Fara `<tool_call>`, UGround `(x, y)`)
  and only *normalises execution*. Native text is stored verbatim in every row.
- **Independent DOM verification** — Stage 2 pass/fail reads the app's own state
  (`window.__state()`), not the model's claim.
- **Test prompts are the models' own.** UI-TARS uses the verbatim
  `COMPUTER_USE_DOUBAO` template; ShowUI/TongUI use their own grounding + navigation
  prompts; Fara uses Microsoft's exact system prompt + `computer_use` tool schema.
  Getting these wrong is the single biggest source of fake failures — see
  *Harness fixes* below.
- **Failures are preserved.** No retry-away; `--reps 2` everywhere.
- Cold load = one load; warm latency = **p50 of 3** repeats (p95 also recorded).
- Memory = peak **accelerator** memory (MLX `get_peak_memory` / torch-MPS driver),
  because RSS alone badly understates unified-memory footprint on Apple Silicon.

### Stage 1 — 8-action grounding screen (1440×900, warm p50)
Scenes: button · input · checkbox · filter tab · dropdown · small text link ·
after-scroll target · two visually identical controls.

### Stage 2 — tiny executable tasks (2 reps each, no retries)
- **T1** one-step: click the named *Open settings* control.
- **T2** input: create + commit one todo.
- **T3** short horizon: add two todos, complete one, select *Active*, end in the
  exact expected state.

Grounder/actor rows are tested as `gold next intent → model point → executor`,
never asked to plan.

---

## 1. Local grounder frontier (where is the thing? — no planning)

| Model | Artifact / quant | Fully local | Stage-1 hits | Warm p50 | Mem | Failure mode | Best role | Verdict |
|---|---|---|---|---|---|---:|---|---|---|
| **TongUI-3B** | `Bofeee5675/TongUI-3B` bf16 (MPS) | ✅ | **8/8** | 3.86 s | 8.8 GB | none on clear targets | grounder/actor | **retest stronger variant** (TongUI-7B/32B) |
| **Fara1.5-4B** | `runanywhere/Fara1.5-4B-mlx-4bit` | ✅ | **8/8** | 3.13 s | 4.1 GB | none | grounder **and** agent | **promote** |
| **ShowUI-2B direct** | `mlx-community/ShowUI-2B-bf16-4bit` | ✅ | 7/8 | 2.57 s | 2.25 GB | missed the visually-duplicated control | local actor/grounder | **promote** |
| **UI-TARS-2B-SFT** | `ByteDance-Seed/UI-TARS-2B-SFT` bf16 (MPS) | ✅ | 5/8 | 4.83 s | 5.4 GB | 3/8 hallucinated "page is loading / not in view" | agent, weak here | **stop** (retest 7B-DPO) |
| **UGround-V1-2B** | `mlx-community/UGround-V1-2B-bf16` | ✅ | 4/8 | 2.56 s | 4.3 GB | falls back to image centre on small/dense targets | pure grounder | **retest stronger variant** |
| **ShowUI-2B actor** | `mlx-community/ShowUI-2B-bf16-4bit` | ✅ | 4/8 | 2.39 s | 2.2 GB | grounding prompt performs *worse* than nav prompt | — | see §4 |

Notes:
- UGround is scored **only** here, as a grounder — it is never ranked as a failed
  autonomous agent.
- UGround's 4/8 is real but not fatal: hits came on the dropdown, input, after-scroll
  target and the *second* similar control; misses were centre-defaults on the small
  text link, the checkbox and the filter tab. It answers in the model's native
  `(x, y)` 0–1000 space (converted to viewport px).

---

## 2. Local actor frontier (reliable low-level action, given a subgoal)

The planner supplies the next *intent*; the model only produces the low-level action.

| Model | Artifact / quant | Fully local | Warm p50 | Grounding (same prompt) | Verdict |
|---|---|---|---|---:|---|
| **ShowUI-2B (native nav prompt)** | `mlx-community/ShowUI-2B-bf16-4bit` | ✅ | 2.57 s | 7/8 | **promote as local actor** |
| UGround-V1-2B | `mlx-community/UGround-V1-2B-bf16` | ✅ | 2.56 s | 4/8 | retest stronger variant |

ShowUI is the fastest credible local actor (2.2 GB, ~2.6 s). It is competitive with
TongUI on latency at a third of the memory. UGround is cheaper in latency but its
point accuracy is not yet good enough to sit under a planner on dense UIs.

---

## 3. Local autonomous-agent frontier (T1/T2/T3 end-to-end)

| Model | Artifact / quant | Fully local | T1 | T2 | T3 | Steps (T1/T2/T3) | e2e T2 | Mem | Failure mode | Verdict |
|---|---|---|---|---:|---:|---:|---|---:|---:|---|---|
| **Fara1.5-4B** | `runanywhere/Fara1.5-4B-mlx-4bit` | ✅ | **2/2** | **2/2** | 0/2 | 1/3/8 | 11.1 s | 4.1 GB | on the 2nd `type` it re-clicked *Add* with an empty field, so only one todo existed | **promote** |
| **TongUI-3B** | `Bofeee5675/TongUI-3B` bf16 (MPS) | ✅ | **2/2** | **2/2** | 0/2 | 1/3/8 | 9.1 s | 8.8 GB | completed the **wrong** todo ("Buy milk" instead of "Walk dog"); correct filter applied | **retest 7B** |
| ShowUI-2B direct | `mlx-community/ShowUI-2B-bf16-4bit` | ✅ | **2/2** | 0/2 | 0/2 | 1/3/8 | 8.2 s | 2.25 GB | typed the text but never committed it (clicked *Add* without the input landing) | promote as actor, not agent |
| ShowUI-2B actor | `mlx-community/ShowUI-2B-bf16-4bit` | ✅ | 0/2 | 0/2 | 0/2 | — | — | 2.2 GB | even with the gold subgoal, its grounding prompt put the point outside the button | use native nav prompt |
| UI-TARS-2B-SFT | `ByteDance-Seed/UI-TARS-2B-SFT` bf16 (MPS) | ✅ | 0/2 | 0/2 | 0/2 | 1/3/8 | 13.3 s | 5.4 GB | hallucinated "page is loading / not in view" while the target was visible; scrolled/waited instead of clicking | **stop** |
| UGround-V1-2B | `mlx-community/UGround-V1-2B-bf16` | ✅ | 0/2 | 0/2 | 0/2 | — | — | 4.3 GB | grounder-only; with a gold plan its points still missed the todo-page controls | grounder only |

**Two models clearly earn real-work testing: Fara1.5-4B (autonomous) and ShowUI-2B
(actor/grounder). TongUI-3B is the best grounder and a plausible autonomous agent at
7B. UI-TARS-2B is not usable at this size.**

---

## 4. Anomaly investigated: ShowUI grounder prompt vs nav prompt

ShowUI's *native navigation* prompt grounded **7/8**, but its *documented grounding*
prompt (`Based on the screenshot ... scaled from 0 to 1`) grounded only **4/8**, and on
the actor task put the point outside the button. This is the reverse of the usual
"grounding prompt is better" expectation, so the 4-bit artifact was checked against a
higher-fidelity bf16 build before drawing any conclusion about ShowUI-as-actor.

RESULT: 4-bit and bf16 are equivalent (2/8 vs 3/8, 7/8 scenes identical) — the quant
is **not** the cause; the grounding-prompt weakness is real base-model behaviour. Full
log: [`results/showui_bf16_verification.md`](results/showui_bf16_verification.md).
**ShowUI should be driven with its native navigation prompt, not the grounding prompt.**

---

## Harness fixes found by real runs (they change the answers)

1. **mlx-vlm's `generate()` does not apply the chat template** — the CLI does it
   first. Calling `generate(..., image=...)` without templating silently drops the
   image, and the model returns constant/echoed text. UGround looked like it always
   answered `(500, 500)` and ShowUI always `[0.9, 0.29]`; both were the harness, not
   the models. All MLX adapters now call `apply_chat_template` explicitly.
   Detected with a solid-colour image test, not by happy-path output.
2. **Test prompts must be the model's own.** UI-TARS needs
   `COMPUTER_USE_DOUBAO` and `<point>x y</point>`; TongUI/ShowUI need their own
   navigation prompts with **relative 0–1** coordinates; Fara needs Microsoft's exact
   system prompt + `computer_use` schema and answers in a **1000×1000** space.
3. **UI-TARS coordinate space** is 0–1000, not viewport px; converted accordingly.
4. **Memory**: MPS/MLX allocations are invisible to RSS; accelerator peaks are recorded.
5. **Scene 7** scrolls the "after-scroll" target into view so it is a fair, visible
   grounding target rather than an off-screen impossible one.

---

## What should graduate to real-work testing

- **Fara1.5-4B → autonomous browser loop** under the common Playwright executor
  (strongest vision-only CUA here; 8/8 grounding, 4/4 on T1+T2). Follow-up issue filed.
- **ShowUI-2B → local actor/grounder under a planner** (GLM/Kimi/Codex or Jev routing);
  fastest credible low-level actor at 2.2 GB. Follow-up issue filed.
- **TongUI-7B (and 32B if useful) → grounder/agent retest**, prompted by the 8/8
  grounding at 3B and the 7B/32B ScreenSpot step-up. Follow-up issue filed.
- **UGround-V1-7B → grounder retest**, replacing the 2B which centre-defaults on dense
  UIs. Follow-up issue filed.
- **Stop**: UI-TARS-2B-SFT at this size (hallucinated page-state failures).

## Do not read this as a single ranking

These rows span unlike roles — a grounder is not a worse agent than an agent, it is a
different component. See the three frontiers above; there is deliberately no overall
leaderboard.

## Reproduce

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python mlx mlx-vlm torch torchvision transformers accelerate qwen-vl-utils pillow playwright numpy
.venv/bin/python -m playwright install chromium
python -m harness.capture            # build Stage-1 fixtures
scripts/run_all_stage1.sh            # ~15 min
scripts/run_all_stage2.sh            # ~25 min
```

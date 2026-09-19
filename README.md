# local_cua

Fast local screening of small open computer-use / GUI models on Apple Silicon.

**Status: the screen has been run. Results are in [RESULTS.md](RESULTS.md).** Raw rows
are in [`results/stage1.jsonl`](results/stage1.jsonl) and
[`results/stage2.jsonl`](results/stage2.jsonl). The programme definition (method,
scoring rules, and what each model is) is kept below under
[Programme definition](#programme-definition).

---

# What we did, in plain English

I wanted to know one practical thing: **can small open "computer use" models run on this
Mac, fast enough that they are worth using, or are they a toy?**

So I built a small test rig and ran five of them, entirely offline, on the M5 Pro MacBook
Pro (48 GB). No cloud calls.

## How the test works

Every one of these models works the same way: you show it a **screenshot** plus an
instruction, and it answers with **one action** ("click at (450, 320)"). It never touches
the Mac itself. So the rig does three things:

1. Opens one **isolated Chromium window** at a fixed 1440×900 size (never your real
   browser), so every model sees the exact same picture.
2. Shows the model the screenshot and records what it says.
3. Checks the answer against the page's real, known element positions — the browser
   itself tells us exactly where the button is, so "did it point at the right thing?" is
   measured, not guessed.

Each model was driven with **its own official prompt and answer format**. This matters a
lot: several of these models look broken if you use the wrong prompt. Getting this right
was most of the work (see [Things that tripped us up](#things-that-tripped-us-up)).

## The two tests

**Test 1 — "can you point at the right thing?" (grounding).**
Eight everyday screen elements: a button, a text box, a checkbox, a tab, a dropdown, a
small link, something you have to scroll to, and two look-alike buttons. One question
each: where is it? Score = did the point land on the element.

**Test 2 — "can you actually do a small job?" (tiny tasks).**
Three small jobs done twice each, checked by reading the page's own state, not the
model's say-so:

- **T1** — click the *Open settings* button.
- **T2** — type a todo and save it.
- **T3** — add two todos, tick one off, switch to the *Active* filter, end in the right state.

Failures were kept, never retried away.

## What we found

| Model | What it is | Grounding | T1 | T2 | T3 |
|---|---|---:|---:|---:|---:|
| **Fara1.5-4B** | all-in-one browser agent | **8 / 8** | ✅✅ | ✅✅ | ✗✗ |
| **TongUI-3B** | all-in-one GUI agent | **8 / 8** | ✅✅ | ✅✅ | ✗✗ |
| **ShowUI-2B** | fast clicker / pointer | 7 / 8 | ✅✅ | ✗✗ | ✗✗ |
| **UI-TARS-2B-SFT** | all-in-one GUI agent (older) | 6 / 8 | ✗✗ | ✗✗ | ✗✗ |
| **UGround-V1-2B** | pointer only, cannot plan | 4 / 8 | – | – | – |

The honest summary:

- **Two models are genuinely useful: Fara1.5-4B and ShowUI-2B.** Both point at the right
  thing almost every time, and Fara can also finish a two-step job on its own.
- **TongUI-3B is the best "where is it?" model** (perfect score) but the heaviest to run,
  and it made a wrong choice on the longest job.
- **No model passed T3** (the 4-step job). Don't read that as "impossible" — it is more
  about short-horizon reliability than raw capability. See
  [What the numbers are and are not](#what-the-numbers-are-and-are-not).
- **UGround is not a weak agent — it is a different part.** It only answers "where is
  X?", so it is scored only on that, and its 4/8 says it needs a bigger build before use.
- **UI-TARS-2B is the weakest here.** It kept deciding the page "was still loading" and
  waiting instead of clicking, even when the button was right there on screen.

## Which models are worth using for real work

| Use it for | Model | Why |
|---|---|---|
| **Doing whole browser jobs on its own** | **Fara1.5-4B** (`runanywhere/Fara1.5-4B-mlx-4bit`) | Perfect grounding, finished the 1-step and 2-step jobs, ~3 s per step, only 4.1 GB |
| **A fast "click here / type this" helper under a bigger planner** | **ShowUI-2B** (`mlx-community/ShowUI-2B-bf16-4bit`) | Best speed for its size (~2.7 s, 2.3 GB), 7/8 grounding |
| **Best pure pointer, if you have the memory** | **TongUI-3B** (`Bofeee5675/TongUI-3B`) | 8/8 grounding, but 8.8 GB and slower to load |
| **Not yet** | UI-TARS-2B-SFT, UGround-V1-2B | See the retest issues #12 (TongUI-7B) and #13 (UGround-7B) |

Start with **Fara** if you want an agent; start with **ShowUI** if you want a cheap local
actor behind an existing planner.

## How much it costs to run

All figures are **peak memory actually used** while running, measured on the Mac (the
model lives in fast "unified memory" shared with the CPU — normal Mac "RAM used" tools
under-report this, so it is measured directly).

| Model | On disk | Resident while running | Peak (high-water) | Boot-up | Time per action | Notes |
|---|---:|---:|---:|---:|---:|---|
| **ShowUI-2B** | 2.2 GB | **2.2 GB** | 3.2 GB | ~4 s | **~2.7 s** | Lightest and quickest |
| **Fara1.5-4B** | 4.0 GB | **4.0 GB** | 6.9 GB | ~4 s | ~3.1 s | Best all-rounder |
| **UGround-V1-2B** | 4.4 GB | **4.4 GB** | 5.4 GB | ~4 s | ~2.6 s | Pointer only |
| **UI-TARS-2B-SFT** | 9.8 GB (shipped fp32) | **4.5 GB** (cast to bf16) | 5.5 GB | ~17 s | ~4.7 s | Slowest to load, weakest answers |
| **TongUI-3B** | 7.5 GB | **7.6 GB** | 9.2 GB | ~16 s | ~3.5 s | Heaviest to run; best grounding |

"Resident" is what the model actually holds while running (the number to compare against
your RAM); "peak" is the high-water mark including load-time buffers, which is what matters
if you are squeezing alongside other apps. Both are accelerator memory in the Mac's shared
pool, measured directly after a real inference (`scripts/measure_memory.py`) - normal Mac
RAM monitors under-report this, so do not trust Activity Monitor for it. UI-TARS ships in
**fp32** (9.8 GB to download) but loads to ~4.5 GB once cast to bf16.

Practical read: **Fara and ShowUI fit comfortably** on a 48 GB Mac next to a browser, and
UGround is close behind. **TongUI-3B is the only one worth thinking about** (~7.6 GB
resident) and it takes ~16 s to start; it still fits, but it is the one that competes with
your other apps.

## What the numbers are and are not

Worth saying plainly, because it changes how much you should trust the table:

- **These are "this exact download, this speed" results, not judgements on the model
  family.** Fara and ShowUI were run in 4-bit; TongUI and UI-TARS in full precision. Those
  are not matched, so a difference could be the model *or* the compression. When a
  comparison mattered I said so.
- **The samples are small.** 8 screens, 2 tries per job, one run each. Treat a gap of one
  or two as noise. Nothing here says a model family is dead — only that a specific
  download did or didn't clear the bar in this run.
- **The full honest report, including these caveats, is [RESULTS.md](RESULTS.md).**

## Things that tripped us up

Two findings worth knowing before you run anything these models:

1. **The image can silently never reach the model.** The MLX library used for Fara and
   ShowUI does not apply the model's chat template for you — the command-line tool does it
   first, and the code entry point does not. If you skip it, the model runs happily but
   never sees your screenshot, and just prints the same nonsense answer every time.
   UGround looked "broken" for exactly this reason. Fix: apply the template yourself.
   Caught with a solid-colour test image, not by looking at the output.

2. **Each model needs its own home-made prompt and answer format.** They are not
   interchangeable:
   - **Fara** uses fixed 0–1000 coordinates (never screen pixels, even though its prompt
     quotes a screen size) and Microsoft's exact system prompt.
   - **ShowUI / TongUI** use 0–1 *relative* coordinates, and their "grounding" prompt is
     actually *worse* than their "navigation" prompt — use the navigation one.
   - **UI-TARS** has its own `Action:` line format with `<point>x y</point>` tags.

   Using the wrong one makes a good model look useless. That is the main lesson of this
   repo.

## Reproduce it

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python mlx mlx-vlm torch torchvision transformers \
    accelerate qwen-vl-utils pillow playwright numpy pytest
.venv/bin/python -m playwright install chromium
.venv/bin/python -m pytest tests/ -q     # parser + metrics + fixture checks
python -m harness.capture                # build the 8 grounding screenshots
scripts/run_all_stage1.sh                # groundings, ~15 min
scripts/run_all_stage2.sh                # tiny tasks, ~25 min
```

---

# Programme definition

The rest of this file is the original plan the runs followed: what each model is, the
scoring rules, and the promotion criteria.

## Goal

Answer one practical question:

> Which local models are fast and reliable enough on Rajeev's M5 Pro / 48 GB MacBook Pro to justify testing on real browser and desktop work?

This repo is deliberately a **small screening harness**, not another large benchmark. Models that fail basic local latency/grounding/action tests stop here. Promising models graduate to `web-automation-microbench` / real-work testing.

## What these models actually do

Most GUI models do **not** directly manipulate the Mac. They consume a screenshot plus an instruction/history and emit an action. A harness executes the action and captures the next screenshot.

```text
task + screenshot/history
          |
          v
     local GUI model
          |
          | structured action / coordinates
          v
      executor/harness
          |
          | mouse / keyboard / browser input
          v
         GUI
          |
          +---- next screenshot ---->
```

### Models

| Model | Role | Model output / action space | What actually executes it | First local route |
|---|---|---|---|---|
| UI-TARS-2B-SFT | end-to-end GUI agent | thought/action; click, double/right click, drag, hotkeys, typing, scroll etc., with coordinates | upstream parser can convert actions to PyAutoGUI; for fair browser tests we map them to Playwright | official Transformers model on MPS first |
| ShowUI-2B | lightweight visual actor / navigation VLA | atomic action dicts such as CLICK, INPUT, SELECT, HOVER, ENTER, SCROLL, SELECT_TEXT with relative screenshot coordinates | official Computer Use OOTB uses PyAutoGUI; our fair browser screen maps to Playwright | MLX 4-bit |
| TongUI-3B | general GUI VLA | JSON action(s); web actions include CLICK, INPUT, SELECT, HOVER, ANSWER, ENTER, SCROLL, SELECT_TEXT, COPY; can emit multiple actions | model output must be executed by a harness; our screen maps supported actions to Playwright | official Transformers/Qwen2.5-VL path on MPS |
| Fara1.5-4B | native browser CUA | `computer_use` tool calls: mouse clicks/move/drag, type/key, scroll, URL/history, web search, memory, ask-user, wait, terminate | Fara/Magentic-UI reference harness; our screen maps browser actions to Playwright | MLX 4-bit conversion with the official Fara prompt/tool schema |
| UGround-V1-2B | **grounder only** | target point/coordinate for a described GUI element | it does not plan or click; a planner + executor must use the predicted point | MLX bf16 |

Important: UGround must **not** be ranked as a failed end-to-end agent. Its job is only “where is the thing I should interact with?”

## Screening philosophy

Keep it cheap and diagnostic. We care about:

1. **Does it actually run locally on Apple Silicon?**
2. **How quickly does it produce the first usable action?**
3. **Can it point at the correct UI element?**
4. **Can its native action format be executed reliably?**
5. **Can the agent-capable models finish a tiny multi-step task?**
6. **Is the performance good enough to justify real-work testing?**

We do **not** start with OSWorld or a large public benchmark.

## Common executor

For browser tests use one isolated Playwright Chromium instance with a fixed viewport.

Why:

- same screenshot dimensions for every model;
- exact DOM bounding boxes provide objective grounding truth;
- actions are reversible and resettable;
- avoids giving PyAutoGUI-based models an artificial disadvantage or advantage;
- does not touch the user's normal browser/session.

Preserve each model's **native output/action schema**; only execution is normalized.

Native-harness sanity checks are secondary:
- UI-TARS -> upstream action parser / PyAutoGUI;
- ShowUI -> Computer Use OOTB;
- Fara -> Fara/Magentic-UI.

## Quick screening suite

### Stage 0 — local viability

One screenshot, one instruction, 3 warm repetitions.

Record:

- model/quant;
- download size;
- cold load seconds;
- first-action latency;
- warm p50 action latency;
- peak process RSS / unified-memory pressure;
- output parse success;
- whether any cloud call occurred.

**Stop** a model if it cannot produce valid GUI output locally after a reasonable compatibility attempt.

### Stage 1 — 8-action grounding screen

Generate deterministic screenshots from TodoMVC / small public browser pages. For each screenshot ask for exactly one target/action.

Target mix:

1. button/link;
2. text input;
3. checkbox;
4. filter/tab;
5. dropdown;
6. small text link;
7. target after scroll;
8. two visually similar controls.

Score predicted point against the target element's DOM bounding box.

Record:

- hit rate;
- pixel/normalized distance to target centre;
- warm inference latency;
- invalid-action rate.

Run all five models where the model supports grounding. UGround is evaluated **only here** in the first pass.

### Stage 2 — three tiny executable tasks

Use a clean Playwright TodoMVC/browser fixture with independent DOM verification.

**T1 — one-step:** click a named visible control.

**T2 — input:** create one named todo and commit it.

**T3 — short horizon:** add two todos, complete one, select Active, end in the expected state.

Run:
- UI-TARS direct;
- TongUI direct;
- Fara direct;
- ShowUI direct **and** actor mode with the next subgoal supplied;
- UGround only as `gold subgoal -> UGround point -> executor`.

2 reps each. No retrying away failures.

### Stage 3 — promotion decision

A model graduates to real-work testing if it offers something useful, e.g.:

- >= 7/8 grounding hits with useful latency; **or**
- 2/2 on T1 and T2 plus >=1/2 on T3; **or**
- uniquely strong grounding/latency as a component even if it is not an autonomous agent.

Do not create a fake single leaderboard across unlike roles. Report separate:

- **local grounder frontier**;
- **local actor frontier**;
- **local autonomous-agent frontier**.

## Metrics

Write one row per inference/action/run:

- model + exact artifact/revision;
- precision/quantization/runtime;
- task;
- native role;
- cold/warm;
- inference_ms;
- action_parse_ms;
- executor_ms;
- end_to_end_ms;
- peak_rss_mb;
- action type;
- predicted coordinates;
- target bounding box;
- hit;
- task pass;
- steps;
- parse/error class.

## Initial Apple-Silicon artifacts

Prefer fidelity and easy reproduction over squeezing every MB:

- **Fara1.5-4B:** `runanywhere/Fara1.5-4B-mlx-4bit` for the first screen; validate it uses Microsoft's exact system/tool schema.
- **ShowUI-2B:** `mlx-community/ShowUI-2B-bf16-4bit`.
- **UGround-V1-2B:** `mlx-community/UGround-V1-2B-bf16`.
- **UI-TARS-2B-SFT:** official `ByteDance-Seed/UI-TARS-2B-SFT` through Transformers/MPS first.
- **TongUI-3B:** official `Bofeee5675/TongUI-3B` through Transformers/MPS first.

Community quantizations are an execution convenience, not a new model. If a quant produces suspicious actions, repeat a tiny subset against the official bf16 model before rejecting the model family.

## Output

`RESULTS.md` should end with a compact table:

| Model | Role | Runs locally? | Warm action latency | Grounding | Tiny-task success | Memory | Promote? | Best use |
|---|---|---:|---:|---:|---:|---:|---|---|

The useful conclusion is not “which model wins?” It is:

> Which model, in which role, deserves integration into Playwriter / Browser Relay / the existing agent rig?

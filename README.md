# local_cua

Fast local screening of small open computer-use / GUI models on Apple Silicon.

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

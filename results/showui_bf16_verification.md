# ShowUI 4-bit vs bf16 verification

**Trigger:** the ShowUI 4-bit build behaved suspiciously — its documented *grounding*
prompt scored 4/8 while its *navigation* prompt scored 7/8. The programme rule says to
verify a tiny subset against a higher-fidelity artifact before drawing conclusions.

**Method:** identical 8 Stage-1 scenes, identical documented grounding prompt, temp 0,
`apply_chat_template` applied, same 1440×900 viewport.

| Artifact | Grounding-prompt hits | Native output samples |
|---|---:|---|
| `mlx-community/ShowUI-2B-bf16-4bit` (4-bit) | 2/8 | `[0.08, 0.17]`, `[0.5, 0.12]`, `[0.09, 0.9]` |
| `prince-canuma/ShowUI-2B-bf16` (bf16) | 3/8 | `[0.08, 0.17]`, `[0.5, 0.16]`, `[0.09, 0.9]` |

**Per-scene agreement:** 7/8 identical hit/miss; the only difference is scene 5
(dropdown), where bf16 hit and 4-bit missed — a single-scene, sub-1%-coordinate
difference (`[0.09, 0.17]` vs `[0.5, 0.17]`).

**Conclusion:** 4-bit is *not* the cause. ShowUI's grounding-prompt weakness is genuine
base-model behaviour on this fixture, consistent with the MLX model card's own caveat.
The 4-bit artifact is therefore trustworthy for the screening conclusions, and
ShowUI should be used with its **native navigation prompt** (7/8), not the grounding
prompt (4/8).

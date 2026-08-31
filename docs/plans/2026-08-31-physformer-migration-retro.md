# PhysFormer migration retro

What the PhysFormer migration decided, and what was awkward to express. The
config notes feed the Phase 5 schema design; the head/tokenization notes are
the record the migration contract (§2) asks a transformer migration to leave
behind.

Migrated: `neural_methods/model/PhysFormer.py`, builder `_build_physformer` in
`MultiSignalTrainer.MODEL_REGISTRY`, configs
`configs/neckflix/NECKFLIX_PHYSFORMER{,_SMOKE}.yaml`, smoke test
`tests/test_physformer_multisignal.py`. Deleted in the same change:
`neural_methods/trainer/PhysFormerTrainer.py`,
`neural_methods/loss/PhysFormerLossComputer.py`.

## Fidelity

Verified numerically, not just by inspection: the migrated module was run
against the pre-migration one (`git show cf52990:...PhysFormer.py`) with the
same weights loaded into both.

- `state_dict` identical — 483 tensors, same names, same shapes.
- Same input, `eval()`, 128x128 frames: `max |old - new| = 1.2e-07` on an
  output of scale 8.2e-02.
- All three attention score maps agree to 2.2e-08.

The attention maps were *bit*-identical until `CDC_T`'s kernel-difference
reduction was rewritten from `.sum(2).sum(2)` to einops `reduce`, which sums
the two kernel axes in one pass rather than two and so associates the floats
differently. Kept anyway: the rule's point is that the axes are spelled out at
the call site, and 2e-08 on a softmax is not a behavioural difference. Noted
because "bit-identical" is a claim worth keeping honest. (PhysMamba, the
reference migration, still has the `.sum(2).sum(2)` spelling in its own
`CDC_T`.)

So the original *is* recoverable at `in_channels=3, out_signals=1`, and a
published checkpoint would load into this module unchanged. An independent
review repeated the comparison across 16 configurations the first check did not
cover — non-square token grids (4x2, 2x4, 3x1, 1x3, 5x3), head counts 1/2/3/4/
6/12/24, batch sizes 1/2/3, windows 32/48/64/160 — and found the same:
`max|old - new| <= 9e-8`, attention maps bit-identical, and in `train()` mode
the BatchNorm running-stat buffers bit-identical with gradients agreeing to
fp32 noise.

**Re-verifying fidelity later.** This check is deliberately *not* a test. The
pre-migration module is deleted, so it survives only in git history, and
pinning a golden output would buy a brittle fixture in exchange for breaking
the "one smoke test per migration" ceiling. To re-run it after touching the
architecture:

```bash
git show cf52990:neural_methods/model/PhysFormer.py > /tmp/PhysFormer_original.py
# build both at patches=(4,4,4), dim=96, ff_dim=144, num_heads=4, num_layers=12,
# theta=0.7, image_size=(160,128,128); new one with in_channels=3, out_signals=1
# new.load_state_dict(old.state_dict()); compare in eval() on the same input.
# The original returns (B, T); the migrated one (B, 1, T).
```

The one break this cannot be traded for is a mis-ordered attention head split
(`"b (nh dh) ..."` vs `"b (dh nh) ..."`): it changes which channels form which
head, so the network is no longer the published one — but it still trains, and
no contract-level assertion can see it. That is exactly the class of defect the
comparison above exists to catch, and the reason it is written down here rather
than left implicit.

Three deviations, all sanctioned by the contract's fidelity principle:

1. `Stem0`'s first conv takes `in_channels` instead of 3. (The original
   constructor already had an `in_channels` argument — it was simply never
   used.)
2. `ConvBlockLast` emits `out_signals` planes instead of 1, and `forward`
   returns `(B, S, T)` rather than squeezing to `(B, T)`.
3. The loss is the trainer's per-signal registry, not PhysFormer's own — see
   "The DLDL loss" below.

Two constants were *derived* rather than hardcoded. Neither changes behaviour
at the published input size; both turn a crash deep inside a `view` into a
construction-time error that names the config key:

- The token grid. The original wrote `view(B, C, P//16, 4, 4)`, i.e. `gh = gw
  = 4` — true only at 128x128 with 4x4 patches. It is now computed from
  `image_size` and `patches`, so `RESIZE: 64` gives a 2x2 grid instead of
  silently reshaping garbage.
- The temporal patch count. `gt = T // ft` already came from the input, so the
  window length was never baked in; what is new is that `T % ft` is *checked*.
  The legacy trainer truncated the batch instead, silently.

All reshaping is einops, including the attention's head split/merge, which
replaced the module-local `split_last` / `merge_last` helpers.

## Head and tokenization (the Phase 6 decision)

**Tokenization: unchanged from the paper.** 3-D stem (three convs, three
`MaxPool3d((1,2,2))` -> H/8, W/8), then a `(4,4,4)` tube patch embedding, giving
`T/4 x H/32 x W/32` tokens — a 4x4 spatial grid at the published 128x128. The
temptation was to widen tokenization for five channels (R,G,B,I,D); resisted,
because the stem's first conv already absorbs channel width and everything
after it is what the fidelity principle protects.

**Head: Style A (widened readout).** The final `Conv1d(dim//2, S, 1)` is S
independent linear readouts of the shared temporal feature. Activation-free, so
ABP and CVP are expressible directly in mmHg; the bias carries the per-signal
physiological prior the builder seeds.

**Style B is not mechanically available for PhysFormer, and the builder says
so.** Style B's justification elsewhere (BigSmall, DeepPhys) is that the head
reads a *flattened spatial map*, so a per-signal head is a per-signal learned
spatial weighting — ABP from the carotid, CVP from the jugular. PhysFormer's
head reads a feature whose token grid has already been mean-pooled away
(`reduce(..., "b c t gh gw -> b c t", "mean")`), so S copies of the readout
would each see the identical vector and buy nothing but parameters. The
equivalent idea here would be a *per-signal pooling* over the 4x4 token grid
before the readout — a design change, not a builder flag. `_build_physformer`
raises with that explanation rather than silently accepting `HEAD_STYLE`.

If per-signal metrics later show ABP/CVP interference, that per-signal token
pooling is the change to make, and it is a genuinely new head, not Style B.

## The DLDL loss

PhysFormer's published recipe is not just negpearson. It adds a
frequency-domain term — `TorchLossComputer.cross_entropy_power_spectrum_DLDL_softmax2`,
a label-distribution-learning KL plus cross-entropy over a 40-180 bpm power
spectrum — under a two-phase weight schedule (`a=1, b=1*1^(epoch/10)` for the
first 10 epochs, then `a=0.05, b=5.0`).

That did not migrate as-is, for three reasons:

- It is single-signal by construction: it needs one heart rate per window,
  derived from the label by Welch. There is no such thing for ABP level or CVP.
- It was hardcoded to CUDA (`.cuda()`, `torch.cuda.FloatTensor` throughout), so
  it could never have run on the CPU smoke path.
- The contract (§3, §7) says the per-signal loss registry is the one place a
  loss lives, and later migrations follow the pilot's precedent rather than
  adding a parallel criterion in the trainer.

So `PhysFormerLossComputer.py` was deleted with the trainer, and the config
expresses the frequency term through the registry instead:

```yaml
LOSS:
  ECG: {TYPE: shape, WEIGHTS: {NEGPEARSON: 1.0, SPECTRAL: 0.5}}
```

`PerSignalLoss`'s `spectral` component is band-limited log-spectrum L1, which
is a *shape* match rather than DLDL's soft-classification over a bpm grid.
It is the right slot but not the same objective, and this is the one place the
migration is knowingly weaker than the paper. Two follow-ups worth considering,
both registry-shaped and neither a migration blocker:

- add a `dldl` component (soft bpm-distribution KL) for pulsatile shape-class
  signals, reusing the deleted implementation from git history but device-clean
  and per-sample-reducing;
- the two-phase `a`/`b` schedule has no home at all in the current design —
  component weights are static. If epoch-dependent weighting turns out to
  matter, it is a `PerSignalLoss` feature, not a PhysFormer one.

The published per-window output normalisation
(`rPPG = (rPPG - mean) / std` before the Pearson loss) also did not migrate: it
would erase exactly the absolute level ABP and CVP are supposed to carry.
negpearson is scale-invariant anyway, so shape-class signals lose nothing.

## What the tests do and do not cover

The smoke test was mutation-tested — each defect injected, suite re-run:

| Injected defect | Caught |
| --- | --- |
| `tanh` applied after the readout | yes |
| token grid returned transposed | yes |
| readout row 0 broadcast to every signal | yes |
| `temporal_divisor` wrong | yes |
| attention head split reversed | **no** — fidelity only, see above |

The first pass of the test file caught only the last of these: it asserted the
readout *was* a `Conv1d` rather than that its output is unsquashed, used a
square 32x32 frame (a 1x1 token grid, where a transposed grid is invisible and
the depth-wise spatio-temporal conv sees nothing but padding), and re-tested
`DictModel` plumbing that `test_batch_contract.py` already owns. Worth stating
as a general lesson for the remaining migrations: **a shape test passes through
almost every interesting bug.** Assert on values and on differences between
signals, size the fixture so the geometry is actually exercised (non-square,
grid > 1x1), and leave the dict plumbing to the contract tests.

Three latent bugs the same review found, all now fixed:

- `forward` validated `T` but never `H`/`W`. A model built for a 4x4 grid and
  fed 64x64 frames did **not** raise — the token count still divided by 16, so
  the reshape succeeded and reinterpreted four time steps' tokens as one step's
  spatial grid, returning a prediction a quarter the window's length. Only
  latent (the builder's frame transform always resizes), but it was precisely
  the failure the derived grid was supposed to have eliminated.
- The spatial error message named the stem's 8x pooling but not the patch
  stride, so `RESIZE.H: 20` was told "must be a multiple of 8" and `24` then
  failed differently. Both branches now state the real requirement (32) and the
  nearest valid value.
- `image_size`'s `T` was accepted and silently discarded while the docstring
  claimed it was checked. It is now actually validated at construction.

## Config retro

What was awkward, duplicated, or forced by the yacs tree.

**1. `MODEL.PHYSFORMER` needs six architectural keys, not "at most a couple".**
`PATCH_SIZE`, `DIM`, `FF_DIM`, `NUM_HEADS`, `NUM_LAYERS`, `THETA`. All six are
genuinely architectural — none is derivable from the data spec — so this is not
a violation of §1 so much as evidence that the "at most a couple" guideline
does not survive contact with a transformer. Phase 5 should expect a per-model
architecture block of arbitrary size and stop treating it as a smell. Two keys
that could have been there and deliberately are not:

- `GRA_SHARP` — the attention softmax temperature. Every published PhysFormer
  trainer hardcodes 2.0, so it is a model default, not a config key. Promote it
  only if someone actually wants to sweep it.
- `HEAD_STYLE` — read from the shared `MODEL.HEAD_STYLE`, and refused for
  anything but `widened` (above).

**2. `PATCH_SIZE` is not really free.** The head upsamples time by 2x2, so the
temporal patch must be 4 for the prediction to come back at the window length.
The constructor refuses anything else by name. A single `PATCH_SIZE` key that
sets all three tube dimensions but is only free in two of them is a small lie;
Phase 5 could split it into `PATCH_SPATIAL` plus a fixed temporal 4.

**3. `STRIDE_SECONDS: 0` is a type error.** yacs coerces strictly against the
default's type, and the default is `0.0`, so a bare `0` — the natural spelling
of "no overlap", and what the key's own comment invites — fails with
`Type mismatch (<class 'float'> vs. <class 'int'>)`. Same trap for any integer
written where a float default lives. Phase 5 wants a schema that coerces
int -> float rather than refusing.

**4. `DATA.FS` cannot hold a non-integer rate.** Its yacs default is `0` (int),
so `FS: 29.97961373390558` is refused by the same coercion rule. That matters
because the Neckflix cache is *not* 30.000 fps (below), so the one value that
would satisfy the strict native-rate check is unwritable. `FS` must default to
`0.0`.

**5. The cache's nominal rate is not its exact rate.** Surveying all 332 stores
in the local `rgb128` cache, the per-stream `video.fps` attr takes three
distinct values across 655 streams:

| native fps | streams |
| --- | --- |
| 29.97961373390558 | 329 |
| 30.0 | 325 |
| 29.98051282051282 | 1 |

so "30 fps Neckflix" mixes ~29.9796 and exactly 30.0 — and, as the pilot's own
survey found, **320 of the 332 stores disagree with themselves**, carrying both
rates across streams within a single perspective. Two stage-0 rules were wrong
on this, not one: the target-rate refusal rejected `FS: 30` outright, and a
separate "streams must agree on fps" check would independently have rejected
96% of the cache. Both are now relative (`same_nominal_rate`, 1% of the larger
rate), and at the same nominal rate the window sampler short-circuits to a
plain contiguous slice rather than resampling on 0.07% of measurement jitter.

The lasting lesson for the config schema: `FS` is a *nominal* rate the config
names, and the loader reconciles it with whatever each store actually recorded.
Any future doc that says "160 frames at 30 fps is 5.333333 s" is stating a
nominal identity, not an arithmetic one — and any equality test against a rate
read from a cache should be a tolerance test.

**6. Four data blocks still restate the same 15 lines.** `TRAIN`/`VALID`/`TEST`
differ only in `STRIDE_SECONDS` (overlap when training, none when evaluating).
Already known to Phase 5; PhysFormer adds no new argument, just another vote.

**7. Good, keep it.** Omitting `LABEL_NORM` entirely and letting the signal
class decide (ABP/CVP raw, ECG zscore) is the single nicest thing about the new
schema — the config says only what is a *choice*, and `signals.py` stays the
one place the classes live. Same for omitting the `LOSS` entry for ABP and CVP.

## Cost note

`MultiHeadedSelfAttention_TDC_gra_sharp` stashes its score map on the module
(`self.scores = scores`), faithfully inherited from upstream. At the shipped
config (batch 4, 4 heads, 640 tokens, 12 blocks) that is ~315 MB of the
~4 GiB peak, retained across steps even though `forward_video` discards the
scores it is handed. Removing it would be a small fidelity deviation rather
than a free win — the attribute is the paper implementation's visualization
hook — so it stays. Worth knowing if PhysFormer ever needs a larger batch than
the card allows: it is the cheapest ~8% of memory available.

## Running it

Two runs, both against the committed configs on the local GPU (see
`docs/project_status.md` for the headline).

**CPU smoke** — `--limit_windows 8 --test_participants P015` against
`NECKFLIX_PHYSFORMER_SMOKE.yaml`: train, test, all three signals scored, four
plot families written, outputs pickled, and `tools/summarise_neckflix_outputs.py`
reads them back with `label_norms={'ABP': 'raw', 'CVP': 'raw', 'ECG': 'zscore'}`
intact.

**Two-epoch GPU run** — `NECKFLIX_PHYSFORMER.yaml` at its real settings
(128x128, 160-frame window, 4x4 token grid, batch 4, bf16), P015 held out,
`--limit_windows 2000`, 500 steps/epoch at 2.25 s/step, ~21 min/epoch.

| | mean train loss | ABP | CVP | ECG |
| --- | --- | --- | --- | --- |
| epoch 0 | 3.6211 | 5.2781 | 3.6806 | 1.9047 |
| epoch 1 | 3.4228 | 5.2824 | 3.1586 | 1.8275 |

Test, 70/70/60 windows: ABP waveform Pearson -0.059, MAE 21.7 mmHg; CVP 0.007,
3.03 mmHg; ECG 0.020, 0.657 (z-scored).

**Read that as a plumbing check, not a result.** Two epochs over 2000 windows
is nowhere near convergence for a 7.4 M-parameter transformer whose published
recipe runs tens of epochs over a full dataset; a waveform Pearson of ~0 and an
ABP MAE of 21.7 mmHg are what an untrained readout sitting near its 90 mmHg
prior looks like. What the run does establish is that every piece of the
contract survives a real training loop end to end: physical-time windowing over
the real cache, per-signal label norms (raw mmHg for ABP/CVP alongside a
z-scored ECG in the same batch), the per-signal composite loss with its
component breakdown, masked absent traces, bias init, the weight-decay
exemption, checkpointing, per-signal scoring in physical units, and all four
plot families.

The one thing worth a second look when a real sweep runs: CVP and ECG both
moved between the two epochs while ABP did not. Far too early to mean anything,
but ABP's absolute level is the genuinely hard inference here, and the
per-signal loss breakdown is exactly the instrument for watching it.

Cost note for whoever schedules the LOSO sweep: measured 4.05 GiB peak and
2.1 s/step at batch 4 on a 16 GB card, 7.38 M parameters, 640 tokens per
window. A full P015 fold is 6002 train / 90 test windows, so an unsubsampled
epoch is ~50 min at that batch size. Batch 4 is comfortable; the ~315 MB of
retained attention maps (above) is the first thing to reclaim if a larger batch
is ever wanted.

## §7a re-verification on the DATA / INTERFACE / MODEL schema

Run against the migration contract's §7a recipe after the config redesign, on
the drafted `NECKFLIX_PHYSFORMER{,_SMOKE}.yaml`. **No architectural
discrepancy was found**; `PhysFormer.py`, `_build_physformer` and
`tests/test_physformer_multisignal.py` were not touched.

**Config.** Every pilot setting survived the conversion: `FS 30`,
`WINDOW_SECONDS 5.333333` (→ T = 160, the published `T_orig`), `CHANNELS
[R,G,B]`, `TRACES [ABP,CVP,ECG]`, `RESIZE 128x128` (→ the 4x4 token grid),
`DATA_TYPE [DiffNormalized]`, all six `MODEL.PHYSFORMER` keys, `DROP_RATE
0.1`, LR 1e-4, batch 4, 30 epochs, bf16, `TRAIN` stride 2.666667 s, and a
`LOSS` block naming only ECG (`shape`, negpearson 1.0 + spectral 0.5) so
ABP/CVP take their class defaults — retro item 7's "good, keep it", still
good. `HEAD_STYLE` is unstated and defaults to `widened`. The `_SMOKE`
variant resolves to `BASE` plus exactly eleven overridden leaves: `DEVICE`,
`LOG.PATH`, the TRAIN stride, `WINDOW_SECONDS`, `RESIZE.H/W`, `TRAIN`
batch/epochs/AMP/filename and `TEST.BATCH_SIZE`. The architecture block is
inherited byte-for-byte, so the smoke run exercises the same tokenization
path at a 2x2 grid.

**Built model.** `build_model(load_config(...))` gives 7 380 963 parameters
and a backbone `state_dict` of **483 tensors** — the pilot's number. The
model's own `state_dict` reports 484: the extra entry is `_fs`, the scalar
rate buffer `DictModel` registers on every dict-contract model, not
architecture. Worth stating explicitly, because "483" and "484" are both
right answers to slightly different questions and the next person to check
will hit the same off-by-one. The readout is a single activation-free
`Conv1d(48, 3, 1)` whose bias is seeded `[90.0, 8.0, 0.0]` mmHg/mmHg/z.

All four refusals still fire, each naming the config key:

| Provocation | Result |
| --- | --- |
| `HEAD_STYLE: per_signal` | refused, with the pooled-grid explanation |
| `WINDOW_SECONDS 5.266667` (158 frames) | refused, "multiple of 4 ... nearest valid 160" |
| `RESIZE.H: 100` | refused, "positive multiple of 32 ... nearest valid 96" |
| `RESIZE.W: 20` / `24` | both refused with the same 32 requirement (the retro's earlier two-step failure stays fixed) |

The latent forward-time H/W bug stays fixed: `forward_video` at 64x64, 128x64
and 96x160 on a model built for a 4x4 grid is refused naming both grids.
Reaching it takes a direct `forward_video` call — through `model(batch)` the
frame transform resizes first, which is why the bug was latent in the first
place.

**Suite and smoke run.** `pytest tests/ -q` → 280 passed. The real smoke run
(`--limit_windows 8 --test_participants P015` against `_SMOKE`) trains,
checkpoints, scores all three signals in physical units and writes all four
plot families.

**Checkpoint authority — and why it is load-bearing here.** A `MODE:
only_test` config with a deliberately wrong `RESIZE: 96x96` pointed at the
smoke checkpoint printed

```
Adopting the checkpoint's interface over the config's:
  INTERFACE.RESIZE: config={'H': 96, 'W': 96} checkpoint={'H': 64, 'W': 64}
```

named the run directory `H-64_W-64`, and reproduced the smoke run's test
metrics exactly (ABP Pearson −0.0565, MAE 21.6968 mmHg).

The reason this matters more for PhysFormer than for a conv model:
**PhysFormer's parameter shapes are entirely grid-invariant.** Built at 64x64
(2x2 grid) and at 96x96 (3x3), the two `state_dict`s have identical keys
*and* identical shapes, so the 2x2 checkpoint loads into the 3x3 model under
`load_state_dict(strict=True)` with "All keys matched successfully" and then
runs to completion. The token grid lives only in the `rearrange` patterns,
never in a weight. So a wrong `RESIZE` that happens to stay a multiple of 32
is invisible to the 32-multiple constructor check, invisible to the
forward-time grid check (the frame transform dutifully resizes to the wrong
size the config asked for), and invisible to checkpoint loading. **The
interface adoption is the only mechanism in the pipeline that catches it** —
the constructor's 32-multiple rule protects against garbage geometry, and
the checkpoint's interface protects against *plausible* wrong geometry.

### What the new schema cannot express

1. **`check_window`'s seconds-based error is dead code for PhysFormer.**
   `build_model` constructs before it calls `check_window`, and
   `PhysFormer.__init__` runs its own `_check_window_length` — so the
   message the user actually sees is "window length must be a multiple of 4;
   got 158. Nearest valid: 160", in frames. The trainer's message, which
   deliberately speaks the config's own language ("`WINDOW_SECONDS` gives 158
   frames. Use `WINDOW_SECONDS: 5.333333` at FS=30"), is never reached. Not
   worth reordering for one model; worth knowing that any model declaring its
   own temporal check shadows the nicer error, and worth a Phase 5 decision
   about which layer owns that message.
2. **`RESIZE` has no way to say "this is architectural".** For PhysFormer it
   is not a rendering preference — it sets the token grid, and the section
   above shows a wrong-but-legal value is undetectable outside the checkpoint
   path. There is no schema-level way for a model to declare which
   `INTERFACE` keys it is *structurally* bound to versus merely consumes.
   Everything works today because the checkpoint carries the whole interface;
   the gap only shows up for a hand-written `only_test` config with no
   checkpoint interface (the pre-redesign bare-`state_dict` path, which warns
   and trusts the config blind).
3. **Retro item 2 stands unresolved by the redesign.** `PATCH_SIZE` is still
   one key setting all three tube dimensions while only the two spatial ones
   are free; the constructor still refuses anything but 4. The new schema did
   not make this expressible, and splitting it into `PATCH_SPATIAL` plus a
   fixed temporal 4 remains the Phase 5 suggestion.
4. **Retro items 3, 4 and 6 are closed by the redesign**, as designed:
   `STRIDE_SECONDS: 0` now coerces int→float, `FS` holds a non-integer rate,
   and the four data blocks are one `DATA` block with a `SPLITS` policy. The
   PhysFormer config's `SPLITS` is two lines (`TRAIN` stride, `TEST: {}`)
   where it used to be four blocks of fifteen.

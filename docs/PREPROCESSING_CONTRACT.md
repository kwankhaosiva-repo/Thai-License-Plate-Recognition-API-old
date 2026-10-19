# Preprocessing Contract — the standard for this project

**Status:** active standard. Enforced by [`src/test_preprocess_contract.py`](../src/test_preprocess_contract.py).
**Companions:** [`BOX_GEOMETRY_AUDIT.md`](BOX_GEOMETRY_AUDIT.md) (crop/box geometry),
[`RETRAIN_V6.md`](RETRAIN_V6.md) (the commands that consume it).

---

## 0. Why this document exists

The project accumulated `ENABLE_*` flags, and each one was justified by a
measurement of the form *"turning it off makes the score worse"*. That pattern
is a trap. It looks like evidence the step is valuable; it is actually evidence
the step is **compensating for a defect somewhere else** — a wrong training
crop size, a wrong annotation frame, or an undertrained model. The flag then
becomes load-bearing, nobody dares remove it, and the underlying defect is never
found.

This document replaces "turn it off and see" with three explicit laws.

---

## 1. The three laws

### Law 1 — EVAL == TRAIN
> Anything a model sees at serve must appear, identically, in the pipeline that
> trained it. Anything that appears only in training is **augmentation** and
> must be absent at serve.

A deterministic operation (resize, letterbox, autocontrast, colour conversion)
is part of the model's *input contract*. Change one side without the other and
you have a train/serve mismatch — you do not have an accuracy gain.

Corollary, and the one people get wrong: **eval must not be "more normalised"
than train.** Making every serve path run the same pile of preprocessing is not
the goal. The goal is that each model's eval transform is a *prefix* of its own
train transform. Three different models legitimately have three different eval
transforms.

### Law 2 — NO CRUTCH
> A preprocessing step is legitimate only if turning it off does not degrade the
> metric. "It gets worse without it" is a **symptom**, never a justification.

When a step appears load-bearing, the response is to find the defect it is
hiding — not to keep the step. The defect is usually one of:

- the crop the model is trained on is a different **size/aspect** than the crop
  serve produces;
- the boxes the model is trained on are drawn in a different **frame**;
- the model itself is undertrained at that resolution or that coordinate range.

### Law 3 — GEOMETRY IS THE CONTRACT
> The crop size, aspect and coordinate frame a model is trained at are a
> **contract, not a tunable**. Rotation, scale and colour jitter belong in TRAIN
> augmentation.

If the model must be fed geometrically-correct input, fix the input at
training time. Do not repair geometry at serve time with a search.

---

## 2. The per-stage contract

`eval transform` is what serve runs. `train prefix` is what the trainer runs
**before** its augmentation. Under Law 1 the eval column must equal the train
prefix, op for op and constant for constant.

| Stage | Eval transform (serve) | Train prefix | Augmentation (train only) |
|---|---|---|---|
| **M1** plate detect | stretch to 416×416, `/255` | same | LibreYOLO defaults |
| **M2** component detect | stretch to 416×416, `/255` | same | LibreYOLO defaults |
| **M3A** char box detect | stretch to 640×640, `/255` | same | LibreYOLO defaults |
| **Province** (thai/lao) | `convert('L')` → **letterbox** to 256×80 / 256×64 with the crop's **corner-median** background → ToTensor → Norm(0.5, 0.25) | identical, in `train_grayscale_province_*.py` | `RandomAffine(degrees=5)`, ColorJitter, RandomAdjustSharpness, RandomAutocontrast |
| **Char** classifier | `autocontrast(cutoff=2)` → square-pad 64×64 corner-median → Resize(64,64) → Norm(ImageNet) | identical | `RandomAffine(degrees=6, scale=(.95,1.05))`, ColorJitter, RandomGrayscale, RandomPerspective |
| **OCR** CTC | `convert('L')` → `autocontrast(cutoff=1)` → `SmartResize(256,64)` black-pad → ToTensor | identical | ColorJitter, `RandomAffine(15°)`, `RandomPerspective` |

Note the deliberate differences, all Law-1-clean:

- province letterboxes with the **crop's own background tone**; OCR letterboxes with
  **black**. Both sides match for each model. They do not have to match *each other*.
- province has **no** autocontrast at eval — `RandomAutocontrast` is augmentation only.
- char classifier has **no** resize/colour jitter at eval beyond the deterministic
  `autocontrast(cutoff=2)` that is also in its `__getitem__`.

**The plate crop itself is a contract, not a step.** A plate crop in this system
*is* the perspective-rectified 320×160 quad (`warp_perspective_plate`, aspect
2.00). There is no raw alternative. See §4.

---

## 3. Why `ENABLE_HOMOGRAPHY_DESKEW` had to be ON for the plate to look straight

This was the question that motivated the whole document, so it gets the full
record.

### What the flag actually did

One switch, two unrelated operations:

1. **quad detection + `warp_perspective_plate()`** — maps the ordered plate quad
   onto an axis-aligned 320×160 rectangle. *Perspective rectification.*
2. **`fine_deskew_plate()`** — an independent search: rotate by every angle in
   ±8° at 0.25° steps, keep the one maximising the variance of the horizontal
   gradient projection. *Residual rotation correction.*

Turning the bundle OFF removed **both**, so M2/M3A were handed a raw
axis-aligned bounding box of a possibly-rotated plate — a geometry no training
crop has ever contained. The plate looked tilted. The flag got the credit.

It never had a fair ON/OFF test, because OFF was not a pipeline, it was a
broken branch.

### What the residual search was actually doing

Measured on the real rectified crops of 120 v5i-valid images
([`scratch/diag_residual_deskew.py`](../scratch/diag_residual_deskew.py)):

| statistic | value |
|---|---|
| crops it actually reaches | 35 / 120 |
| crops it rotates by ≥ 1° | **69 %** |
| \|angle\| median | 1.25° |
| \|angle\| p90 | 2.25° |
| \|angle\| max | **2.75°** |
| crops needing ≥ 3° | **0 %** |

So the ±8° search range was **3× wider than anything ever used**, and on the
two-thirds of plates it touched it injected a per-image angle that is by
construction outside the training distribution.

### Does it earn its keep?

Same code state, same 120 images, single loaded service, flags flipped live
([`scratch/ab_flag_matrix.py`](../scratch/ab_flag_matrix.py)):

| mode | rectified | residual | detected | province | quad present |
|---|---|---|---|---|---|
| `K2_default` (**new default**) | yes | no | 116 | **111/120 = 92.5 %** | 73 % |
| `K_residual_on` (old behaviour) | yes | **yes** | 118 | 112/120 = 93.3 % | 75 % |
| `B2_no_rectify` | **no** | no | 117 | **111/120 = 92.5 %** | 0 % |

**Δ = 1 image (0.8 points) on a 120-image sample whose margin is ±4 points.**
And rectification itself — once the province-padding contract was fixed — buys
**province accuracy of exactly zero**: 111 vs 111. It changes 18/120 plate texts,
but the v5i set carries no character ground truth, so which direction is better
is undecidable.

### The resolution

- `ENABLE_HOMOGRAPHY_DESKEW` stays **True**, and is documented as *the crop
  definition* — not an accuracy knob. It has no effect on the metric.
- The residual search split out into **`ENABLE_RESIDUAL_DESKEW`, default
  `False`**, routed through one gate `apply_residual_deskew()` so it can no
  longer hide inside the rectification path.
- **The rotation tolerance moved into train augmentation.** It was already
  there and already sufficient: province trains at `RandomAffine(degrees=5)`,
  the char classifier at `degrees=6`, OCR at `degrees=15`. The models have been
  trained to tolerate 2–5× more tilt than the crutch ever applied.

Re-enabling it requires a recorded A/B, not a hunch.

---

## 4. Law 2 applied to every other flag

Same 120-image harness. `A_current` = the shipped defaults at the time of the
audit.

| mode | what it turns off | province | Δ vs baseline | plate_text changed |
|---|---|---|---|---|
| baseline | — | 112/120 = **93.3 %** | — | — |
| `ENABLE_AMBIENT_CORNER_PAD=False` | province letterbox → stretch | 105/120 = 87.5 % | **−5.8** | 0/120 |
| `ENABLE_AUTOCONTRAST=False` | char `autocontrast(cutoff=2)` | 105/120 = 87.5 % | −5.8 | 9/120 |
| `ENABLE_GRAYSCALE_PROV_NORM=False` | province RGB instead of `L` | 104/120 = 86.7 % | −6.6 | 6 province flips |
| `PURE_MODEL_MODE` | everything, all heuristics | 104/120 = 86.7 % | −6.6 | 50/120 |
| `I_chain_no_heuristics` | heuristics only, geometry kept | 112/120 = **93.3 %** | **0.0** | 16/120 |

Read the last two rows together. **All the model-visible gain is in the
geometry/colour contracts; none of it is in the post-processing heuristics.**
The CLAHE recovery path, the stroke disambiguation, the regex repair, the CTC
fallback and the box/CTC reconciliation together account for
province 0.0 points and 16/120 text changes — on a set with no character labels.

### The one real regression, explained

`ENABLE_AMBIENT_CORNER_PAD` was the only flag that looked like a free win, and it
was a bug. `train_grayscale_province_thai.py:41` carries its **own private
`GrayscaleSmartResize` with no toggle** that *always* letterboxes with the
corner-median background. Serve read the flag and silently substituted a pure
**STRETCH**. The native province crop is ~118×30 (aspect 3.93) into a 256×80
tensor (aspect 3.20) — stretch squashes the glyphs **23 % horizontally**, a
geometry the classifier never saw.

So the +5.8 points were never "ambient padding is good". They were **the model
being fed the geometry it was trained on**. The flag now defaults to `True` and
`src/test_preprocess_contract.py` asserts the letterbox, not the stretch.

---

## 5. Review checklist

Before merging any change to a trainer, a harvester, or `src/preprocess.py`:

- [ ] Does the eval transform equal the train **prefix**? (Law 1)
- [ ] Is every op in the eval transform present in training, with the same
      constants — including the resize *mode* and the padding colour?
- [ ] Is every op in training **absent** at eval unless it is deterministic?
- [ ] Does the crop the model is trained on have the same aspect and coordinate
      frame as the crop serve produces? Measure both, don't assume.
      (`scratch/audit_retrain_datasets.py`)
- [ ] If a flag looks load-bearing, find the defect it hides before keeping it.
      (Law 2)
- [ ] Is any rotation/scale correction done at serve time? Move it to
      augmentation. (Law 3)
- [ ] Do `src/test_preprocess_contract.py`, `src/test_classifier_preprocess.py`
      and `src/test_preprocess_registry.py` still pass?

## 6. How to measure an A/B properly

`scratch/ab_flag_matrix.py` is the harness. Two rules learned the hard way:

1. **Re-baseline after every code change.** Rows are keyed by
   `(filename, mode)`, so a mode scored before an unrelated fix is silently
   stale — that cost one round of nonsense conclusions. Fresh modes are named
   `K2_default` / `B2_no_rectify`; older ones are `K_residual_on` / `B_deskew_off`.
2. **Report paired deltas, not totals.** 120 images ⇒ margin ≈ ±4 points. Only
   deltas above ~8 points are decisive; everything else needs a bigger sample or
   a character-labelled subset.

## 7. Known open items

- **No character-level ground truth on v5i.** Every `plate_text` delta in this
  document is "how many predictions changed", not "how many are now correct".
  Any decision about characters — including whether the residual deskew helps —
  is undecidable until a labelled subset exists. This is the single highest-value
  thing to fix next.
- **27 % of plates get no rectification at all** (no quad, so the AABB crop
  stands). Both branches therefore feed the models at once. Training the
  recognisers on rectified crops only, or removing the fallback, would make the
  contract single-valued.
- **`upscale_to_train_geometry` is a geometric no-op** (see
  `BOX_GEOMETRY_AUDIT.md` §4). It only changes interpolation quality. Left as-is
  deliberately; fixing it needs a non-square tensor and nothing uses
  `ResizeMode.ASPECT_STRETCH` today.
- **`cfg.USE_X` and `Config.is_x_enabled()` are duplicated** for every flag.
  Two definitions of the same predicate is drift waiting to happen.
- **`ENABLE_STROKE_DISAMBIGUATION` still gates the truck-banner DLT path**
  (`api_server.py:1470`), i.e. a geometric switch is load-bearing for a
  syntax rule. Decouple.
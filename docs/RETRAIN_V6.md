# RETRAIN V6 — dataset audit and the exact commands

**Read first:** [`PREPROCESSING_CONTRACT.md`](PREPROCESSING_CONTRACT.md) (the rules these commands implement).
**Reproduce the audit:** `python scratch/audit_retrain_datasets.py`
**Enforce the rules:** `python src/test_preprocess_contract.py` → 46 PASS.

---

## 0. Pre-flight — every command in this document was executed (2026-10-03)

Each command below was run (dry-run where the tool has one) against the current
checkout. Three of them were **wrong as first written** and are fixed in place;
the rest are confirmed copy-pasteable.

| # | command | result |
|---|---|---|
| 1 | `convert_coco_to_yolo.py --dry-run` | ✅ runs — 657 imgs, 3398 boxes, 110 without a `char` box, aspect 0.597, **0 % wider-than-tall** |
| 2 | `convert_coco_to_yolo.py` (real) | ✅ **already done** — `datasets/Thai/LPR_CharBox_New` holds `train 390 / valid 106 / test 51`, `nc: 1`, `names: [char]` |
| 3 | `prepare_serve_frame_components.py --dry-run --limit 300` | ✅ runs — province `h` 0.224 (target 0.225), char-row `h` 0.628 (target 0.531) |
| 4 | `prepare_serve_frame_components.py` (real) | ⏳ **not built yet** — `datasets/Thai/LPR2_ServeFrame.yolov11` does not exist |
| 5 | `train_libreyolo_task.py --aug plate` | ✅ all 12 aug keys exist in the installed `TrainConfig` — nothing is silently dropped |
| 6 | `train_grayscale_province_thai.py` / `train_character_classifier.py` | ✅ `--data-dir` accepts both `recognition_v2/*_crops_v2` roots (no `splits/` subdir → used directly) |
| 7 | `extract_recognition_crops_v2.py --overwrite` | ⚠️ **destructive** — `shutil.rmtree(recognition_v2/)` removes artefacts the harvester does not regenerate. See §4.3. |
| 8 | `train_plate_corner_regressor.py --epochs 40` | ❌ **no CLI exists** — the script has no `argparse`. Fixed in §4.6. |
| 9 | config edits in §5 | ❌ **pointed at `.onnx`** and at `resnet34`; both wrong. Fixed in §5. |

Two measured facts that were not in the first draft:

- **The new charbox export is better aligned with the GT text source.** The
  harvester's exact-count filter (`len(M3A boxes) == len(GT chars)`) looks the
  plate text up by filename in `thai_character_crops/metadata.csv`:

  | char-row source | images | GT lookup hit |
  |---|---|---|
  | `LPR 2 - Character Box Detection.yolov11` | 658 | **545 (82.8 %)** |
  | `LPR_CharBox_New` (new export) | 547 | **544 (99.5 %)** |

  17 % of the old set is thrown away as `no_gt_text` before any box is even
  considered. The new export is a subset that is almost fully covered — one more
  reason to prefer it beyond the aspect fix.

- **Two character classes have no data at all.** `recognition_v2/char_crops_v2`
  carries 48 of the 50 classes in `char_classifier_map.json`; `ฏ` and `ฑ` are
  absent, and the GT text contains each exactly **once** in 545 plates. The
  trainer keeps a 50-output head (it reads the map, not the directories) so
  retraining is safe, but those two classes are unreachable and will never
  improve from more harvesting of this corpus.

---

## 1. The headline

The code fixes are done. What they exposed is that **the models were trained with
no augmentation at all**, and every "preprocessing improvement" in this repo was
patching that:

| LibreYOLO default | value | serve-side crutch it created |
|---|---|---|
| `degrees` | **0.0** | `fine_deskew_plate()` ±8° — now split out, default **off** |
| `translate` | 0.0 | — |
| `perspective` | 0.0 | — |
| `mosaic_prob` | **0.0** | 3× LANCZOS4 upscale before M1 to recover small plates |
| `hsv_prob` | **0.0** | `ENABLE_OPENCV_CLAHE` M1 recovery path |
| `flipud` / `mixup` / `erasing` | 0.0 | — |
| `flip_prob` | 0.5 | the *only* augmentation any detector has had |

So the retrain below is not a routine refresh. It is the step that makes the
serve-side switches removable, which is the whole point of
[`PREPROCESSING_CONTRACT.md`](PREPROCESSING_CONTRACT.md).

---

## 2. Dataset audit — does train match serve?

Measured on 250 images per detector dataset (`scratch/audit_retrain_datasets.py`)
and 120 live-pipeline images (`scratch/ab_flag_matrix.py`, mode `K2_default`).

### Detectors — geometry is fine, **augmentation is the defect**

| Stage | dataset | imgs | train aspect | serve aspect | Δ | verdict |
|---|---|---|---|---|---|---|
| M1 plate | `LPR_2_BBox` | 1526 | 1.33 | 1.33 | 0 % | ✅ geometry, ❌ no aug |
| M2 components | `LPR 2 - Charactor Detection.yolov11` | 879 | 1.94 | 2.00 | 3 % | ✅ geometry, ❌ **wrong frame** |
| M3A char box | `LPR 2 - Character Box Detection.yolov11` | 3219 | 3.77 | 3.73 | 1 % | ✅ geometry, ⚠️ box-shape gap |

No split leakage in any of the three (checked by de-hashed stem).

### M2 — the box **frame** is wrong. This is the real finding.

Box positions as a fraction of the plate image:

| box | train (`LPR 2 - Charactor Detection`, 162×84) | serve (rectified 320×160) | Δ |
|---|---|---|---|
| **province** cy | **0.697** | **0.856** | **+0.159** |
| province h | 0.186 | 0.225 | +0.039 |
| province aspect | 5.72 | 3.40 | −2.32 |
| char-row cy | 0.385 | 0.419 | +0.034 |
| char-row h | 0.381 | 0.531 | +0.150 |
| char-row aspect | 3.83 | 3.22 | −0.61 |

**M2 is asked to regress a province box 16 % of the plate height lower than any
province box in its training set.** It copes — measured IoU against the v5i
ground truth is median 0.798 — but only by generalising onto a plate layout it
was never taught. That is a Law-2 situation: something else is compensating.

Cause: v5i plates and Roboflow `Charactor Detection` crops were rectified by
different conventions. v5i's human province boxes sit at **cy 0.85 / h 0.24**,
which matches **serve** (0.856 / 0.225), not the M2 training set (0.697 / 0.186).

### M3A — box-shape gap

| | median aspect | wider-than-tall |
|---|---|---|
| train | 0.59 | **0 %** |
| serve | 0.57 | **19 %** |

The model has never been asked for a wide box. `split_wide_char_boxes` and NMS
at serve are compensating for the missing training distribution. `--aug plate`
(now includes `mosaic_prob`/`translate`/`shear`) will produce some of it, but a
dedicated horizontal-scale jitter would be better; see §7.

### Classifiers

| dataset | classes | train | valid | native aspect | serve crop aspect | verdict |
|---|---|---|---|---|---|---|
| `thai_province_crops` (current root) | 77 | 9120 | 558 | 2.97 [1.96–5.12] | **3.23** [2.04–4.80] | ⚠️ built from **human** boxes |
| `recognition_v2/province_crops_v2` | 77 | 7804 | 1285 | 2.76 [1.97–4.15] | — | ✅ built through the **serve** M2 crop rules |
| `thai_character_crops/splits` | 50 | 29385 | 3612 | 1.00 (64×64) | 1.00 | ✅ |
| `recognition_v2/char_crops_v2` | 48 | 27395 | 3511 | 1.00 (64×64) | — | ✅ rebuilt, margin synced |

The serve province crop aspect (3.23) matches the letterbox tensor aspect
(256/80 = 3.20) almost exactly, so the letterbox adds almost no bars and almost
no distortion. That is why fixing `ENABLE_AMBIENT_CORNER_PAD` to `True` was
worth +5.8 points and why the contract test now asserts the letterbox.

**`recognition_v2` is the correct root for both classifiers.** `thai_province_crops`
was harvested from human-drawn boxes, which is precisely the mismatch
`extract_recognition_crops_v2.py` was written to eliminate.

⚠️ Both `thai_province_crops/valid` and `recognition_v2/province_crops_v2/valid`
have a long tail of classes with < 30 samples (49 and 72 of 77). The v2 set is
still larger overall but its per-class tail is worse — check
`scripts/balance_valid_v4.py` / `balance_v4.py` before trusting the val Top-1.

---

## 2b. The new `LPR 2 - Character Box Detection.coco` export

Downloaded 2026-10-02, 657 images, 3398 annotations, **COCO format**
(`train/valid/test/*.jpg` + one `_annotations.coco.json` per split).
Two categories: `0 = "LPR-2-Character-Box-Detection"` (Roboflow parent),
`1 = "Each Thai Charactor car plate"` (the real boxes).

### What it actually is

| question | answer |
|---|---|
| new images? | **no** — 100 % of its source stems exist in `LPR 2 - Character Box Detection.yolov11` (658 stems) |
| wider / squarer boxes? | **no** — box aspect 0.578 → 0.597 (+3.2 %), area/image 0.1144 → 0.1178 (+3.0 %) |
| wider-than-tall share? | **still 0 %** (serve measures **19 %**) |
| image aspect? | **3.72 vs serve 3.73 — the one real improvement** (old set: 3.77) |
| trainable as-is? | **no** — COCO layout, and `nc` would be 2 with the parent class |

So it is a **format re-export, not a re-annotation**. It fixes the image-aspect
alignment and nothing else. The wide-box gap that makes M3A under-call wide
characters at serve is **still open**.

### Convert it first

```bash
PY=/Users/kwankhaos/miniconda3/envs/thai-lpr/bin/python
$PY src/convert_coco_to_yolo.py --dry-run \
    --src "datasets/Thai/LPR 2 - Character Box Detection.coco" \
    --out "datasets/Thai/LPR_CharBox_New"

$PY src/convert_coco_to_yolo.py \
    --src "datasets/Thai/LPR 2 - Character Box Detection.coco" \
    --out "datasets/Thai/LPR_CharBox_New"
```

Expected: `train 390 / valid 106 / test 51` (547 of 657 — **110 images carry no
`category_id: 1` box and are skipped**; they are not negative samples, they are
unlabelled). `nc: 1`, `names: [char]`, which matches `TASKS["charbox"]`.

⚠️ 657 images is a small detector set, and it is the same images you already
have. Expect a retrain, not a breakthrough.

---

## 3. Prerequisite: rebuild M2 in the serve frame

```bash
cd "/Users/kwankhaos/Desktop/Personal Projects/Thai-License-Plate-Recognition-API-old"
PY=/Users/kwankhaos/miniconda3/envs/thai-lpr/bin/python

# dry run first — reports the resulting frame, writes nothing
$PY src/prepare_serve_frame_components.py --dry-run --limit 300
```

Expected (measured 900 images):

```
province cy  med=0.820   province h med=0.224   province aspect med=3.29
char-row cy  med=0.334   char-row h med=0.628   char-row aspect med=3.00

serve-measured targets: province cy 0.856 h 0.225 aspect 3.40 | char-row cy 0.419 h 0.531 aspect 3.22
old train frame:         province cy 0.697 h 0.186 aspect 5.72 | char-row cy 0.385 h 0.381 aspect 3.83
```

The **province** box is an exact remap of a real human annotation through the
exact serve warp, so it lands on target (`h` 0.224 vs 0.225). The **char-row**
box is *derived* from it (v5i has no character annotations) and lands at
cy 0.334 / h 0.628 against the serve 0.419 / 0.531 — close, but an approximation.
Treat that as the acknowledged weak point of this dataset.

Build it for real:

```bash
$PY src/prepare_serve_frame_components.py --out "datasets/Thai/LPR2_ServeFrame.yolov11"
```

**✅ BUILT 2026-10-03** — `datasets/Thai/LPR2_ServeFrame.yolov11`:
3219 images (train 2349 / valid 481 / test 389), 77 province classes, reusing
the existing v5i train/valid/test split (so the grouping is unchanged and no
new leakage is introduced). Verified after the build: labels match images
3219/3219, 0 empty label files, 0 stems in more than one split, `data.yaml`
absolute paths valid, `nc=2 names=[plate_char, province]`. Measured frame of
the built set vs serve: province `cy 0.821 / h 0.218 / aspect 3.37` against
serve `0.856 / 0.225 / 3.40` (old train `0.697 / 0.186 / 5.72`) — the frame
gap shrinks from 0.159 to 0.035 of plate height. Char-row stays
`cy 0.336 / h 0.632` vs serve `0.419 / 0.531` (derived-box caveat above).
§4.2 can run as written.

---

## 4. Retrain commands, in dependency order

All detectors: `--aug none` is the **control arm** (reproduces today's weights);
`--aug plate` is the retrain. Run both for any model you want to *claim* an
improvement from — an unpaired retrain proves nothing.

`--tag v6` never overwrites; `weights/<artifact>_<variant>_v6.{pt,onnx}`.

### 4.0 Dependency order — do not skip a link

```
4.6 corner regressor  ──►  4.1 M1 plate  ──►  4.2 M2 components  ──►  province
      (rectification)        (the crop)        (the two boxes)       crops + 4.4
                                                       │
                                                       └──►  4.3 M3A  ──►  char
                                                                          crops + 4.5
```

Two consequences that are easy to miss:

1. **The classifiers are downstream of the detectors.** `extract_recognition_crops_v2.py`
   crops with whatever `MODEL_2_FILENAME` / `MODEL_3A_FILENAME` currently point
   at. If you swap M2 for `v6frame`, `province_crops_v2` is now from the old
   detector and §4.4 trains the classifier on a distribution its detector no
   longer produces. Re-harvest `--only province` and re-split `--task province`
   before §4.4 whenever M2 changes.
2. **The corner regressor changes the plate frame for everything.** It is
   upstream of M1's crop and therefore of every measurement in §2. Train it
   first, then re-run `scratch/audit_retrain_datasets.py` — the "serve" column
   of that audit is produced through this same warp.

### 4.1 M1 — plate detector (PicoDet-S)

```bash
$PY src/train_libreyolo_task.py \
    --task plate --variant picodet_s \
    --imgsz 416 --epochs 35 --batch 16 --tag v6aug --aug plate
```

`--imgsz 416` is **mandatory**: `M1_PICODET` in `src/preprocess_registry.py` is
416 and the ONNX graph is pinned (`dynamic=False`). Measured recall at conf 0.35
over 60 images: 416 → 30.0 %, 640 → 6.7 %, 1280 → 26.7 %. **416 is correct; do
not "fix" it to 640.** The 3× LANCZOS4 upscale that was recovering small plates
is the crutch `--aug plate` (via `mosaic_prob=0.5`, `mosaic_scale=(0.3,1.5)`) is
meant to replace.

### 4.2 M2 — component detector (the important one)

```bash
$PY src/train_libreyolo_task.py \
    --task components --variant picodet_s \
    --data "datasets/Thai/LPR2_ServeFrame.yolov11" \
    --imgsz 416 --epochs 40 --batch 8 --tag v6frame --aug plate
```

Control arm for comparison (same code, wrong frame):

```bash
$PY src/train_libreyolo_task.py \
    --task components --variant picodet_s \
    --data "datasets/Thai/LPR 2 - Charactor Detection.yolov11" \
    --imgsz 416 --epochs 40 --batch 8 --tag v6ctl --aug plate
```

`v6ctl` vs `v6frame` is the clean measurement of the frame defect.

### 4.3 M3A — character box detector

**PicoDet is already the project's default for M1 and M2**, and the config
comment records PicoDet-S v2 as the *best measured* M3A on val
(P=0.81 / R=0.93 with NMS, dup ~0 %) — yet `MODEL_3A_FILENAME` still points at
`character_box_detector_dfine_small_v2.pt`. If PicoDet is the intent, use the
416 variant; the spec is picked automatically from the filename
(`get_m3a_spec` → `M3A_CHARBOX_PICODET`, tensor 416×416, aspect 3.73, unit norm).

```bash
# ── PicoDet-S for M3A  (imgsz 416 is MANDATORY — NOT 640)
$PY src/train_libreyolo_task.py \
    --task charbox --variant picodet_s \
    --data "datasets/Thai/LPR_CharBox_New" \
    --epochs 60 --batch 16 --patience 20 \
    --tag v6new --aug plate
```

`--imgsz` is omitted on purpose: `VARIANTS["picodet_s"]["imgsz"] = 416` already
matches `M3A_CHARBOX_PICODET`. Passing 640 would silently break the ONNX
contract. For reference, measured M1 recall at conf 0.35 over 60 images:
**416 → 30.0 %**, 640 → 6.7 %, 1280 → 26.7 % — PicoDet is a 416 model.

Switch it on after training:

```python
# src/config.py:51
MODEL_3A_FILENAME = "character_box_detector_picodet_s_v6new.pt"
```

`preprocess_registry.get_m3a_spec()` sees `"picodet"` in the name and returns
`M3A_CHARBOX_PICODET` automatically. **No registry edit needed** — this is the
one place where the contract is already wired.

Control arm (same code, old annotations):

```bash
$PY src/train_libreyolo_task.py \
    --task charbox --variant picodet_s \
    --data "datasets/Thai/LPR 2 - Character Box Detection.yolov11" \
    --epochs 60 --batch 16 --patience 20 \
    --tag v6ctl --aug plate
```

**Re-harvest the character crops with the NEW M3A before training the classifier.**
The crop distribution is a function of the boxes, so harvesting with the old
detector would put the classifier back on the old distribution.

⚠️ **Do NOT use `--overwrite` here.** It is not a "replace my crops" switch —
line 584 is `shutil.rmtree(recognition_v2/)`, i.e. the whole directory. That
directory also holds `ocr_crops/`, `ocr_gt_REVIEW_ME.csv` and
`manifest.gt_backup.csv`, which the harvester does **not** regenerate (only
`harvest_crops_generic.py` / `merge_harvest_v4.py` write those). You would lose
the OCR training set to re-extract a folder. Harvest into a scratch tree instead,
and keep `recognition_v2/` as the fallback:

```bash
# 1. re-harvest into a NEW tree (no rmtree anywhere)
$PY src/extract_recognition_crops_v2.py \
    --only char --charbox-dataset "datasets/Thai/LPR_CharBox_New"
#    (writes into recognition_v2/char_crops_v2/all + manifest.csv, leaving
#     ocr_crops/, manifest.gt_backup.csv and ocr_gt_REVIEW_ME.csv untouched)

# 2. re-split — source-image level, so no new leakage
$PY src/split_recognition_dataset_v2.py --task char

# 3. then §4.5 with --data-dir datasets/Thai/recognition_v2/char_crops_v2
```

If you *do* want a clean tree, move the directory aside first and restore the
three OCR artefacts afterwards — never let `--overwrite` do it.

⚠️ **Known limitation of the re-harvest.** The exact-count filter needs the GT
plate text (`thai_character_crops/metadata.csv`, 545 keys). The new charbox
export is 99.5 % covered (§0), but the crop set is still gated by the *old*
`MODEL_3A_FILENAME` for one more run — the harvester loads whatever
`cfg.ACTIVE_CHAR_BOX_MODEL_PATH` resolves to (`load_models`, line 134), so
**swap `MODEL_3A_FILENAME` to the `v6new` checkpoint before running this**, or
you will re-harvest with the detector you just replaced.

#### 4.3b D-FINE Small alternative (only if you keep D-FINE for M3A)

Retrain on the converted new annotations:

```bash
$PY src/train_libreyolo_task.py \
    --task charbox --variant dfine_small \
    --data "datasets/Thai/LPR_CharBox_New" \
    --imgsz 640 --epochs 60 --batch 8 --patience 20 \
    --tag v6new --aug plate
```

`--imgsz 640` is mandatory: `M3A_CHARBOX_DFINE` is 640 and
`character_box_detector_dfine_small_v2.onnx` is `[1,3,640,640]`. Keep the
control arm so the change is attributable:

```bash
$PY src/train_libreyolo_task.py \
    --task charbox --variant dfine_small \
    --data "datasets/Thai/LPR 2 - Character Box Detection.yolov11" \
    --imgsz 640 --epochs 60 --batch 8 --patience 20 \
    --tag v6ctl --aug plate
```

Same re-harvest rule as §4.3 applies if you take this arm instead — see the box
below it. Do **not** re-run it twice; one re-harvest is shared by both arms.

### 4.4 Province classifier

```bash
$PY src/train_grayscale_province_thai.py \
    --data-dir datasets/Thai/recognition_v2/province_crops_v2 \
    --epochs 25 --batch 32 \
    --save-name province_model_grayscale_thai_v6.pth \
    --patience 8
```

`--backbone` is **omitted on purpose**. Two reasons:

1. The current production checkpoint `province_model_grayscale_thai_v5.pth` is
   `backbone: 'resnet18'` with `n_classes: 77` (verified from the checkpoint).
   The R34 sibling `province_model_resnet34_grayscale_thai_v5.pth` exists with
   the same `val_top1: 1.0` and still lost the serve A/B — that is what
   `USE_RESNET34_PROVINCE_THAI = False` records. Changing backbone and data in
   one step makes the comparison uninterpretable.
2. `--data-dir datasets/Thai/recognition_v2/province_crops_v2` has **no `test/`
   split** (only `train/` + `valid/`). The trainer falls back to `val_ds` for
   test (`test_path.is_dir()` guard) and reports the valid number twice — fine,
   but know that the printed "test" accuracy is not an independent set.

⚠️ **`val_top1: 1.0` on this dataset is not a real score.** The v2 split is
leak-free at the *source image* level, but the valid tail is 72 of 77 classes
with < 30 crops (§2), so Top-1 saturates. Judge §4.4 by the end-to-end A/B
(`prov_ok` in §6), never by the printed validation accuracy.

Do **not** point this at `thai_province_crops` — that is the human-box
distribution. The resize contract is already correct: the trainer's private
`GrayscaleSmartResize` always letterboxes with the corner-median tone, and
`src/preprocess.py:GrayscaleSmartResize` mirrors it. Tensor `(256, 80)` must not
change.

If you specifically want the R34 arm, add `--backbone resnet34` and keep the
`resnet34` token in the save name; `ACTIVE_PROV_MODEL_THAI_PATH` picks the
backbone from the checkpoint's own `backbone` field, so the name is cosmetic.

### 4.5 Character classifier

```bash
$PY src/train_character_classifier.py \
    --data-dir datasets/Thai/recognition_v2/char_crops_v2 \
    --epochs 20 --batch-size 64 \
    --save-name character_classifier_v6.pth \
    --patience 6
```

Crops are 64×64 with corner-median square padding, `autocontrast(cutoff=2)` is
deterministic in `__getitem__` **and** at serve. Leave both alone.

### 4.6 Plate corner regressor — root cause of the rectification quality

This is what made the residual deskew look necessary. Train it before judging
§4.1–4.3.

⚠️ **This script has no command-line interface.** There is no `argparse` block —
`--epochs`/`--device` are silently ignored, and `__main__` calls
`train_corner_regressor(epochs=25, batch_size=32, lr=1e-3)` with the values
hardcoded. Worse, the checkpoint path is a module constant:
`SAVE_PATH = WEIGHTS_DIR / "plate_corner_regressor.pth"` — the same file serve
loads via `cfg.PLATE_CORNER_MODEL_PATH`. Running it **overwrites the live
rectifier** on the first improving epoch, and there is no `--tag` to protect you.

Edit the two call-site values, and back up the current checkpoint first:

```bash
cp weights/plate_corner_regressor.pth weights/plate_corner_regressor_v5.pth.bak
```

```python
# src/train_plate_corner_regressor.py, __main__ (last 2 lines)
train_corner_regressor(epochs=40, batch_size=32, lr=1e-3)
```

```bash
$PY src/train_plate_corner_regressor.py
```

Input dataset `datasets/Thai/LPR 2 - Polygon.yolov11_new` exists (1068 train /
305 val corner polygons). Restore with
`cp weights/plate_corner_regressor_v5.pth.bak weights/plate_corner_regressor.pth`
if the new one loses on `scratch/diag_residual_deskew.py`.

Confirm afterwards with:

```bash
$PY scratch/diag_residual_deskew.py     # the |angle| histogram must shrink toward 0
```

If it does, `ENABLE_RESIDUAL_DESKEW` can stay `False` permanently. If it does not,
the corner regressor is the thing still wrong.

---

## 5. Registry / config edits each retrain requires

⚠️ **Edit the `.pt` name, not the `.onnx` name.** `src/api_server.py` has **zero**
references to `.onnx` (verified by grep) — it loads `.pt` checkpoints through
`ACTIVE_MODEL_1_PATH` / `ACTIVE_MODEL_2_PATH` / `ACTIVE_CHAR_BOX_MODEL_PATH`.
The `.onnx` export is for the C# deployment described in the README. Pointing
`MODEL_1_FILENAME` at a `.onnx` would make the file "exist", so the fallback
chain would never fire, and the loader would be handed a graph it cannot open.

| Retrain | Edit needed | Where |
|---|---|---|
| M1 `--tag v6aug` | `MODEL_1_FILENAME` → `plate_detector_picodet_s_v6aug.pt` | `src/config.py:28` |
| M2 `--tag v6frame` | `MODEL_2_FILENAME` → `component_detector_picodet_s_v6frame.pt` | `src/config.py:36` |
| M3A `--tag v6new` (PicoDet) | `MODEL_3A_FILENAME` → `character_box_detector_picodet_s_v6new.pt` | `src/config.py:51` |
| Province §4.4 | `MODEL_3B_THAI_FILENAME` → `province_model_grayscale_thai_v6.pth`; leave `USE_RESNET34_PROVINCE_THAI = False` | `src/config.py:76-77` |
| Char §4.5 | `CHAR_CLASSIFIER_THAI_FILENAME` → `character_classifier_v6.pth` | `src/config.py` |
| **none** | `preprocess_registry.py` — **do not touch**. The imgsz contract is satisfied as written. | `src/preprocess_registry.py` |

Two resolution rules that decide whether an edit works:

- `ACTIVE_MODEL_1_PATH` checks `WEIGHTS_DIR / MODEL_1_FILENAME` **first**, and only
  falls through to the legacy chain when that file is missing. A typo therefore
  does not crash — it silently serves an older checkpoint. Confirm with the
  startup log (`model_1` line at `api_server.py:3196`) after every edit.
- The M3A spec is selected from the **filename** (`get_m3a_spec` → any name
  containing `picodet` → `M3A_CHARBOX_PICODET`, tensor 416). Switching M3A from
  `dfine_small` to `picodet_s` therefore changes the ONNX input size **and** the
  normalization together, with no registry edit — and `m3a_conf` is also
  filename-dependent (`0.22` for picodet vs `0.10` otherwise, `api_server.py:1276`).
  The C# side must be updated to match (README §"ONNX spec").

### ⚠️ The fallback chain changes ARCHITECTURE, not just the file

Simulated against the real `weights/` directory (2026-10-03). A wrong filename
does not raise — it silently serves a different model **family**:

| config says | `ACTIVE_*` resolves to | what changed |
|---|---|---|
| `MODEL_1_FILENAME = "plate_detector_picodet_s_v6aug.pt"` (missing) | `plate_detector_rfdetr_small.pt` | PicoDet 416 → **RF-DETR 560** |
| `MODEL_3B_THAI_FILENAME = "province_model_grayscale_thai_v6.pth"` (missing) | `province_model_grayscale_thai.pth` | **R18 v5 → R18 v1** (Sep 17) |

Both destinations exist on disk, so both fallbacks are live. The province case is
the dangerous one: the file it lands on is *named* like the v5 checkpoint, so the
startup log shows a plausible name and the only clue is the `PROV_MODEL_THAI_TAG`
string. **Always confirm the log line, never the filename you typed.**

`--save-name` must therefore be spelled exactly as the config edit:

```python
# src/train_grayscale_province_thai.py --save-name province_model_grayscale_thai_v6.pth
#   -> weights/province_model_grayscale_thai_v6.pth        (write it exactly like this)
```

One more consequence for §4.4: `province_model_resnet34_grayscale_thai_v5.pth`
**already exists** (`backbone: resnet34`, `val_top1: 1.0`). The R34 arm has been
tried on this same v5 data and lost the serve A/B despite tying on validation —
which is the `USE_RESNET34_PROVINCE_THAI = False` comment in action. Re-running
the R34 arm on the same distribution is a repeat of a settled experiment; only
the *data* change (re-harvest through M2 `v6frame`) is new.

`M2_COMPONENTS_PICODET.train_aspect_w_over_h = 2.0` stays correct: the new
dataset is warped to exactly 320×160 = 2.00, which is *better* aligned than the
old 1.94 crops.

Nothing in `src/api_server.py` changes for the retrain. The only serve-side
change already made is `ENABLE_RESIDUAL_DESKEW = False`.

---

## 6. Validate each model before accepting it

```bash
# detector geometry + leakage + frame — always re-run
$PY scratch/audit_retrain_datasets.py

# end-to-end A/B, paired, on the same 120 images.
# AB_MODES must name modes that EXIST in scratch/ab_flag_matrix.py — a new
# checkpoint is tested by the mode you are already on, not by a new mode name.
FORCE_CPU=True AB_N=120 AB_MODES=K2_default \
    $PY scratch/ab_flag_matrix.py
```

⚠️ `ab_flag_matrix.py` toggles `Config.ENABLE_*` **flags**, not checkpoint
filenames. So the paired comparison for a retrain is:

1. Run `AB_MODES=K2_default` with `MODEL_x_FILENAME` = the **old** checkpoint →
   save the CSV.
2. Edit `MODEL_x_FILENAME` to the new `_v6` checkpoint (the only config edit).
3. Run the identical command again and diff the two `prov_ok` columns.

`K2_default` is the shipped default and must reproduce `A_current`; if it does
not, the code moved and the baseline is stale. Available modes (verified in
`scratch/ab_flag_matrix.py:58`): `A_current`, `B_deskew_off`, `C_ambient_on`,
`D_autocon_off`, `E_gray_off`, `F_pad_on`, `G_pad_on_deskew_off`, `H_pure_models`,
`I_chain_no_heuristics`, `J_rectify_only`, `K_residual_on`, `K2_default`,
`B2_no_rectify`. `scratch/ab_flag_matrix_results.csv` is append-only and
resumable, so both arms land in the same file — add an `AB_TAG` column note by
hand or diff on the row range.

Acceptance bar, from `PREPROCESSING_CONTRACT.md` §6: a 120-image sample has a
±4 point margin. **Only accept a change worth ≥ 8 points**, or re-measure on a
larger set. And remember the ground truth: v5i carries **province labels only**,
so `prov_ok` is a real metric and every `plate_text` column is only a change
counter.

---

## 7. What this retrain does **not** fix

- **No character-level ground truth.** Every character claim in this document is
  "N predictions changed", not "N are now correct". Building a labelled subset is
  the highest-value next task; without it the char models cannot be judged.
- **27 % of plates are never rectified** (no quad → AABB crop). Both branches
  feed the recognisers simultaneously, so "the plate crop" is still not a
  single-valued contract. Retraining does not change that; only fixing the
  corner regressor's coverage does.
- **The M3A wide-box gap** is only partly addressed: `mosaic_prob` + `translate`
  produce some horizontal variation, but nothing guarantees the 19 % wide-box
  distribution appears. A dedicated horizontal-scale augmentation would be the
  honest fix.
- **`recognition_v2` crops come from 192×99 v5i plates, not through the serve
  warp**, so they lack the 6 % background ring that
  `warp_perspective_plate(padding_frac=0.06)` adds at serve. Small, but real. A
  `--serve-frame` mode on `extract_recognition_crops_v2.py` would close it;
  the province `h` measurement says it is not currently costing anything.
- **Only the province box in the new M2 set is exact.** The char-row box is
  derived, not annotated (§3).
- **`upscale_to_train_geometry` is still a geometric no-op** (`BOX_GEOMETRY_AUDIT.md` §4).

---

## 8. The verdict: "just match the size" is NOT enough

This is the direct answer to *"ถ้า retrain จะได้ผลดีขึ้นไหม และต้องเขียน preprocess เยอะไหม — แค่เตรียมขนาดให้ตรงพอหรือเปล่า?"*

**Matching the input size is already done — and it is not the problem.**

| stage | train size | serve size | verdict |
|---|---|---|---|
| M1 plate | 416 (`LPR_2_BBox`) | 416 | ✅ matched |
| M2 components | 416, aspect 1.94 | 416, aspect 2.00 | ✅ 3 % — matched |
| M3A charbox | 640 (D-FINE) / 416 (PicoDet) | 640 / 416 | ✅ matched |
| province cls | (256, 80) letterbox | (256, 80) letterbox | ✅ matched **after** the `ENABLE_AMBIENT_CORNER_PAD=True` fix |
| char cls | 64×64 square pad | 64×64 square pad | ✅ matched |

Every tensor size already agrees. What does **not** agree is the *content* of
those tensors, and no resize can fix content:

| defect | kind | fixed by resizing? | fixed by retraining? |
|---|---|---|---|
| M2 province box sits at `cy` 0.697 in train, 0.856 at serve | **annotation frame** | ❌ | ✅ §3 + §4.2 |
| province letterbox vs stretch | resize geometry | — | already fixed in `config.py` (+5.8 pt) |
| M3A has never seen a wide box (train 0 %, serve 19 %) | **augmentation** | ❌ | ⚠️ partly (`--aug plate`) |
| zero geometry/photometric augmentation (LibreYOLO defaults) | **augmentation** | ❌ | ✅ `--aug plate` |
| `fine_deskew` ±8° at serve, never in training | **augmentation** | ❌ | ✅ `degrees=6` |
| 3× LANCZOS4 upscale to recover small plates | **augmentation** | ❌ | ✅ `mosaic_prob=0.5` |
| CLAHE recovery path | **augmentation** | ❌ | ✅ `hsv_prob=0.5` |

So: **the answer is "no, size alone is not enough", but the work is also not
"write more preprocessing" — it is the opposite.** The retrain is what lets you
*delete* preprocessing. The measured evidence, from the 120-image A/B matrix:

| mode | province | what it proves |
|---|---|---|
| `A_current` (correct flags) | **112/120 = 93.3 %** | baseline |
| `I_chain_no_heuristics` (regex/fallback/reconciliation/CLAHE/stroke **off**) | **112/120 = 93.3 %** | province identical 120/120 — the heuristics buy **nothing** once geometry is right |
| `H_PURE_MODEL_MODE` (all switches off) | 104/120 = 86.7 % | the model alone is not enough **as currently trained** |

Read those three rows together: heuristics contribute **0 points** on top of a
correct geometry contract, but stripping the contract (H) costs **6.6 points**.
The contract is the letterbox + autocontrast + crop rules — cheap, already
written, and verified by 46 contract tests. The heuristics are what a retrain
with `--aug plate` is meant to make removable.

### What to do, in order

1. **Do not write new preprocessing.** §5 shows the only edits are filenames in
   `config.py`. `preprocess_registry.py` is explicitly "do not touch".
2. **Do not retrain everything.** The province classifier is already at 93.3 %
   end-to-end and its tensor geometry was the actual bug — that is fixed.
   Retraining it is the lowest-value item on the list.
3. **Do retrain M2 in the serve frame** (§3 + §4.2). That is the single largest
   *train/serve contract* violation left: a 16 %-of-plate-height frame shift.
4. **Do retrain M1/M3A with `--aug plate`** — not for accuracy, but to remove
   the three crutches (§1). Without this, `fine_deskew`, the 3× upscale and
   CLAHE must stay.
5. **Always run the paired control arm.** A 120-image sample has a ±4 pt margin;
   accept nothing under 8 pt without a bigger set.

### Honest expected magnitude

The only *measured* frame fix so far is the province letterbox: **+5.8 pt**
(87.5 % → 93.3 %). The M2 frame rebuild is the same class of defect and is the
best candidate for a comparable gain. The augmentation retrains are a different
bet: they are expected to *hold* accuracy while making three serve-side patches
delectable, which is a maintainability win rather than a headline accuracy win.
No one can promise the second kind — measure it with the control arm.
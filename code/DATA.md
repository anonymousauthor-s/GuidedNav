# Data and teacher preparation

The frozen final-stage annotations, sample-selection arrays, object masks and agent poses have been prepared locally but are not publicly hosted yet. They contain 157,465 selected R2R rows and 472,323 selected RxR rows (629,788 total), not the full Guided-R2R/Guided-RxR datasets. A separate annotations-only release candidate is being reviewed for upstream licensing and distribution requirements. Scene assets and RGB observations are separate: obtain Matterport3D access under its official terms. Source code alone is not sufficient for training; exact final-stage reproduction also needs the frozen labels and collected poses, whose public availability must be checked on the release page.

## 1. Obtain official benchmark assets

Follow the [VLN-CE data setup](https://github.com/jacobkrantz/VLN-CE#data) for R2R episodes and MP3D scene preparation, the [official RxR-Habitat data instructions in VLN-CE](https://github.com/jacobkrantz/VLN-CE#episodes-room-across-room-rxr) for RxR guide episodes and ground-truth trajectories, and the [Matterport3D access instructions](https://niessner.github.io/Matterport/) for scene assets. Download through the upstream access process; this release does not replace those permissions.

For evaluation, arrange paths relative to the **release root**:

```text
data/scene_datasets/mp3d/<scan>/<scan>.glb
data/datasets/r2r/val_unseen/val_unseen.json.gz
data/datasets/rxr/val_unseen/val_unseen_guide.json.gz
data/datasets/rxr/val_unseen/val_unseen_guide_gt.json.gz
```

Keep the matching dataset/ground-truth files from the same upstream release. The evaluator's Habitat extensions register RxR trajectory metrics before configuration loading. Habitat environment settings are in `config/vln_r2r.yaml` and `config/vln_rxr.yaml`: RGB 640×480, horizontal FOV 79°, forward action 0.25 m, turn action 15°, maximum 500 steps. The camera uses Habitat 0.2.4's default position `[0,1.25,0]` metres and zero relative rotation; these defaults are also explicit in the pose renderer. See the [upstream sensor definitions](https://github.com/facebookresearch/habitat-lab/blob/v0.2.4/habitat-lab/habitat/config/default_structured_configs.py).

The final RxR evaluation uses `RxREnglishGuideVLNCE-v1`, retaining all 1223 en-US and 2446 en-IN guide episodes (3669 total). Its simulator is `RxRFullGeometrySim-v1`, collision height 0.88 m and radius 0.18 m, with sliding enabled. The research RGB/action protocol differs from the RxR challenge configuration. Rebuild and audit its explicitly pinned navigation meshes without modifying the original scenes:

```bash
python tools/build_rxr_navmesh.py \
  --dataset ../data/datasets/rxr/val_unseen/val_unseen_guide.json.gz \
  --gt ../data/datasets/rxr/val_unseen/val_unseen_guide_gt.json.gz \
  --scenes-root ../data/scene_datasets --output ../data/rxr_navmesh
```

This CPU geometry tool creates a renderer-free simulator, applies default `NavMeshSettings` with height/radius 0.88/0.18, saves each scene's new mesh, and checks finite start-to-goal reachability plus valid reference trajectories for every English episode. It requires the exact benchmark data SHA256 `93abddce58e8232867a8738daa8107eb82953693a9d5e70779d1eb00386b222a` and GT SHA256 `bdfbb624132501bf297eb0ea16a99e38682668ba5e9ce1554ca5cdcde6b65a0d`. A different or recompressed file must be reconciled with the original benchmark payload before claiming the same protocol. `audit.json` and `*.navmesh` stay together; the evaluator verifies each mesh hash and agent dimensions. `scripts/evaluate.sh rxr` sets the audit/dataset/GT environment paths automatically. Override `RXR_NAVMESH_AUDIT_PATH`, `RXR_EVAL_DATA_PATH`, `RXR_EVAL_GT_PATH` and `GUIDEDNAV_SCENES_DIR` when using a different local layout.

## 2. Render the released training poses

The frozen final-stage package preserves the original per-video trajectory NPZ files. They store world-space **agent** positions and rotations in **w,x,y,z** order, with one pose per observation. Their filenames match the video basename in each annotated row. `tools/render_training_rgb.py` converts to Habitat's x,y,z,w coefficients and reapplies the camera offset once. It does not rerun a shortest-path follower or sample a different trajectory.

In the simulator environment, from `code/`:

```bash
python tools/render_training_rgb.py \
  --trajectories ../data/r2r/traj \
  --scenes-root ../data/scene_datasets \
  --output ../data/r2r/rgb
python tools/render_training_rgb.py \
  --trajectories ../data/rxr/traj \
  --scenes-root ../data/scene_datasets \
  --output ../data/rxr/rgb
```

Use `--rank i --world-size N --gpu j` to shard videos across rendering workers, and `--limit 1` to inspect one trajectory first. The renderer writes `images/<video>/rgb/001.jpg`, matching recorded relative image paths. It renders every pose of each selected trajectory, covering uniform-history images and the three local predecessor frames required for temporal fusion. Existing frames are retained. Use a fresh output directory when changing simulator/sensor settings.

The source collection code wrote RGB JPEGs using Pillow's default JPEG settings. The renderer uses the same save operation, but byte-for-byte RGB reproduction has not been verified during packaging; renderer, driver and JPEG library differences can change pixels. Use original RGB assets and their integrity records when available. Before training, inspect a rendered frame and its released object mask at the same frame index. This is especially important because the masks supervise spatial attention on the rendered current observation.

## 3. Verify and relocate frozen training data

The final stage selects 157,465 of 629,862 R2R source rows and 472,323 of 1,889,294 RxR source rows: 629,788 total. Sampling is row based, seed 42, without replacement, using annotation-complete candidates. It is not a random 25% of trajectories. The candidate snapshot is fixed; later completed exports must not alter the selection.

Each source has JSONL records and seven `.npy` arrays: `offsets` (byte offsets), `source_indices` (original row identities), `spans` (subtask character bounds), `phases`, `groups`, `weights` and `o_eligible`. S weights are balanced across selected trajectory phases and normalized within each source. R2R and RxR both use their released object masks. No visible-object mask means missing supervision, not a negative object target.

Use the anonymous data package's manifest, not a newly sampled dataset. After extraction, relocate and verify it:

```bash
python tools/prepare_data_manifest.py \
  --manifest ../data/manifest.json --array-root ../data \
  --r2r-jsonl ../data/r2r/samples_perstep.jsonl \
  --rxr-jsonl ../data/rxr/samples_perstep.jsonl \
  --r2r-rgb ../data/r2r/rgb --rxr-rgb ../data/rxr/rgb \
  --r2r-masks ../data/r2r/masks --rxr-masks ../data/rxr/masks \
  --output ../data/manifest.local.json
```

If the data package uses different JSONL or mask subdirectory names, pass those actual locations. Mask roots must be the directories against which the row's relative `object_supervision.active_masks[].mask` paths resolve. The anonymous export preserves original selected-row identities and targets while replacing private paths and, when compacting the JSONL, recalculating byte offsets. Its manifest records the original full-manifest SHA256 `cf78986d6a58f22be9bacc707e4b3d7f630d60940e2a7acc50e7d7cfddceb33f` in export provenance. Its new manifest/JSONL/offset-array byte hashes therefore differ from the private source; that is not a different row selection. The relocation tool verifies the anonymous export's immutable JSONL and array hashes, updates filesystem modification times, and writes a new manifest without modifying the frozen reference. Extracting/repacking changes file mtimes, so pointing training directly at the reference manifest can intentionally fail its integrity checks.

In the training environment, decode and collate examples without loading the 8B model:

```bash
export PYTHONPATH="$PWD/src:$PWD/third_party/vggt"
python tools/smoke_dataset.py --manifest ../data/manifest.local.json \
  --processor ../weights/initialization-6656
```

This checks positive S/O examples from both sources, image decoding, tokenizer span mapping, object-mask pooling, temporal predecessor frames, VGGT image preprocessing and collation. Run the two-update gradient smoke in README.md before a formal distributed run.

## Annotation schema and provenance

Rows contain `instruction`, chronological relative `images` (history followed by one current frame), `n_memory`, `conversations`, `action_text`, current step `c`, `subtask`, `subtask_phase_index`, and `object_supervision`. The latter records `mask_loss_eligible` and `active_masks`, whose entries retain mask filenames, hashes, frame/phase identities and pipeline metadata. RxR rows additionally retain their original instruction identity. Exact character spans and group weights come from the sidecars; do not approximate them by a different tokenizer search.

The data loader sanitizes legacy subtask-answer text into an action-only policy target. Ground-truth subtasks, character spans and object masks supply training losses only. They are not included in the inference prompt. The current-frame spatial-forcing weight is 0.5, temporal-difference weight 0.1, native subtask/object weights 0.01 each, with a 0.03 auxiliary warmup ratio.

`tools/regenerate_expert25.py` preserves the selected run's sampling/export logic with explicit path arguments. It can regenerate the selection **if** the original full-source indexed R2R manifest/arrays, frozen RxR completed-case snapshot and schema-v2 completed exports are available:

```bash
python tools/regenerate_expert25.py \
  --prior-full-data /path/to/full-indexed-data \
  --rxr-completed-exports /path/to/completed-schema-v2-exports \
  --completed-snapshot /path/to/frozen/rxr_completed_snapshot.json \
  --output /path/to/new-selection
```

Those upstream annotation-production inputs are not required to train from the already frozen released data package. This script does not regenerate subtask/object labels by invoking a different VLM or segmenter, which could produce different labels. The released labels and masks define the reproducible final-stage targets.

For collecting **new** expert datasets, `src/data_process/gt_following_rollout.py` and `build_vln_samples.py` expose `--help`; the latter uses memory size 8, four future actions and seed 42. These generic collection tools are not a substitute for the released original poses or frozen annotation targets when reproducing final-2461.

## 4. Obtain the exact training teacher

The vendored `third_party/vggt/vggt` source is the teacher implementation used by the final stage. Its source hashes and model hash are in `config/vggt_reference.json`. The original snapshot had no recorded Git commit, so the included file hashes identify its source directly.

The exact 5,026,874,952-byte teacher matches the [official VGGT-1B model payload](https://huggingface.co/facebook/VGGT-1B/blob/main/model.pt), SHA256 `d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0`. After reviewing upstream terms, download it with:

```bash
python tools/download_teacher.py --output ../assets/vggt/vggt.pt
```

The downloader verifies size and SHA256 before completing the file. The teacher is frozen and used only in training. It is not needed for inference or benchmark rollout.

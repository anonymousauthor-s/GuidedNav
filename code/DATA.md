# Data and teacher preparation

> **Upload in progress.** Model and data download links will become active after all files pass the public-download checks.


The complete selected Stage 3 annotation, trajectory and prompt-metadata add-on
is being uploaded to the [public release](https://github.com/anonymousauthor-s/GuidedNav/releases/tag/stage3-epoch1). From `code/`, download it with:

```bash
python tools/download_assets.py --group data
```

It installs `annotations/`, `data/r2r/traj/`, `data/rxr/traj/`,
`data/pose-manifest.json` and `prompt-metadata/` at the repository root.
Combine these files with official benchmark instructions and licensed
Matterport3D scenes to reconstruct the training inputs. The package supplies
exact collected poses/actions, aligned labels/masks, frozen selection arrays and
per-row prompt choices. RGB is rendered locally. See the [model guide](https://anonymousauthor-s.github.io/GuidedNav/guide/models/)
for the full weights and [data terms](https://github.com/anonymousauthor-s/GuidedNav/releases/tag/stage3-epoch1) for attribution and upstream conditions.

The stage-3 epoch uses 157,465 R2R rows and 472,323 English RxR rows: 629,788 selected observations from 18,656 collected trajectories. This is the exact training selection, rather than the total size of Guided-R2R/Guided-RxR. The model directories are `navigation-base`, `stage3-initialization` and `stage3-epoch1`; see README.md for training and inference.

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

## 2. Match the addon to public instructions

The correspondence is explicit and has been checked across the **entire stage-3 selection**:

| Source | Selected rows | Unique instructions | Exact upstream matches |
| --- | ---: | ---: | ---: |
| R2R | 157,465 | 10,703 | 10,703 / 10,703 |
| English RxR | 472,323 | 7,953 | 7,953 / 7,953 |

R2R joins `fgr_path_id` to the public FGR2R `path_id`, then identifies the original instruction by its exact UTF-8 SHA256. The processed `ins_id` is **not** assumed to be the index among the original three instructions. RxR joins the original `rxr_instruction_id`, again checking the exact text hash. No whitespace normalization is applied; trailing spaces matter. The video identity and zero-based frame `c` connect labels and masks to the corresponding collected agent pose. Character spans are zero-based, half-open Unicode character bounds, not tokenizer offsets; `[-1,-1]` denotes unavailable span supervision.

Obtain `FGR2R_train.json` from the [official FGR2R data directory](https://github.com/YicongHong/Fine-Grained-R2R/tree/master/data) and `rxr_train_guide.jsonl.gz` using the [official RxR guide-annotation instructions](https://github.com/google-research-datasets/RxR#downloading-guide-annotations). The verified public payload SHA256 values are:

```text
FGR2R_train.json           703837ee87a14d1b1fcb7d5bd6ce261faacc82e0d911a16396792cfa53d54fbc
rxr_train_guide.jsonl.gz   b28b374b6e3203874f8bf65293df102b698efd51109e7e81a0e86fe3e3f780f2
```

With the addon extracted locally, verify all pointers without downloading any data inside the tool:

```bash
python tools/verify_annotation_alignment.py \
  --annotations ../annotations \
  --r2r-source /path/to/FGR2R_train.json \
  --rxr-source /path/to/rxr_train_guide.jsonl.gz \
  --report ./alignment-report.json
```

The complete addon needs the following information. It does not need to duplicate full upstream instructions or scene imagery:

| Information | Purpose |
| --- | --- |
| Public instruction ID/hash, video/frame identity | Recover the correct original text and observation index |
| Exact collected agent poses and actions, with per-file hashes | Render the original observation sequence and recover the four-action targets |
| Masks, character spans, phases, groups, weights and eligibility | Preserve the added supervision |
| Frozen selected `source_indices` | Preserve the exact stage-3 examples and their order |
| Per-row `conjunction_id` and its fixed seven-string vocabulary | Reproduce the original action prompt without drawing a new random choice |
| Renderer/sensor and sample-building settings | Preserve camera geometry, history sampling and action indexing |

The conjunction IDs require one byte per selected row, plus small headers. The current action-training loader does not use stored waypoint coordinates or distances as targets. It consumes the recovered instruction, image/history paths, action-only conversation and the S/O supervision. Legacy subtask-answer text is removed before training, so its original string does not have to be redistributed.

`tools/materialize_annotation_addon.py` reconstructs these loader records from the public text and the complete addon. It never reruns an expert follower, resamples examples or regenerates labels:

```bash
python tools/materialize_annotation_addon.py \
  --annotations ../annotations \
  --r2r-source /path/to/FGR2R_train.json \
  --rxr-source /path/to/rxr_train_guide.jsonl.gz \
  --trajectories ../data --pose-manifest ../data/pose-manifest.json \
  --prompt-metadata ../prompt-metadata \
  --r2r-rgb ../data/r2r/rgb --rxr-rgb ../data/rxr/rgb \
  --output ../stage3-training-data
```

The output is a new local `manifest.local.json` and compatible JSONL/array files; it does not alter the input artifacts. Full-selection verification matched the original action labels, history-frame paths, normalized action prompts and mask pointers for all 629,788 observations. This verifies data correspondence and input reconstruction, rather than a new GPU training run or a byte-identical RGB-rendering result.

## 3. Render training observations from collected poses

The add-on provides the original per-video collected positions, rotations and actions in NPZ files. Unused scene semantics and reference-path fields are omitted. The files store world-space **agent** positions and rotations in **w,x,y,z** order, with one pose per observation, as well as the recorded discrete actions. Their filenames match the video basename in each annotated row. All 18,656 selected trajectories were checked against the training observation/action indices. `tools/render_training_rgb.py` converts to Habitat's x,y,z,w coefficients and reapplies the camera offset once. It does not rerun a shortest-path follower or sample a different trajectory.

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

## 4. Verify and relocate frozen training data

The final stage selects 157,465 of 629,862 R2R source rows and 472,323 of 1,889,294 RxR source rows: 629,788 total. Sampling is row based, seed 42, without replacement, using annotation-complete candidates. It is not a random 25% of trajectories. The candidate snapshot is fixed; later completed exports must not alter the selection.

Each source has JSONL records and seven `.npy` arrays: `offsets` (byte offsets), `source_indices` (original row identities), `spans` (subtask character bounds), `phases`, `groups`, `weights` and `o_eligible`. S weights are balanced across selected trajectory phases and normalized within each source. R2R and RxR both use their frozen object masks. No visible-object mask means missing supervision, not a negative object target.

Use the add-on's frozen selection. The materializer in Step 2 writes
`stage3-training-data/manifest.local.json`, reconstructed JSONL records and their
array sidecars. Its hashes and offsets refer to those newly written local files.
No additional manifest relocation is needed. Recreate the local manifest after
moving or changing the training inputs.

In the training environment, set `FULL_MANIFEST` to the actual materialized or relocated manifest, then decode and collate examples without loading the 8B model:

```bash
export PYTHONPATH="$PWD/src:$PWD/third_party/vggt"
export FULL_MANIFEST=../stage3-training-data/manifest.local.json
python tools/smoke_dataset.py --manifest "$FULL_MANIFEST" \
  --processor ../weights/stage3-initialization
```

This checks positive S/O examples from both sources, image decoding, tokenizer span mapping, object-mask pooling, temporal predecessor frames, VGGT image preprocessing and collation. Run the two-update gradient smoke in README.md before a formal distributed run.

## Annotation schema and provenance

Materialized action-training rows contain `instruction`, chronological relative `images` (history followed by one current frame), `n_memory`, `conversations`, `action_text`, current step `c`, `subtask_phase_index`, and `object_supervision`. The latter records `mask_loss_eligible` and `active_masks`, including mask filenames/hashes and frame identities. Public instruction IDs and original selected-row identities remain in the addon. Exact character spans and group weights come from the sidecars; do not approximate them by a different tokenizer search. The inherited R2R subtask lineage is FGR2R; the added object grounding/mask labels are automatically generated derived annotations.

The data loader sanitizes legacy subtask-answer text into an action-only policy target. Ground-truth subtasks, character spans and object masks supply training losses only. They are not included in the inference prompt. The current-frame spatial-forcing weight is 0.5, temporal-difference weight 0.1, native subtask/object weights 0.01 each, with a 0.03 auxiliary warmup ratio.

`tools/regenerate_expert25.py` preserves the selected run's sampling/export logic with explicit path arguments. It can regenerate the selection **if** the original full-source indexed R2R manifest/arrays, frozen RxR completed-case snapshot and schema-v2 completed exports are available:

```bash
python tools/regenerate_expert25.py \
  --prior-full-data /path/to/full-indexed-data \
  --rxr-completed-exports /path/to/completed-schema-v2-exports \
  --completed-snapshot /path/to/frozen/rxr_completed_snapshot.json \
  --output /path/to/new-selection
```

Those annotation-production inputs are not required when reconstructing from the complete frozen addon. This script does not regenerate subtask/object labels by invoking a different VLM or segmenter, which could produce different labels. The frozen labels and masks define the reproducible stage-3 targets.

For collecting **new** expert datasets, `src/data_process/gt_following_rollout.py` and `build_vln_samples.py` expose `--help`; the latter uses memory size 8, four future actions and seed 42. These generic collection tools are not a substitute for the exact collected poses, recorded prompt choices or frozen annotation targets when reproducing `stage3-epoch1`.

## 5. Obtain the exact training teacher

The vendored `third_party/vggt/vggt` source is the teacher implementation used by the final stage. Its source hashes and model hash are in `config/vggt_reference.json`. The original snapshot had no recorded Git commit, so the included file hashes identify its source directly.

The exact 5,026,874,952-byte teacher matches the [official VGGT-1B model payload](https://huggingface.co/facebook/VGGT-1B/blob/main/model.pt), SHA256 `d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0`. After reviewing upstream terms, download it with:

```bash
python tools/download_teacher.py --output ../assets/vggt/vggt.pt
```

The downloader verifies size and SHA256 before completing the file. The teacher is frozen and used only in training. It is not needed for inference or benchmark rollout.

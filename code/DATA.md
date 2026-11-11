# Data and teacher preparation

## Data for the three training stages

Prepare the following sources for the [three training stages](TRAINING.md).

| Stage | Navigation source | Visual instruction source |
| --- | --- | --- |
| 1 | R2R and English RxR expert rollouts | LLaVA-Video + VSI |
| 2 | Stage 1 policy rollouts relabeled by an expert (DAgger) | The same frozen LLaVA-Video + VSI selection |
| 3 | Annotated R2R and English RxR expert observations | None |

Obtain official training episodes and scenes through [VLN-CE](https://github.com/jacobkrantz/VLN-CE#data)
and [Matterport3D](https://niessner.github.io/Matterport/). Download visual
instruction annotations and their media from the official
[LLaVA-Video-178K](https://huggingface.co/datasets/lmms-lab/LLaVA-Video-178K)
and [VSI-590K](https://huggingface.co/datasets/nyu-visionx/VSI-590K) repositories.
The paper uses 495,164 LLaVA-Video and 295,333 VSI examples in the VQA mixture.
The example configuration selects those counts deterministically from your
local input pool. It saves the selected indices for reuse.

### Collect Stage 1 expert navigation data

In the Habitat environment, prepare training-only episode files and set
`GUIDEDNAV_SCENES_DIR` to the official scene root. From `code/`, an R2R example is:

```bash
export GUIDEDNAV_SCENES_DIR=/absolute/path/to/data/scene_datasets
python src/data_process/gt_following_rollout.py \
  --dataset r2r --split train --config_path config/vln_r2r_train.yaml \
  --data_path /absolute/path/to/r2r/train/train.json.gz \
  --output_path ../data/expert/r2r --sim_gpu 0 \
  --follow_to_goal --goal_stop_dist 0.5
python src/data_process/gt_following_rollout.py \
  --output_path ../data/expert/r2r --merge
python src/data_process/build_vln_samples.py \
  --dataset_dir ../data/expert/r2r --memory_size 8 --num_future_steps 4 --seed 42
python src/data_process/build_vln_samples.py \
  --dataset_dir ../data/expert/r2r --merge
```

Repeat for English RxR using `--dataset rxr`, `config/vln_rxr_train.yaml`, the
official `train_guide.json.gz` and a separate `../data/expert/rxr` directory.
The collectors write chronological images and recorded poses/actions; the
sample builder writes `samples_perstep.jsonl`. The training configurations
are distinct from the validation-evaluation configurations.

### Collect Stage 2 DAgger data

Train Stage 1 first. Then roll out that policy on **training episodes** and
query the expert at each visited state. The collector records both the policy's
executed actions and the expert's counterfactual action chunks. Use separate
physical GPUs for policy inference and simulator rendering:

```bash
python src/data_process/dagger_rollout.py \
  --model_path /absolute/path/to/runs/stage1/stage1-epoch1 \
  --dataset r2r_dagger --split train \
  --habitat_config_path config/vln_r2r_train.yaml \
  --data_path /absolute/path/to/r2r/train/train.json.gz \
  --scenes_dir "$GUIDEDNAV_SCENES_DIR" \
  --output_path ../data/dagger/r2r --model_gpu 1 --sim_gpu 0 \
  --num_history 8 --num_future_steps 4 --num_exec 4 --seed 42 --require_stop
python src/data_process/dagger_rollout.py --output_path ../data/dagger/r2r --merge
python src/data_process/build_vln_samples.py \
  --dataset_dir ../data/dagger/r2r --memory_size 8 --num_future_steps 4 \
  --seed 42 --action_source expert_chunks
python src/data_process/build_vln_samples.py \
  --dataset_dir ../data/dagger/r2r --merge
```

Repeat with `--dataset rxr_dagger`, `config/vln_rxr_train.yaml`, the official
RxR training guide file and a separate RxR output directory. Use `--rank` and
`--world_size` to shard collection. Collection defaults and recovery settings
are exposed by `--help`; retain the configuration with your generated corpus.

The DAgger target must be `expert_chunks`, **not the policy's executed actions**.
The collected states and sample count depend on your Stage 1 policy.
Keep the generated corpus and collection settings with the training run.

### Freeze Stage 1 and Stage 2 manifests

Edit `config/training_stage1.example.json` and
`config/training_stage2.example.json` to point to your files. Paths in these
specifications are relative to the specification file. Each navigation source
uses `video_folder` and `samples_path`; each VQA source uses `annotation_path`
(a JSON/JSONL path or list of globs) and `data_path` for media. VQA inputs support
image paths, video paths or extracted video-frame directories and human/GPT
conversation records. Keep the original media references when normalizing
upstream annotations.

```bash
python tools/prepare_training_manifest.py \
  --config config/training_stage1.example.json --output-dir ../training/stage1
python tools/prepare_training_manifest.py --verify ../training/stage1/manifest.json
python tools/prepare_training_manifest.py \
  --config config/training_stage2.example.json --output-dir ../training/stage2
python tools/prepare_training_manifest.py --verify ../training/stage2/manifest.json
```

Run Stage 2 manifest preparation after Stage 1 data preparation: the example
reuses Stage 1's already selected VQA JSONL files. The tool freezes source
hashes, selected indices and normalized JSONL files, and rejects role mismatches,
invalid action labels and changed inputs. `--check-media all` checks every
referenced media path; the default checks all selected media references. Use the resulting
`../training/stage1/manifest.json` and `../training/stage2/manifest.json` as
`TRAIN_MANIFEST` in the corresponding launch commands.

## Released Stage 3 annotation add-on

The complete selected Stage 3 annotation, trajectory and prompt-metadata add-on
is available from the [public release](https://github.com/anonymousauthor-s/GuidedNav/releases/). From `code/`, download it with:

```bash
python tools/download_assets.py --group data
```

It installs `annotations/`, `data/r2r/traj/`, `data/rxr/traj/`,
`data/pose-manifest.json` and `prompt-metadata/` at the repository root.
Combine these files with official benchmark instructions and licensed
Matterport3D scenes to reconstruct the training inputs. The package supplies
exact collected poses/actions, aligned labels/masks, frozen selection arrays and
per-row prompt choices. RGB is rendered locally. See the [model guide](https://anonymousauthor-s.github.io/GuidedNav/guide/models/)
for the full weights and [data terms](https://github.com/anonymousauthor-s/GuidedNav/releases/) for attribution and upstream conditions.

The released Stage 3 subset contains 157,465 R2R and 472,323 English RxR observations from 18,656 trajectories. Use your Stage 2 model as the training backbone; see [TRAINING.md](TRAINING.md).

## 1. Obtain official benchmark assets

Follow the [VLN-CE data setup](https://github.com/jacobkrantz/VLN-CE#data) for R2R episodes and MP3D scene preparation, the [official RxR-Habitat data instructions in VLN-CE](https://github.com/jacobkrantz/VLN-CE#episodes-room-across-room-rxr) for RxR guide episodes and ground-truth trajectories, and the [Matterport3D access instructions](https://niessner.github.io/Matterport/) for scene assets. Follow each source's access and licensing terms.

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

This CPU geometry tool creates a renderer-free simulator, applies default `NavMeshSettings` with height/radius 0.88/0.18, saves each scene's new mesh, and checks finite start-to-goal reachability plus valid reference trajectories for every English episode. It requires the exact benchmark data SHA256 `93abddce58e8232867a8738daa8107eb82953693a9d5e70779d1eb00386b222a` and GT SHA256 `bdfbb624132501bf297eb0ea16a99e38682668ba5e9ce1554ca5cdcde6b65a0d`. Keep the original compressed files to preserve these hashes. `audit.json` and `*.navmesh` stay together; the evaluator verifies each mesh hash and agent dimensions. `scripts/evaluate.sh rxr` sets the audit/dataset/GT environment paths automatically. Override `RXR_NAVMESH_AUDIT_PATH`, `RXR_EVAL_DATA_PATH`, `RXR_EVAL_GT_PATH` and `GUIDEDNAV_SCENES_DIR` when using a different local layout.

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

The add-on provides the following information:

| Information | Purpose |
| --- | --- |
| Public instruction ID/hash, video/frame identity | Recover the correct original text and observation index |
| Exact collected agent poses and actions, with per-file hashes | Render the original observation sequence and recover the four-action targets |
| Masks, character spans, phases, groups, weights and eligibility | Preserve the added supervision |
| Frozen selected `source_indices` | Preserve the exact stage-3 examples and their order |
| Per-row `conjunction_id` and its fixed seven-string vocabulary | Reproduce the original action prompt without drawing a new random choice |
| Renderer/sensor and sample-building settings | Preserve camera geometry, history sampling and action indexing |

The conjunction IDs require one byte per selected row, plus small headers. The current action-training loader does not use stored waypoint coordinates or distances as targets. It consumes the recovered instruction, image/history paths, action-only conversation and the S/O supervision. Legacy subtask-answer text is removed before training, so its original string does not have to be redistributed.

`tools/materialize_annotation_addon.py` reconstructs loader records from the public text and the add-on:

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

The output contains `manifest.local.json`, reconstructed JSONL records and array sidecars. The action labels, history-frame paths, prompts and mask pointers have been checked across all 629,788 observations.

## 3. Render training observations from collected poses

The add-on provides the original per-video collected positions, rotations and actions in NPZ files. Unused scene semantics and reference-path fields are omitted. The files store world-space **agent** positions and rotations in **w,x,y,z** order, with one pose per observation, as well as the recorded discrete actions. Their filenames match the video basename in each annotated row. All 18,656 selected trajectories were checked against the training observation/action indices. `tools/render_training_rgb.py` converts to Habitat's x,y,z,w coefficients and reapplies the camera offset once.

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

The renderer saves RGB images with Pillow's default JPEG settings. Before training, inspect a rendered frame and its mask at the same frame index to check their alignment.

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
  --processor /absolute/path/to/runs/stage2/stage2-epoch1
```

This checks positive S/O examples from both sources, image decoding, tokenizer span mapping, object-mask pooling, temporal predecessor frames, VGGT image preprocessing and collation. Run the gradient smoke described in TRAINING.md before a formal distributed run.

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

For the released subset, use the add-on's labels, masks and selection arrays directly.

For additional expert data, use `src/data_process/gt_following_rollout.py` and `build_vln_samples.py` with memory size 8, four future actions and seed 42.

## 5. Obtain the exact training teacher

The vendored `third_party/vggt/vggt` source is the teacher implementation used by the final stage. Its source hashes and model hash are in `config/vggt_reference.json`. File hashes identify the included source snapshot.

The exact 5,026,874,952-byte teacher matches the [official VGGT-1B model payload](https://huggingface.co/facebook/VGGT-1B/blob/main/model.pt), SHA256 `d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0`. After reviewing upstream terms, download it with:

```bash
python tools/download_teacher.py --output ../assets/vggt/vggt.pt
```

The downloader verifies size and SHA256 before completing the file. The frozen teacher provides spatial supervision during training.

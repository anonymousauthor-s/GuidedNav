# GuidedNav · Code

> **Upload in progress.** Model and data download links will become active after all files pass the public-download checks.


[Project page](https://anonymousauthor-s.github.io/GuidedNav/) · [Inference](#inference) · [Training](#stage-3-training) · [Data](#training-data) · [Models](#models-and-directory-layout) · [Evaluation](#habitat-evaluation)

This directory contains frame-policy inference, R2R-CE/RxR-CE evaluation, and **Stage 3 guidance fine-tuning for one epoch**. It starts from navigation-pretrained weights and a supplied Stage 3 initialization. Reconstructing the earlier training stages is outside this package's scope.

## Models and directory layout

The complete model bundle and selected Stage 3 data add-on are being uploaded to the [public release](https://github.com/anonymousauthor-s/GuidedNav/releases/tag/stage3-epoch1). See the [download guide](https://anonymousauthor-s.github.io/GuidedNav/guide/models/) or run from `code/`:

```bash
python tools/download_assets.py --group inference
```

```text
GuidedNav/
├── code/                              # inference, training and evaluation
├── weights/
│   ├── navigation-base/               # full navigation-pretrained backbone
│   ├── stage3-initialization/         # Stage 3 initialization
│   └── stage3-epoch1/                 # reported final adapter and auxiliary weights
├── annotations/                      # frozen labels, masks and selection arrays
├── prompt-metadata/                  # exact per-row prompt choices
├── data/
│   ├── pose-manifest.json            # hashes of collected trajectories
│   ├── r2r/traj/                     # released poses and actions
│   ├── rxr/traj/
│   ├── r2r/rgb/                      # locally rendered observations
│   ├── rxr/rgb/
│   ├── datasets/                     # official benchmark episodes
│   └── scene_datasets/               # licensed scenes from upstream
├── stage3-training-data/             # locally materialized training records
└── assets/vggt/vggt.pt               # separately obtained training teacher
```

Both adapters require `navigation-base`; neither is a standalone 8B model. Keep each adapter's auxiliary weights, processor, and recipe together. The inference bundle is approximately **16.60 GiB**, including the full BF16 navigation VLM, adapter, and learned auxiliary weights with the temporal gate. Including the Stage 3 initialization gives approximately **16.86 GiB** of model assets. A vanilla Qwen checkpoint does not replace the navigation-pretrained backbone. The VGGT teacher and training annotations are not needed at inference and are separate from these model-size totals.

All commands below run from `code/`. Use the separate environments in [ENVIRONMENT.md](ENVIRONMENT.md): frame inference, Stage 3 training, and Habitat rendering/evaluation have different dependencies.

## Inference

Download the model first with `python tools/download_assets.py --group inference`.

### 1. Set up the frame-policy environment

The recorded frame runtime uses PyTorch 2.8.0 with CUDA 12.8, Transformers 4.57.6, PEFT 0.17.1, and Accelerate 1.10.1. A Python 3.10/3.11 environment can be prepared with:

```bash
python -m venv .venv-inference
source .venv-inference/bin/activate
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-inference.txt
export PYTHONPATH="$PWD/src"
python tools/check_environment.py
```

See [ENVIRONMENT.md](ENVIRONMENT.md#frame-policy-inference) for the runtime details. A CUDA GPU with sufficient memory for the 8B policy is required for the standard BF16 path.

### 2. Prepare chronological observations

Create `frames.txt` with one RGB image path per line. Paths can be absolute or relative to that text file:

```text
observations/0001.jpg
observations/0002.jpg
observations/0003.jpg
```

The **current observation must be last**. Supply the accumulated observation sequence: the policy samples its history and separately builds the recent four-frame temporal window. Providing only a short suffix changes that history protocol.

### 3. Generate the next action chunk

```bash
python -m qwenvl.infer_frames \
  --base ../weights/navigation-base \
  --checkpoint ../weights/stage3-epoch1 \
  --instruction 'Walk through the doorway and stop beside the table.' \
  --frames-list /absolute/path/to/frames.txt
```

The JSON output contains up to four actions:

| ID | Action |
| --- | --- |
| `0` | Stop |
| `1` | Move forward 0.25 m |
| `2` | Turn left 15° |
| `3` | Turn right 15° |

The default frame path uses BF16 and SDPA. It loads the adapter together with its learned temporal components; loading the LoRA file alone is insufficient.

### Use a model you trained

The default `--weights-source released` verifies the released model's identity. After Stage 3 training, use the exported model directory with `--weights-source trained`:

```bash
python -m qwenvl.infer_frames \
  --base ../weights/navigation-base \
  --checkpoint /absolute/path/to/your-run/stage3-epoch1 \
  --weights-source trained \
  --instruction 'Walk through the doorway and stop beside the table.' \
  --frames-list /absolute/path/to/frames.txt
```

The trained-model path still checks backbone compatibility, the saved recipe, and required tensors.

## Training data

Follow [DATA.md](DATA.md) to obtain the official benchmark assets, prepare the complete annotation/trajectory package, render RGB observations, relocate the manifest, and obtain the VGGT teacher.

| Stage 3 source | Selected training rows |
| --- | ---: |
| R2R | 157,465 |
| RxR | 472,323 |
| **Total** | **629,788** |

This stage adds no DAgger or VQA rows. Earlier navigation training is carried by the pretrained assets.

Download the complete selected Stage 3 add-on with `python tools/download_assets.py --group data`. Combine it with official instructions and licensed scenes to reconstruct the Stage 3 training data: it supplies step-aligned labels, exact selected rows, recorded poses/actions, and prompt metadata. RGB can be rendered from the recorded poses. Preserve the frozen sample selection and frame correspondence when preparing the data.

After preparing a relocated manifest, check example decoding and collation in the **training environment** before launching a distributed run:

```bash
export PYTHONPATH="$PWD/src"
export FULL_MANIFEST=/absolute/path/to/manifest.local.json
python tools/smoke_dataset.py \
  --manifest "$FULL_MANIFEST" \
  --processor ../weights/stage3-initialization \
  --per-source 2
```

This check loads the processor and examples without loading the 8B policy or VGGT weights.

## Stage 3 training

### Reference setup

The launcher reproduces the one-epoch guidance stage with a fresh optimizer and scheduler:

| Setting | Value |
| --- | --- |
| Training GPUs | 32: 4 nodes × 8 GPUs |
| Batch size | 1 per GPU, accumulation 8; global batch 256 |
| Epochs | 1 |
| Optimizer | AdamW; learning rate `2e-4`; weight decay `0.01` |
| Schedule | Cosine decay; 3% warmup |
| LoRA | Rank 32, alpha 16, dropout 0; all supported linear layers |
| Precision and distribution | BF16; DeepSpeed ZeRO-2 |
| Seed | 42 |
| Guidance weights | Subtask `0.01`, landmark `0.01`, long-horizon `0.5`, local `0.1` |

The paper's 64-GPU cluster includes training and asynchronous evaluation. This training launcher uses 32 GPUs. A smaller configuration needs a separately validated batch and schedule.

### Launch on four nodes

Prepare the training environment in [ENVIRONMENT.md](ENVIRONMENT.md#final-stage-training). Mount the repository, data, teacher, and weights at the same absolute paths on every node. On each node, set its rank and a shared rendezvous address:

```bash
export FULL_MANIFEST=/absolute/path/to/manifest.local.json
export MASTER_ADDR=training-rendezvous-host
export NODE_RANK=0  # use 0, 1, 2, or 3 on the corresponding node
export NCCL_SOCKET_IFNAME=your_training_network_interface
export FULL_OUTPUT_DIR=/absolute/path/to/new-empty-run
bash scripts/train_stage3.sh
```

The launcher uses `weights/navigation-base` and `weights/stage3-initialization` by default. Set `GUIDEDNAV_BASE`, `GUIDEDNAV_INIT`, `VGGT_SOURCE`, or `VGGT_WEIGHTS` to override their locations. It restores the initialization adapter and all learned auxiliary components, then starts the one-epoch Stage 3 optimization.

After the epoch, the inference-ready adapter, auxiliary weights, and processor are exported to:

```text
$FULL_OUTPUT_DIR/stage3-epoch1/
```

Complete `checkpoint-*` trainer directories retain the optimizer state for interrupted-run recovery. To resume, set `FULL_RESUME` to a complete trainer checkpoint and rerun the same launcher. The supplied Stage 3 initialization starts a fresh optimization stage and does not contain optimizer-resume state.

For a two-update diagnostic, use `FULL_SMOKE=1` and a separate empty output directory. `NNODES=1 NPROC_PER_NODE=1` can reduce its worker count if memory permits. Changing the worker count changes the effective batch and schedule; this diagnostic does not reproduce the reference model.

## Habitat evaluation

Install the Habitat environment in [ENVIRONMENT.md](ENVIRONMENT.md#habitat-rendering-and-benchmark-evaluation) and prepare the official episodes, scenes, and audited RxR navigation meshes in [DATA.md](DATA.md). Then run:

```bash
bash scripts/evaluate.sh r2r
bash scripts/evaluate.sh rxr
```

Both commands evaluate `val_unseen` with greedy decoding, history 8, four-action prediction/execution chunks, and at most 500 episode steps. R2R uses 1,839 episodes. RxR uses all 3,669 English guide episodes, audited navigation meshes, a 0.88 m / 0.18 m collision body, and sliding enabled. See DATA.md for the complete sensor/action protocol and the navmesh build command.

Use a fresh `EVAL_OUTPUT` directory for each run because episode results are appended. `EVAL_WORKERS` sets the number of distributed evaluator workers; allocate one simulator worker per rendering GPU.

To evaluate your own exported model:

```bash
GUIDEDNAV_MODEL=/absolute/path/to/your-run/stage3-epoch1 \
GUIDEDNAV_WEIGHTS_SOURCE=trained \
EVAL_OUTPUT=/absolute/path/to/new-evaluation-directory \
bash scripts/evaluate.sh r2r
```

Replace `r2r` with `rxr` for RxR evaluation. A one-episode diagnostic is available by appending `--max_episodes 1`.

## Release validation

The recorded checks include CPU imports, argument parsing, configuration composition, and sample decoding/collation. Full GPU retraining, complete benchmark evaluation, and full-dataset RGB rendering have not been rerun during packaging. See [VALIDATION.json](VALIDATION.json) for the exact checks and limits, [ENVIRONMENT.md](ENVIRONMENT.md) for runtime versions, and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for retained upstream notices.

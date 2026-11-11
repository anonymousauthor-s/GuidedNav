# Three-stage training

Train GuidedNav-8B from Qwen3-VL-8B-Instruct through Stages 1, 2 and 3.
Each stage uses the preceding stage's exported model.

All commands run from `GuidedNav/code/`. Install the [training environment](ENVIRONMENT.md#training)
and use the same absolute model and data paths on every node.

## Recipe at a glance

| | Stage 1 | Stage 2 | Stage 3 — GuidedNav |
| --- | --- | --- | --- |
| Start from | Official Qwen3-VL-8B-Instruct | Your Stage 1 model | Your Stage 2 model |
| Navigation data | R2R + English RxR expert examples | DAgger-collected examples | Annotated R2R + English RxR experts |
| Visual instruction data | LLaVA-Video + VSI | Same VQA mixture | None |
| Parameters updated | Full model | Full model | Rank-32 LoRA + guidance components |
| Language/policy peak LR | `2e-5` | `1e-5` | `2e-4` |
| Vision/projector peak LR | `5e-6` | `5e-6` | LoRA policy LR |
| Epochs / global batch | 1 / 256 | 1 / 256 | 1 / 256 |
| Export | `stage1-epoch1/` | `stage2-epoch1/` | `stage3-epoch1/` |

All stages use AdamW (`betas=(0.9,0.999)`, weight decay `0.01`, gradient clipping
`1.0`), cosine decay, 3% warmup, BF16, DeepSpeed ZeRO-2 and seed 42. Maximum
sequence length is 8,192; image pixel bounds are 12,544–200,704.
Stage 3 uses LoRA rank 32, alpha 16 and dropout 0. Its subtask, landmark,
long-horizon spatial and local spatial loss weights are `0.01, 0.01, 0.5, 0.1`,
with 3% auxiliary warmup. Ground-truth annotations and the frozen VGGT teacher
are used only during training.

## Prepare the model and data

Download [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)
from its official model repository into a local directory, including tokenizer
and processor files. This is the **starting model for Stage 1**.

Follow [data preparation](DATA.md) to prepare expert navigation examples,
DAgger trajectories, the visual instruction mixture and guidance annotations.
The paper uses approximately 0.53M R2R experts, 1.90M RxR experts, 2.97M DAgger
examples and 0.79M visual instruction examples. Freeze the actual source files
and selected rows in the training manifests before launching a run.

Download the Stage 3 annotations, poses/actions and prompt metadata with the
provided downloader. Prepare expert rollouts, DAgger rollouts and VQA media
using the data guide. Keep the same VQA selection across Stages 1 and 2.

## Launch Stage 1 — expert navigation and visual instruction tuning

```bash
export GUIDEDNAV_START=/absolute/path/to/Qwen3-VL-8B-Instruct
export TRAIN_MANIFEST=/absolute/path/to/training/stage1/manifest.json
export TRAIN_OUTPUT=/absolute/path/to/runs/stage1
export MASTER_ADDR=10.0.0.1
export NNODES=4 NPROC_PER_NODE=8 NODE_RANK=0
bash scripts/train_stage1.sh
```

Run the same command on each node, setting `NODE_RANK` to its zero-based rank
and keeping the rendezvous address and input/output paths consistent.
The full-parameter output is `TRAIN_OUTPUT/stage1-epoch1/`.

## Launch Stage 2 — DAgger navigation and visual instruction tuning

Collect and label navigation rollouts with your Stage 1 policy as described in
[data preparation](DATA.md). Use the DAgger manifest for this stage, together
with the same visual instruction examples used by Stage 1.

```bash
export GUIDEDNAV_START=/absolute/path/to/runs/stage1/stage1-epoch1
export TRAIN_MANIFEST=/absolute/path/to/training/stage2/manifest.json
export TRAIN_OUTPUT=/absolute/path/to/runs/stage2
bash scripts/train_stage2.sh
```

The full-parameter output is `TRAIN_OUTPUT/stage2-epoch1/`.
Stages 1 and 2 update the language model, vision tower and multimodal projector;
they do not enable GuidedNav attention or spatial guidance.

## Launch Stage 3 — GuidedNav guidance

Download the annotation add-on and the training-only VGGT teacher:

```bash
python tools/download_assets.py --group data
python tools/download_teacher.py --output ../assets/vggt/vggt.pt
```

Recover the official instruction text, render observations from the released
poses and materialize the Stage 3 manifest following [DATA.md](DATA.md).
Validate decoded examples using your own Stage 2 processor:

```bash
export GUIDEDNAV_BASE=/absolute/path/to/runs/stage2/stage2-epoch1
export FULL_MANIFEST=/absolute/path/to/stage3-training-data/manifest.local.json
export FULL_OUTPUT_DIR=/absolute/path/to/runs/stage3
export PYTHONPATH="$PWD/src:$PWD/third_party/vggt"
python tools/smoke_dataset.py --manifest "$FULL_MANIFEST" --processor "$GUIDEDNAV_BASE"
bash scripts/train_stage3.sh
```

Stage 3 initializes LoRA adapters, spatial projectors and a temporal gate on
your Stage 2 backbone. It saves the adapter, processor, temporal state and recipe
in `FULL_OUTPUT_DIR/stage3-epoch1/`.
Keep that directory with the exact Stage 2 backbone used to train it.

## Hardware and run checks

The default launch uses four nodes with eight GPUs each. Stages 1 and 2 use
microbatch 2 and accumulation 4; Stage 3 uses microbatch 1 and accumulation 8.
The scripts calculate accumulation from worker count to maintain global batch
256 and reject incompatible combinations. Set `NNODES` and `NPROC_PER_NODE` for your hardware; `MICROBATCH` can also be
changed for Stages 1/2. Stage 3 uses microbatch 1.

For a two-update diagnostic, use a separate empty output directory and set
`TRAIN_SMOKE=1` for Stages 1/2 or `FULL_SMOKE=1` for Stage 3. Complete trainer checkpoints preserve
optimizer state for interrupted-run recovery; epoch exports are for stage
handoff and inference.

## Evaluate a model you trained

Use your own Stage 2 backbone and your own Stage 3 adapter together:

```bash
export GUIDEDNAV_BASE=/absolute/path/to/runs/stage2/stage2-epoch1
export GUIDEDNAV_MODEL=/absolute/path/to/runs/stage3/stage3-epoch1
export GUIDEDNAV_WEIGHTS_SOURCE=trained
export EVAL_OUTPUT=/absolute/path/to/new-results/r2r
bash scripts/evaluate.sh r2r
```

Replace `r2r` with `rxr` for RxR. To evaluate the **released GuidedNav-8B checkpoint**,
leave the default released-model settings in place and follow the
[evaluation guide](README.md#habitat-evaluation).

See the [validation record](VALIDATION.json) for completed checks.

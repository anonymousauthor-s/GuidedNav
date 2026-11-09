# GuidedNav · Code

[Project](https://anonymousauthor-s.github.io/GuidedNav/) · [Inference](#inference) · [Evaluation](#habitat-evaluation) · [Training](TRAINING.md) · [Data](DATA.md) · [Model](MODELS.md)

Download the final paper model for inference and benchmark evaluation, or
train your own model with the three-stage recipe. Training starts from official
Qwen3-VL-8B-Instruct and uses your own outputs at each stage.

## Model and directory layout

```text
GuidedNav/
├── code/                         # training, inference, evaluation and tools
├── weights/                      # downloaded final paper model only
│   ├── navigation-base/          # required backbone component
│   └── stage3-epoch1/             # matching final adapter and temporal state
├── runs/                         # your locally trained outputs
│   ├── stage1/stage1-epoch1/
│   ├── stage2/stage2-epoch1/
│   └── stage3/stage3-epoch1/
├── annotations/                  # released Stage 3 labels and masks
├── prompt-metadata/              # released prompt choices
├── data/                         # trajectories, official scenes and datasets
├── stage3-training-data/         # locally materialized guidance records
└── assets/vggt/vggt.pt           # training-only teacher
```

The downloaded backbone and final adapter together are **one evaluation model**,
approximately 16.60 GiB in BF16. Intermediate training models are not released.
All commands below run from `code/`; see [environments](ENVIRONMENT.md).

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
  --base /absolute/path/to/runs/stage2/stage2-epoch1 \
  --checkpoint /absolute/path/to/your-run/stage3-epoch1 \
  --weights-source trained \
  --instruction 'Walk through the doorway and stop beside the table.' \
  --frames-list /absolute/path/to/frames.txt
```

The trained-model path still checks backbone compatibility, the saved recipe, and required tensors.

## Training

The [three-stage guide](TRAINING.md) gives data preparation, launch commands and
stage handoff:

| Stage | Data | Update | Entry point |
| --- | --- | --- | --- |
| 1 | R2R/RxR experts + visual instruction data | Full parameters | `scripts/train_stage1.sh` |
| 2 | DAgger + the same visual instruction mixture | Full parameters | `scripts/train_stage2.sh` |
| 3 | Annotated R2R/RxR experts | LoRA + guidance | `scripts/train_stage3.sh` |

No training launcher requires a downloaded intermediate GuidedNav checkpoint.
Each stage reads your preceding stage's export.

## Training data

Follow [DATA.md](DATA.md) for original benchmark assets, upstream visual
instruction data, DAgger collection, guidance annotations and the teacher.
Download the selected guidance add-on with:

```bash
python tools/download_assets.py --group data
```

The add-on covers 157,465 R2R and 472,323 English RxR rows, with aligned masks,
collected poses/actions and prompt choices. Its correspondence to official text
has been checked across all 629,788 rows. Earlier-stage corpora are prepared
separately and are not included in this annotation download.

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
GUIDEDNAV_BASE=/absolute/path/to/runs/stage2/stage2-epoch1 \
GUIDEDNAV_MODEL=/absolute/path/to/runs/stage3/stage3-epoch1 \
GUIDEDNAV_WEIGHTS_SOURCE=trained \
EVAL_OUTPUT=/absolute/path/to/new-evaluation-directory \
bash scripts/evaluate.sh r2r
```

Replace `r2r` with `rxr` for RxR evaluation. A one-episode diagnostic is available by appending `--max_episodes 1`.

## Release validation

The recorded checks include CPU imports, argument parsing, configuration composition, and sample decoding/collation. Full GPU retraining, complete benchmark evaluation, and full-dataset RGB rendering have not been rerun during packaging. See [VALIDATION.json](VALIDATION.json) for the exact checks and limits, [ENVIRONMENT.md](ENVIRONMENT.md) for runtime versions, and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for retained upstream notices.

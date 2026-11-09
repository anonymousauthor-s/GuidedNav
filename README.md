# GuidedNav

### Shaping Vision-Language Navigation Representations through Pre-Action Attention and Spatial Guidance

**Anonymous release for peer review**

[Project page](https://anonymousauthor-s.github.io/GuidedNav/) · [Inference & evaluation](https://anonymousauthor-s.github.io/GuidedNav/guide/inference/) · [Three-stage training](https://anonymousauthor-s.github.io/GuidedNav/guide/training/) · [Training data](https://anonymousauthor-s.github.io/GuidedNav/guide/data/) · [Final model](https://anonymousauthor-s.github.io/GuidedNav/guide/models/)

GuidedNav learns to navigate from RGB observations and language instructions
with subtask, landmark and spatial guidance. This repository provides the
**final paper model for evaluation**, inference and evaluation code, and
**training code and instructions for Stages 1, 2 and 3**.

## Evaluate the final paper model

```bash
git clone https://github.com/anonymousauthor-s/GuidedNav.git
cd GuidedNav/code
python tools/download_assets.py --group inference
```

Install the [evaluation environment](https://anonymousauthor-s.github.io/GuidedNav/guide/environment/), prepare the official
benchmark episodes and licensed scenes, then run:

```bash
bash scripts/evaluate.sh r2r
bash scripts/evaluate.sh rxr
```

The only released policy is **Stage 3, epoch 1**. Its complete BF16 model is
16.60 GiB: the matching navigation backbone plus final adapter and temporal
state. They are installed together and loaded as one policy. Intermediate and
training-initialization checkpoints are not distributed.

See [inference and evaluation](https://anonymousauthor-s.github.io/GuidedNav/guide/inference/) for the full setup and
[model downloads](https://anonymousauthor-s.github.io/GuidedNav/guide/models/) for SHA256-verified, resumable downloads.

## Train your own model

| Stage | Initialization | Training data | Update |
| --- | --- | --- | --- |
| 1 | Official Qwen3-VL-8B-Instruct | R2R/RxR experts + VQA | Full parameters |
| 2 | Your Stage 1 output | DAgger + the same VQA mixture | Full parameters |
| 3 | Your Stage 2 output | Annotated R2R/RxR experts | Rank-32 LoRA + guidance |

The [three-stage training guide](https://anonymousauthor-s.github.io/GuidedNav/guide/training/) gives launch commands,
data manifests, the paper's optimizer settings and model handoff. Training uses
your own outputs; downloading the final paper model is a separate evaluation path.

## Prepare training data

```bash
python tools/download_assets.py --group data
```

The selected Stage 3 add-on contains **629,788 annotated rows, 379,428 masks and
18,656 collected trajectories/actions**, together with instruction IDs/hashes
and prompt metadata. Obtain official instructions and licensed scenes from
upstream; the supplied tools reconstruct text and render observations.
Expert navigation, DAgger and VQA inputs for Stages 1 and 2 are prepared
separately. See [data preparation](https://anonymousauthor-s.github.io/GuidedNav/guide/data/) for sources, formats and
the exact scope of the released labels.

## Validation

Release checks cover model payload hashes, CPU helpers and command-line entry
points, data correspondence and documentation links. The new three-stage
training sequence and complete GPU benchmarks have not been rerun from this
public package. See [validation details](code/VALIDATION.json) and retained
[upstream notices](code/THIRD_PARTY_NOTICES.md).

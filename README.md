# GuidedNav

### Shaping Vision-Language Navigation Representations through Pre-Action Attention and Spatial Guidance

**Anonymous release for peer review**

[Project page](https://anonymousauthor-s.github.io/GuidedNav/) · [Inference](https://anonymousauthor-s.github.io/GuidedNav/guide/inference/) · [Training](https://anonymousauthor-s.github.io/GuidedNav/guide/training/) · [Data](https://anonymousauthor-s.github.io/GuidedNav/guide/data/) · [Model](https://anonymousauthor-s.github.io/GuidedNav/guide/models/)

GuidedNav learns to navigate from RGB observations and language instructions
with subtask, landmark and spatial guidance. We release the **GuidedNav-8B
checkpoint**, training code, inference and evaluation code, and training annotations.

## Inference and evaluation

```bash
git clone https://github.com/anonymousauthor-s/GuidedNav.git
cd GuidedNav/code
python tools/download_assets.py --group inference
```

Install the [evaluation environment](https://anonymousauthor-s.github.io/GuidedNav/guide/environment/)
and prepare the official benchmark episodes and scenes, then run:

```bash
bash scripts/evaluate.sh r2r
bash scripts/evaluate.sh rxr
```

For inference on your own images, see the [inference guide](https://anonymousauthor-s.github.io/GuidedNav/guide/inference/).

## Training

| Stage | Initialization | Training data | Update |
| --- | --- | --- | --- |
| 1 | Qwen3-VL-8B-Instruct | R2R/RxR experts + VQA | Full parameters |
| 2 | Stage 1 output | DAgger + the same VQA mixture | Full parameters |
| 3 | Stage 2 output | Annotated R2R/RxR experts | Rank-32 LoRA + guidance |

Follow the [three-stage training guide](https://anonymousauthor-s.github.io/GuidedNav/guide/training/)
for data preparation, launch commands and hyperparameters.

## Training data

```bash
python tools/download_assets.py --group data
```

The Stage 3 annotation package contains **629,788 annotated samples, 379,428 masks
and 18,656 trajectories**, with recorded poses, actions and prompt metadata.
Download the original instructions and scenes from their official sources,
then use the provided tools to reconstruct the training inputs.

The [data guide](https://anonymousauthor-s.github.io/GuidedNav/guide/data/) also
covers expert rollouts, DAgger collection and VQA preparation for Stages 1 and 2.

## Model download

The **GuidedNav-8B checkpoint** is 16.60 GiB in BF16. The download command installs
its backbone, adapter, temporal state and processor together.
Downloads support resuming and SHA256 verification.

See [model downloads](https://anonymousauthor-s.github.io/GuidedNav/guide/models/)
and the [GitHub release](https://github.com/anonymousauthor-s/GuidedNav/releases/tag/stage3-epoch1).

[Validation record](code/VALIDATION.json) · [Third-party notices](code/THIRD_PARTY_NOTICES.md)

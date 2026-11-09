# GuidedNav

### Shaping Vision-Language Navigation Representations through Pre-Action Attention and Spatial Guidance

**Anonymous code release for peer review**

[Project page](https://anonymousauthor-s.github.io/GuidedNav/) · [Inference](code/README.md#inference) · [Training](code/README.md#stage-3-training) · [Data](code/DATA.md) · [Models](#model-downloads)

GuidedNav learns to navigate from RGB observations and language instructions with subtask, landmark, and spatial guidance. This repository provides **frame-policy inference, Habitat evaluation, and Stage 3 guidance fine-tuning** for the reported model. Earlier navigation training is represented by the required pretrained assets; this release does not reconstruct all earlier training stages.

## Start here

| Goal | Guide |
| --- | --- |
| Run the policy on an RGB trajectory | [Inference quick start](code/README.md#inference) |
| Evaluate on R2R-CE or RxR-CE | [Habitat evaluation](code/README.md#habitat-evaluation) |
| Reproduce the one-epoch guidance stage | [Stage 3 training](code/README.md#stage-3-training) |
| Prepare scenes, labels, and training observations | [Data preparation](code/DATA.md) |
| Set up inference, training, or Habitat | [Environments](code/ENVIRONMENT.md) |

```bash
git clone https://github.com/anonymousauthor-s/GuidedNav.git
cd GuidedNav/code
```

Follow the environment guide and obtain the assets below before running the model.

## Model downloads

**Model weights and the complete annotation, trajectory, and prompt-metadata add-on are pending release.** Download links will be added here when they are available.

| Asset | Directory relative to the repository root | Used for | Availability |
| --- | --- | --- | --- |
| Navigation-pretrained backbone | `weights/navigation-base/` | Inference and training | Pending release |
| Stage 3 initialization | `weights/stage3-initialization/` | Start guidance fine-tuning | Pending release |
| Stage 3, epoch 1 | `weights/stage3-epoch1/` | Reported GuidedNav policy | Pending release |
| Training annotation, trajectory, and prompt-metadata add-on | `data/` | Reproduce the training examples | Pending release |

The inference bundle is approximately **16.60 GiB**, including the full BF16 navigation VLM, Stage 3 adapter, and learned auxiliary weights with the temporal gate. Including the Stage 3 initialization brings the model assets to approximately **16.86 GiB**; training data and the VGGT teacher are separate. Use the matching navigation-pretrained backbone and adapter together. A vanilla Qwen model cannot replace that backbone. The frozen VGGT teacher is needed for training only; [DATA.md](code/DATA.md) describes its upstream source and preparation.

## Training data

The final guidance stage uses **157,465 R2R examples and 472,323 RxR examples**. Reconstructing its observations requires the official instructions and licensed scene assets, together with our aligned labels, exact sample selection, recorded agent poses/actions, and prompt metadata. RGB images can be rendered from the recorded poses, so RGB redistribution is not required in principle.

The complete annotation, trajectory, and prompt-metadata add-on is pending release. Combine this add-on with the official benchmark data and licensed scenes to prepare the Stage 3 training examples. [DATA.md](code/DATA.md) explains the required files, upstream access, rendering, and availability.

## Repository

```text
code/    Inference, Stage 3 training, evaluation, and setup instructions
docs/    Static project website
```

See the [validation record](code/VALIDATION.json) for checks performed on the release and their limits. Upstream copyright and license notices are retained in [THIRD_PARTY_NOTICES.md](code/THIRD_PARTY_NOTICES.md) and the corresponding source files.

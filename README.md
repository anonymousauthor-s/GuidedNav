# GuidedNav

> **Upload in progress.** Model and data download links will become active after all files pass the public-download checks.


### Shaping Vision-Language Navigation Representations through Pre-Action Attention and Spatial Guidance

**Anonymous release for peer review**

[Project page](https://anonymousauthor-s.github.io/GuidedNav/) · [Inference](https://anonymousauthor-s.github.io/GuidedNav/guide/inference/) · [Training](https://anonymousauthor-s.github.io/GuidedNav/guide/training/) · [Training data](https://anonymousauthor-s.github.io/GuidedNav/guide/data/) · [Model downloads](https://anonymousauthor-s.github.io/GuidedNav/guide/models/)

GuidedNav learns to navigate from RGB observations and language instructions
with subtask, landmark and spatial guidance. This repository provides the
**full navigation model, frame-policy inference, Habitat evaluation, and the
one-epoch Stage 3 guidance training recipe**.

## Get started

```bash
git clone https://github.com/anonymousauthor-s/GuidedNav.git
cd GuidedNav/code
python tools/download_assets.py --group inference
```

The downloader requires Python 3.9+ and no account or token. It resumes downloads,
checks SHA256 and reassembles the weights automatically. Set up the CUDA runtime
and run your RGB sequence using the [inference guide](https://anonymousauthor-s.github.io/GuidedNav/guide/inference/).

| I want to… | Open |
| --- | --- |
| Run the policy on my images | [Inference quick start](https://anonymousauthor-s.github.io/GuidedNav/guide/inference/) |
| Evaluate R2R-CE / RxR-CE | [Habitat evaluation](https://anonymousauthor-s.github.io/GuidedNav/guide/inference/#habitat-evaluation) |
| Reproduce Stage 3 training | [Training setup and launch](https://anonymousauthor-s.github.io/GuidedNav/guide/training/) |
| Reconstruct the selected training data | [Data preparation](https://anonymousauthor-s.github.io/GuidedNav/guide/data/) |
| Download models and annotations | [Downloads and integrity](https://anonymousauthor-s.github.io/GuidedNav/guide/models/) |
| Install the required environments | [Environment guide](https://anonymousauthor-s.github.io/GuidedNav/guide/environment/) |

## Model downloads

All model assets are being uploaded to the [public release](https://github.com/anonymousauthor-s/GuidedNav/releases/tag/stage3-epoch1).

| Model directory | Contents |
| --- | --- |
| `weights/navigation-base/` | Full BF16 navigation-pretrained vision-language backbone |
| `weights/stage3-epoch1/` | Reported final LoRA adapter, temporal gate, auxiliary projections and processor |
| `weights/stage3-initialization/` | Adapter and auxiliary weights used to start Stage 3 training |

The full inference model is **16.60 GiB**. This includes the vision encoder and
language model, plus final adapter and auxiliary weights. The shared download
also includes training initialization, bringing installed model payloads to
**16.86 GiB**. It is not an adapter-only or quantized release. Use the supplied
navigation base with its matching adapter; a vanilla Qwen model cannot replace it.
See [downloads](https://anonymousauthor-s.github.io/GuidedNav/guide/models/) for disk requirements and checksums.

## Training data

```bash
python tools/download_assets.py --group data
```

The add-on contains the **complete frozen Stage 3 training selection**:

| Source | Training rows |
| --- | ---: |
| Guided-R2R selection | 157,465 |
| Guided-RxR English selection | 472,323 |
| **Total** | **629,788** |

It includes **379,428 landmark masks, 18,656 collected trajectories/actions**, exact
instruction IDs/hashes, selection arrays and prompt choices. This is the
selection used by Stage 3, not the entire Guided annotation corpus. Download
official instructions and licensed scenes from upstream, then recover the
training text and render RGB with the supplied tools. The [data guide](https://anonymousauthor-s.github.io/GuidedNav/guide/data/)
explains the exact correspondence and all preparation steps.

## Train Stage 3

The reference recipe uses one epoch, 32 training GPUs, global batch 256, BF16,
LoRA rank 32 and DeepSpeed ZeRO-2. Earlier navigation training is represented by
the supplied pretrained assets. Start with the [training guide](https://anonymousauthor-s.github.io/GuidedNav/guide/training/)
to install the environment, obtain the training-only VGGT teacher, validate the
data and launch training. The exported model is `stage3-epoch1/`.

## Repository and validation

```text
code/    Inference, training, evaluation, download tools and setup instructions
docs/    Anonymous project website and browser-accessible guides
```

Release checks cover model hashes, CPU imports/CLI, data alignment and recovered
training inputs. Full GPU retraining, complete benchmark evaluation and
full-dataset RGB rendering were not rerun during packaging. The [guides](https://anonymousauthor-s.github.io/GuidedNav/guide/inference/)
describe these limits. Upstream copyright and license notices are retained;
model/data terms are included with the [release assets](https://github.com/anonymousauthor-s/GuidedNav/releases/tag/stage3-epoch1).

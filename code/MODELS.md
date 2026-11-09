# Model downloads

> **Upload in progress.** Model and data download links will become active after all files pass the public-download checks.


The full navigation model and both Stage 3 adapters are being uploaded to the
[public release](https://github.com/anonymousauthor-s/GuidedNav/releases/tag/stage3-epoch1). No account or access token is required.

## Download and install

Clone the code, then run the standard-library downloader with Python 3.9 or later:

```bash
git clone https://github.com/anonymousauthor-s/GuidedNav.git
cd GuidedNav/code
python tools/download_assets.py --group inference
```

The tool resumes interrupted downloads, verifies every asset with SHA256,
reassembles the original tensor files, and installs the weights at the repository
root. Rerun the same command after an interruption. Reserve about **24 GiB of free
disk space** for model installation, including temporary transport parts.

| Directory | Contents | Required for |
| --- | --- | --- |
| `weights/navigation-base/` | Full BF16 navigation-pretrained vision-language backbone and processor | Inference and training |
| `weights/stage3-epoch1/` | Final Stage 3, epoch 1 LoRA adapter, temporal gate, auxiliary projections and processor | Reported policy |
| `weights/stage3-initialization/` | Initialization adapter, auxiliary weights and processor | Stage 3 training |

The model download includes all three directories. The initialization is a small
extra included in the shared support archive; inference loads only the navigation
base and final epoch. `--group training` downloads the same model files.

## What does 16.60 GiB mean?

**16.60 GiB is the full inference model**, including the navigation backbone and
final adapter/auxiliary weights. It includes the vision encoder and language
model in BF16. It is not a quantized model or an adapter-only download. Including
the training initialization brings the installed model payloads to **16.86 GiB**.
Archive compression and transport splitting can make download sizes differ.

Use the supplied navigation-pretrained backbone with its matching adapter.
An unmodified Qwen3-VL model cannot replace that backbone. The frozen VGGT teacher
is needed only for training and is downloaded separately in the [data guide](../data/).

## Training data

```bash
python tools/download_assets.py --group data
```

This installs the **complete frozen Stage 3 selection**: 629,788 annotated rows,
379,428 masks, 18,656 collected trajectories/actions and exact prompt-choice
metadata. It is a roughly **459 MiB** compressed add-on, rather than the full
Guided-R2R/Guided-RxR annotation corpus. Original instructions and licensed scenes
are obtained from upstream. Follow the [data guide](../data/) to reconstruct text,
render RGB and prepare training records. Allow additional disk space for the
extracted labels and rendered observations.

To fetch models and the add-on together:

```bash
python tools/download_assets.py --group all
```

## Manual downloads and integrity

The [release assets](https://github.com/anonymousauthor-s/GuidedNav/releases/tag/stage3-epoch1) include the model-support archive, data add-on,
18 transport parts for the four original backbone tensor files, a download
manifest, and SHA256SUMS. Use the downloader for automatic ordered reassembly.
The assembled tensor files are byte-identical to the exported navigation model.

- [Data add-on](https://github.com/anonymousauthor-s/GuidedNav/releases/download/stage3-epoch1/guidednav-stage3-data-addon.tar.gz)
- [Download manifest](https://github.com/anonymousauthor-s/GuidedNav/releases/download/stage3-epoch1/download-manifest.json)
- [SHA256 checksums](https://github.com/anonymousauthor-s/GuidedNav/releases/download/stage3-epoch1/SHA256SUMS)
- [Model terms and attribution](https://github.com/anonymousauthor-s/GuidedNav/releases/download/stage3-epoch1/MODEL_TERMS.md)
- [Data terms and attribution](https://github.com/anonymousauthor-s/GuidedNav/releases/download/stage3-epoch1/DATA_TERMS.md)

The navigation adaptations and added supervision are shared for non-commercial
research, subject to the listed upstream terms. Original Qwen Apache-2.0 rights
and original benchmark rights remain unchanged. Matterport3D scenes must be
obtained through the official access process.

Continue with [inference](../inference/) or [Stage 3 training](../training/).

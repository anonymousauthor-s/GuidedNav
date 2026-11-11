# Model and data downloads

We release the **GuidedNav-8B checkpoint** for inference and R2R-CE / RxR-CE evaluation.

## Download GuidedNav-8B

```bash
git clone https://github.com/anonymousauthor-s/GuidedNav.git
cd GuidedNav/code
python tools/download_assets.py --group inference
```

The downloader requires Python 3.9+, supports resuming and verifies SHA256.
Downloads are public. Reserve approximately 24 GiB of free disk space during installation.

The checkpoint is **16.60 GiB in BF16**, organized as follows:

| Directory | Contents |
| --- | --- |
| `weights/navigation-base/` | Vision-language backbone and processor |
| `weights/stage3-epoch1/` | Stage 3 adapter, temporal state, auxiliary weights and processor |

The inference loader uses both directories. The downloader automatically
reassembles the large backbone files from their transport parts.

Continue with [inference and evaluation](README.md#inference).

## Download training annotations

```bash
python tools/download_assets.py --group data
```

The **457 MiB** Stage 3 annotation package contains 629,788 samples,
379,428 masks, 18,656 trajectories, recorded actions and prompt metadata.
Download the original instructions and scenes from their official sources,
then reconstruct the training inputs using the [data guide](DATA.md).
That guide also describes Stage 1/2 expert, DAgger and VQA preparation.

Use `--group all` to download the checkpoint and annotation package together.
For training, follow the [three-stage guide](TRAINING.md).

## Files

- [GitHub release](https://github.com/anonymousauthor-s/GuidedNav/releases/tag/stage3-epoch1)
- [Model support files](https://github.com/anonymousauthor-s/GuidedNav/releases/download/stage3-epoch1/guidednav-final-model-support.tar.gz)
- [Training annotations](https://github.com/anonymousauthor-s/GuidedNav/releases/download/stage3-epoch1/guidednav-stage3-data-addon.tar.gz)
- [Download manifest](https://github.com/anonymousauthor-s/GuidedNav/releases/download/stage3-epoch1/download-manifest.json)
- [SHA256 checksums](https://github.com/anonymousauthor-s/GuidedNav/releases/download/stage3-epoch1/SHA256SUMS)
- [Model terms](https://github.com/anonymousauthor-s/GuidedNav/releases/download/stage3-epoch1/MODEL_TERMS.md)
- [Data terms](https://github.com/anonymousauthor-s/GuidedNav/releases/download/stage3-epoch1/DATA_TERMS.md)

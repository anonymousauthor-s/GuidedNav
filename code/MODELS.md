# Final model and data downloads

The only released policy is the **final paper model — Stage 3, epoch 1**,
provided for inference and R2R-CE / RxR-CE evaluation. No Stage 1, Stage 2 or
Stage 3 initialization checkpoint is distributed.

## Download the final paper model

```bash
git clone https://github.com/anonymousauthor-s/GuidedNav.git
cd GuidedNav/code
python tools/download_assets.py --group inference
```

No account or token is required. Python 3.9+ is sufficient for the downloader.
It resumes interrupted downloads, checks SHA256 and reassembles the original
tensor files. Reserve approximately 24 GiB of free disk space during installation.

The complete final model is **16.60 GiB**, including the BF16 vision encoder,
language model, final LoRA adapter and learned temporal/auxiliary state. Its
loader uses two component directories:

| Component | Contents |
| --- | --- |
| `weights/navigation-base/` | Full backbone and processor required by the final policy |
| `weights/stage3-epoch1/` | Matching final adapter, temporal state, auxiliary weights and processor |

These two components jointly form **one final evaluation model**. They are
not alternative checkpoints. Keep them together; a vanilla Qwen model cannot
replace the matching backbone for evaluation of the released policy.

The model-support archive contains only those final-model components and
metadata. The four large backbone tensors use 18 transport parts; the downloader
restores their original files automatically.

Continue with [inference and evaluation](README.md#inference).

## Download the training annotation add-on

```bash
python tools/download_assets.py --group data
```

The approximately 457 MiB add-on contains 629,788 selected Stage 3 rows,
379,428 masks, 18,656 collected trajectories/actions and prompt metadata.
Obtain original instructions and licensed scenes from upstream, then recover
text and render RGB using the [data guide](DATA.md).
This is the selected Stage 3 add-on, rather than the full Guided annotation
corpus or the Stage 1/2 expert, DAgger and VQA corpora.

`--group all` fetches the final evaluation model and this data add-on together.
Training starts from the official Qwen model and your locally produced
Stage 1/2 outputs; follow [three-stage training](TRAINING.md).

## Release files and integrity

- [Release assets](https://github.com/anonymousauthor-s/GuidedNav/releases/)
- [Final-model support archive](https://github.com/anonymousauthor-s/GuidedNav/releases/download/stage3-epoch1/guidednav-final-model-support.tar.gz)
- [Training annotation add-on](https://github.com/anonymousauthor-s/GuidedNav/releases/download/stage3-epoch1/guidednav-stage3-data-addon.tar.gz)
- [Download manifest](https://github.com/anonymousauthor-s/GuidedNav/releases/download/stage3-epoch1/download-manifest.json)
- [SHA256 checksums](https://github.com/anonymousauthor-s/GuidedNav/releases/download/stage3-epoch1/SHA256SUMS)
- [Model terms](https://github.com/anonymousauthor-s/GuidedNav/releases/download/stage3-epoch1/MODEL_TERMS.md)
- [Data terms](https://github.com/anonymousauthor-s/GuidedNav/releases/download/stage3-epoch1/DATA_TERMS.md)

Upstream notices and terms are retained. Obtain Matterport3D scenes through
the official access process; the add-on does not redistribute those scenes.

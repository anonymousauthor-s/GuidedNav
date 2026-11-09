# Environments

Training and Habitat use separate runtimes. The original training image did not contain Habitat. Do not force old Habitat simulator dependencies into the training image.

## Final-stage training

`config/training_environment.json` records package versions read from the exact image used by the selected final-stage run. The important versions are Python 3.12.3, Torch 2.10.0a0+b558c986e8.nv25.11, TorchVision 0.25.0a0+7a13ad0f, Transformers 4.57.6, PEFT 0.19.1, Accelerate 1.14.0, DeepSpeed 0.16.5 and FlashAttention 2.7.4.post1+25.11. This is a development Torch build, so a normal `pip install torch==2.10.0` does not recreate it.

The public [NVIDIA PyTorch 25.11 container](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-25-11.html) contains the same Torch build and CUDA 13.0.2. `Dockerfile.training` provides an application-environment reconstruction based on that public image:

```bash
docker build -f Dockerfile.training -t guidednav-training .
docker run --rm --gpus all --ipc=host --network host \
  -v /absolute/path/to/release:/release \
  -e PYTHONPATH=/release/code/src:/release/code/third_party/vggt \
  --entrypoint python guidednav-training \
  /release/code/tools/check_environment.py --training
```

The Dockerfile has not been rebuilt during packaging. Compare the check output with the recorded environment before the dataset/gradient smoke. Configure NVIDIA drivers compatible with the container's CUDA release according to NVIDIA's documentation. Distributed launch requires the same release and data paths on every node, working NCCL transport and enough memory for an 8B VLM plus frozen VGGT teacher. The reference recipe used 32 GPUs; a smaller hardware configuration needs a separately validated recipe.

## Frame-policy inference

The deployed frame runtime recorded Torch 2.8.0+cu128, Transformers 4.57.6, PEFT 0.17.1 and Accelerate 1.10.1. Create a separate Python 3.10/3.11 CUDA environment:

```bash
python -m venv .venv-inference
source .venv-inference/bin/activate
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-inference.txt
export PYTHONPATH="$PWD/src"
python tools/check_environment.py
```

The ancillary NumPy/Pillow pins in the inference requirements are packaging choices; only the four core runtime versions above were recorded from the deployed service. Frame inference does not import Habitat. The original frame service used SDPA; the Habitat benchmark path requests FlashAttention 2, which must be built or installed for its Torch/CUDA ABI.

## Habitat rendering and benchmark evaluation

The reference evaluator targets Habitat-Lab 0.2.4 and Habitat-Sim 0.2.4. Their old binary and Python dependencies are easier to keep in a separate Python 3.9 environment. Follow the [Habitat-Lab 0.2.4 setup](https://github.com/facebookresearch/habitat-lab/tree/v0.2.4) and [Habitat-Sim 0.2.4 build/install instructions](https://github.com/facebookresearch/habitat-sim/tree/v0.2.4). A starting simulator environment is:

```bash
conda create -n guidednav-habitat python=3.9 cmake=3.22
conda activate guidednav-habitat
conda install habitat-sim=0.2.4 headless -c conda-forge -c aihabitat
git clone --branch v0.2.4 --depth 1 https://github.com/facebookresearch/habitat-lab.git third_party/habitat-lab
python -m pip install -e third_party/habitat-lab/habitat-lab
python -m pip install -e third_party/habitat-lab/habitat-baselines
python -m pip install scipy pillow
```

Rendering from released poses uses `habitat_sim` directly and does not need the VLM. The benchmark additionally needs a compatible CUDA Torch, Transformers 4.57.6, PEFT, Accelerate and FlashAttention build. `config/habitat_environment.json` and `requirements-habitat-vlm.txt` record the pre-existing evaluation environment in which packaging validated the evaluator imports, CLI and both YAML compositions without a visible GPU: Python 3.9.23, Torch 2.7.0+cu128, TorchVision 0.22.0+cu128, Transformers 4.57.6, PEFT 0.17.1, Accelerate 1.10.1, FlashAttention 2.8.3, Habitat-Lab/Sim 0.2.4. Install matching CUDA Torch/TorchVision first and build FlashAttention for that ABI. The supplied nDTW measure implements the radius-1 FastDTW recurrence in Python and does not require the external `fastdtw` package.

A fresh full Habitat+VLM environment build has not been validated during packaging, and the recorded environment read is not a historical immutable image freeze. Check simulator-only rendering, then a one-episode benchmark before launching the complete evaluation. Do not silently upgrade Habitat or substitute an evaluator with different episode/action settings.

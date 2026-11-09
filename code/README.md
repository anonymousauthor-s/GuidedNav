# GuidedNav: anonymous review code

This package contains the selected final model's training stage, frame-policy inference, and Habitat evaluation. The selected model is **base-19450 → initialization-6656 → final-2461**, before any later continuation. Final-2461 is a LoRA adapter plus learned temporal auxiliary weights; it is not a standalone 8B model. A vanilla Qwen checkpoint cannot replace base-19450.

Large weights and annotated training data are separate release artifacts. This source archive alone does not contain them. See [DATA.md](DATA.md), [ENVIRONMENT.md](ENVIRONMENT.md), and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

Expected layout, with this directory named `code`:

```text
release/
  code/
  weights/base-19450/
  weights/initialization-6656/
  weights/final-2461/
  data/r2r/                 # frozen records, arrays, masks and agent trajectories
  data/rxr/
  assets/vggt/vggt.pt        # training teacher, downloaded separately
```

## Inference from RGB frames

Install the frame-policy environment described in ENVIRONMENT.md. From this directory:

```bash
export PYTHONPATH="$PWD/src"
python tools/check_environment.py
python -m qwenvl.infer_frames \
  --base ../weights/base-19450 \
  --checkpoint ../weights/final-2461 \
  --instruction 'Walk through the doorway and stop beside the table.' \
  --frames-list /absolute/path/to/frames.txt
```

`frames.txt` lists one RGB filename per line in **chronological trajectory order**, including the current frame last. Relative paths are resolved against the list file. Supply every accumulated observation when possible: the verified policy uniformly samples history and separately takes the recent four-frame temporal window. Passing only a few recent images changes the history protocol. The output contains up to four actions: 0 STOP, 1 forward 0.25 m, 2 left 15°, 3 right 15°. The default frame CLI uses BF16 and SDPA, as in the deployed frame runtime. The benchmark evaluator uses FlashAttention 2.

The loader verifies the pinned SHA256 of all four base shards, the selected adapter and its auxiliary weights, then checks the vocabulary, base loading result, exact loaded LoRA tensors, saved temporal recipe and learned gate. Successful hashes are cached locally against absolute path, file size and nanosecond mtime to avoid repeatedly reading the 17.5 GB base on every rank. `tools/verify_model_identity.py --force --base ... --checkpoint ...` forces a full rehash. A missing, changed or incorrect payload fails closed. The loader merges the adapter and restores the gate; loading the adapter alone is insufficient. VGGT and training-only subtask/object supervision are not required for inference.

## Reproduce the final training stage

Prepare the annotated data, RGB and teacher using DATA.md, then run the data decode smoke before training. The reference stage used **32 GPUs (4 nodes × 8), batch 1/GPU, accumulation 8, global batch 256**, a fresh optimizer and cosine schedule, LR 2e-4, warmup 0.03, one epoch, seed 42 and all-linear LoRA rank 32/alpha 16/dropout 0. It consumed 157,465 R2R and 472,323 RxR annotated rows, finishing at step 2461. No additional DAgger or VQA rows were added in this final stage; base-19450 and initialization-6656 retain their earlier training history.

On all four nodes, using the same absolute data/model mount paths:

```bash
export FULL_MANIFEST=/absolute/path/to/manifest.local.json
export MASTER_ADDR=training-rendezvous-host
export NODE_RANK=0  # 0, 1, 2 or 3 on the corresponding node
export NCCL_SOCKET_IFNAME=your_training_network_interface
export FULL_OUTPUT_DIR=/absolute/path/to/new-empty-run
bash scripts/train_final.sh
```

The training entrypoint loads initialization-6656's adapter and **all three** auxiliary modules, verifies their exact values, and starts a fresh optimizer/scheduler. It rejects a changed formal GPU count, accumulation, LR, seed or LoRA recipe. To resume this new stage, set `FULL_RESUME` to one of its complete trainer checkpoints, including optimizer partitions; the released initialization snapshot is for a new stage and is not an optimizer-resume checkpoint.

For a two-update smoke, set `FULL_SMOKE=1`, a separate empty output directory, and optionally `NNODES=1 NPROC_PER_NODE=1`. This exercises data, losses and gradient paths, subject to GPU memory. A smoke with fewer ranks changes the effective batch and schedule and is **not** a reproduction of the final model. Do not use its checkpoint as final-2461.

## Habitat evaluation

Install the separate Habitat environment in ENVIRONMENT.md, arrange the official R2R/RxR VLN-CE episodes and licensed MP3D scenes as described in DATA.md, and run:

```bash
bash scripts/evaluate.sh r2r
bash scripts/evaluate.sh rxr
```

Both commands use `val_unseen`, maximum 500 episode steps, history 8, four future actions and four executed actions. RxR uses all 3669 English guide episodes with the saved protocol's 0.88 m / 0.18 m collision body, explicitly audited navigation meshes, sliding enabled and RGB/action settings described in DATA.md. This is the research protocol used by the reported result; it is not the RxR challenge sensor/action configuration. Build its navigation-mesh audit before evaluation. Use a new `EVAL_OUTPUT` directory for each run: the evaluator appends episode results. `EVAL_WORKERS` controls distributed evaluation; allocate one simulator worker per rendering GPU. `--max_episodes 1` is a diagnostic cap, not the full benchmark. For separate model/rendering GPUs, the inherited evaluator supports `REMOTE_SIM=1`, `MODEL_GPUS` and `SIM_GPUS`.

Historical final-model results are R2R: NE 3.828, SR 65.416%, SPL 59.066% over 1839 episodes; RxR: NE 4.404, SR 64.922%, SPL 54.334%, nDTW 69.203% over 3669 episodes. These are recorded results, not a claim that this newly packaged release has already been rerun end to end.

## Validation boundary

The package uses the original final-stage training source and the verified final evaluator's prompt, history, temporal hook and greedy generation methods. Packaging changes remove private paths, replace the legacy dataset registry with explicit manifests, remove inactive alternative model loaders, add a strict final-model loader, and disable unrelated asynchronous cluster evaluation. Source/CLI imports and static checks are recorded in `VALIDATION.json`. Full GPU retraining, full benchmark evaluation, and RGB re-rendering of the entire dataset have not been rerun during packaging. Small numerical differences can arise from CUDA kernels, simulator versions and image encoding; the original weights remain the reference result.

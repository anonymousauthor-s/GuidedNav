"""Collect NEW train-only DAgger trajectories using your Stage 1 full model.

The Stage 1 policy visits states, while a simulator reference-path expert
provides a coherent expert_chunks label at every visited state. Executed policy
actions are retained separately and must never be used as Stage 2 targets.
Prompt construction, history sampling, action parsing and greedy generation
reuse VLNEvaluator. Stage 1 has no GuidedNav temporal fusion or LoRA hooks.

This collection path does not recreate the exact historical 2.97M-row pool.
Use separate physical GPUs for model inference and simulator rendering.

Example (run from code/ with the Habitat/inference environment active):
  PYTHONPATH=src python src/data_process/dagger_rollout.py \
    --model_path ../training/stage1/run/stage1-epoch1 \
    --dataset r2r_dagger --data_path ../data/datasets/r2r/train/train.json.gz \
    --scenes_dir ../data/scene_datasets --output_path ../data/dagger/r2r \
    --model_gpu 1 --sim_gpu 0 --rank 0 --world_size 1

For RxR use --dataset rxr_dagger and its train_guide.json.gz. The collector
selects config/vln_rxr_train.yaml, using the English guide dataset and ordinary
HabitatSim; the final benchmark's val_unseen-only navmesh audit is not used.
Merge workers with --output_path <directory> --merge, then build targets:
  PYTHONPATH=src python src/data_process/build_vln_samples.py \
    --dataset_dir ../data/dagger/r2r --action_source expert_chunks \
    --memory_size 8 --num_future_steps 4 --seed 42

Output: images/<scan>_<dataset>_<id>/rgb/001.jpg, traj/*.npz (executed actions,
poses, expert_chunks, action_sources), annotations_<rank>.json and summaries.
"""
import argparse
import gzip
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

# Same belt-and-suspenders path setup as the evaluator: make `qwenvl.*` and
# `habitat_extensions` importable when launched as a plain script, and this
# file's own directory importable so `merge_shards` can be reused verbatim.
_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1]))       # code/src
sys.path.insert(0, str(_HERE.parent))           # repo/src/data_process


def _configure_gpus(model_gpu, sim_gpu):
    """Pin model and sim to their cards BEFORE torch/habitat are imported.

    ``resolve_sim_gpu()`` reads ``SIM_GPUS``; with a single entry it returns
    that card regardless of LOCAL_RANK, which is what we want since each worker
    is an independent process rather than a torchrun rank. Absolute device
    indices are used throughout, so the launcher must expose ALL cards to every
    worker (do not set a per-worker CUDA_VISIBLE_DEVICES).
    """
    if model_gpu == sim_gpu:
        raise SystemExit(
            f"--model_gpu and --sim_gpu must be different physical cards "
            f"(both are {model_gpu}). Sharing one card makes env.step ~400x "
            f"slower: CUDA compute preempts habitat's GL render."
        )
    os.environ["REMOTE_SIM"] = "1"
    os.environ["SIM_GPUS"] = str(sim_gpu)
    os.environ.setdefault("MODEL_GPUS", str(model_gpu))


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _load_stage1_policy(args):
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    from qwenvl.model_identity import verify_training_backbone

    path = Path(args.model_path).resolve()
    recipe_path = path / 'training_recipe.json'
    recipe = json.loads(recipe_path.read_text(encoding='utf-8'))
    if (type(recipe.get('stage')) is not int or recipe['stage'] != 1
            or recipe.get('smoke') is not False or recipe.get('epochs') != 1
            or recipe.get('policy_update') != 'full_parameters'):
        raise ValueError('DAgger must use your completed, non-smoke Stage 1 full-model export')
    verify_training_backbone(path)
    processor = AutoProcessor.from_pretrained(path, min_pixels=args.min_pixels,
                                            max_pixels=args.max_pixels)
    model, info = Qwen3VLForConditionalGeneration.from_pretrained(
        path, dtype=torch.bfloat16, attn_implementation='flash_attention_2',
        output_loading_info=True)
    if any(info.get(key) for key in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
        raise ValueError('Stage 1 full-model weights did not load exactly: ' + str(info))
    model.to(device=f'cuda:{args.model_gpu}').eval()
    model.config.use_cache = True
    return model, processor, processor.tokenizer


def merge_collection(output_path, keep_shards=False):
    """Merge incrementally without importing Habitat or discarding old batches."""
    output = Path(output_path)
    final = output / 'annotations.json'
    merged = {}
    if final.is_file():
        for row in json.loads(final.read_text(encoding='utf-8')):
            merged[int(row['id'])] = row
    shards = sorted(output.glob('annotations_*.json'))
    if not shards:
        raise ValueError('No annotation shards to merge')
    for shard in shards:
        for row in json.loads(shard.read_text(encoding='utf-8')):
            key = int(row['id'])
            if key in merged and merged[key] != row:
                raise ValueError('Conflicting annotation identity across collection shards: ' + str(key))
            merged[key] = row
    partial = output / 'annotations.json.partial'
    partial.write_text(json.dumps(sorted(merged.values(), key=lambda row: int(row['id'])),
                                  ensure_ascii=False) + '\n', encoding='utf-8')
    partial.replace(final)
    if not keep_shards:
        for shard in shards:
            shard.unlink()
    print(f'Merged {len(shards)} shards: {len(merged)} collected training trajectories')


def _check_training_inputs(args):
    if args.split != 'train':
        raise ValueError('DAgger collection is restricted to the training split')
    data = Path(args.data_path).resolve()
    forbidden = {'val_seen', 'val_unseen', 'test', 'test_challenge'}
    if (any(part.lower() in forbidden for part in data.parts)
            or not data.name.startswith('train') or not data.name.endswith('.json.gz')):
        raise ValueError('Use the official train*.json.gz episodes, never validation/test data')
    with gzip.open(data, 'rt', encoding='utf-8') as handle:
        payload = json.load(handle)
    episodes = payload.get('episodes')
    if not isinstance(episodes, list) or not episodes:
        raise ValueError('Training episode file is empty or has the wrong schema')
    if args.dataset == 'rxr_dagger':
        episodes = [row for row in episodes if
                    (row.get('instruction') or {}).get('language') in {'en-US', 'en-IN'}]
        if not episodes:
            raise ValueError('RxR collection requires English guide training episodes')
    scenes = Path(args.scenes_dir).resolve()
    if not scenes.is_dir():
        raise NotADirectoryError('Missing scene root: ' + str(scenes))
    args.data_path, args.scenes_dir = str(data), str(scenes)
    os.environ['VLN_TRAIN_DATA_PATH'] = args.data_path
    os.environ['GUIDEDNAV_SCENES_DIR'] = args.scenes_dir


def _write_collection_recipe(args):
    output = Path(args.output_path)
    output.mkdir(parents=True, exist_ok=True)
    recipe = dict(vars(args))
    recipe['format'] = 'guidednav_new_train_dagger_v1'
    recipe['data_sha256'] = _file_sha256(args.data_path)
    recipe['config_sha256'] = _file_sha256(args.habitat_config_path)
    if args.model_path:
        recipe['stage1_recipe_sha256'] = _file_sha256(Path(args.model_path) / 'training_recipe.json')
    recipe['historical_pool_identity'] = False
    path = output / f'collection_recipe_{args.rank}.json'
    if path.exists() and json.loads(path.read_text(encoding='utf-8')) != recipe:
        raise ValueError('Collection input, policy or recipe changed; use a new output directory')
    path.write_text(json.dumps(recipe, indent=2) + '\n', encoding='utf-8')


def _main():
    args = _parse_args()
    if args.merge:
        merge_collection(args.output_path, keep_shards=args.keep_shards)
        return
    _check_training_inputs(args)
    _configure_gpus(args.model_gpu, args.sim_gpu)
    import torch
    from transformers import set_seed
    set_seed(args.seed + args.rank)
    model = processor = tokenizer = None
    if args.beta < 1.0:
        torch.cuda.set_device(args.model_gpu)
        model, processor, tokenizer = _load_stage1_policy(args)
    else:
        print('[dagger] beta=1.0: expert-only diagnostic; policy is not loaded', flush=True)
    _write_collection_recipe(args)
    collector = DaggerCollector(
        config_path=args.habitat_config_path, split=args.split,
        output_path=args.output_path, model=model, processor=processor,
        tokenizer=tokenizer, args=args)
    collector.collect(rank=args.rank, world_size=args.world_size, limit=args.limit)


class DaggerCollector:
    """Thin wrapper that owns a ``VLNEvaluator`` for its policy machinery.

    Composition, not inheritance of the run loop: ``eval_action`` and
    ``collect_episode`` share nothing but the env, while the *prompting* half
    must be shared exactly. So the evaluator is instantiated and its prompting
    methods are called directly.
    """

    def __init__(self, config_path, split, output_path, model, processor,
                 tokenizer, args):
        from qwenvl.eval.eval_discrete_action import VLNEvaluator

        self.args = args
        self.output_path = output_path
        self.dataset = args.dataset
        self.chunk_size = args.chunk_size

        # The evaluator writes its own output_path (for result.json, unused
        # here) and reads args.{num_history,num_future_steps,num_exec,
        # max_new_tokens,max_episodes,save_video,scenes_dir}. env_num=1 /
        # epoch=0 are inert on the paths we use.
        self.ev = VLNEvaluator(
            config_path=config_path,
            split=split,
            env_num=1,
            output_path=output_path,
            model=model,
            processor=processor,
            tokenizer=tokenizer,
            args=args,
            epoch=0,
        )
        if model is not None:
            import torch
            # VLNEvaluator hardcodes torch.device("cuda") (index-free), which
            # resolves to the *current* device. Be explicit: the model was
            # loaded onto args.model_gpu.
            self.ev.device = torch.device(f"cuda:{args.model_gpu}")

        if args.data_path:
            # The evaluator only overrides `split` and `scenes_dir`; the episode
            # file itself comes from config/vln_r2r.yaml, which points at r2r.
            # RxR needs a different one. Safe to mutate here — the config is not
            # consumed until config_env(). Pass a FULLY-RESOLVED path (split
            # already substituted): OmegaConf mangles a "{split}/{split}"
            # template injected via .update() and habitat's .format() then
            # raises. Same trap rollout.sh documents.
            import habitat
            with habitat.config.read_write(self.ev.config):
                self.ev.config.habitat.dataset.data_path = args.data_path

        if self.ev.config.habitat.dataset.split != 'train':
            raise ValueError('Training collection must not consume a validation/test split')
        expected_dataset = 'RxREnglishGuideVLNCE-v1' if args.dataset == 'rxr_dagger' else 'R2RVLN-v1'
        if self.ev.config.habitat.dataset.type != expected_dataset:
            raise ValueError('Collection configuration has the wrong dataset type')
        if self.ev.config.habitat.simulator.type == 'RxRFullGeometrySim-v1':
            raise ValueError('The final val_unseen-only RxR navmesh audit cannot be used for training collection')
        self.rng = np.random.RandomState(args.seed + args.rank)

    # -- one episode --------------------------------------------------------
    def collect_episode(self, env, ep_desc):
        """Roll the policy out, relabel every frame, return the arrays.

        Returns ``(record, stats)`` or ``(None, stats)`` when the episode must
        be dropped. ``record`` holds everything the npz needs.

        Episode structure::

            [policy phase]  frames 0..Tp   states the policy produces
            [expert tail]   frames Tp..T   the expert walks it home

        The tail earns its keep twice: it is the recovery supervision DAgger
        exists for, and it is the only source of near-goal STOP labels (a policy
        that never arrives contributes none — and STOP is this model's known
        weak spot). It runs after EVERY policy phase, however that phase ended,
        so every kept episode terminates in a genuine expert STOP.
        """
        a = self.args
        instruction = ep_desc.instruction.instruction_text

        t0 = time.perf_counter()
        obs = env.reset()
        t_reset = time.perf_counter() - t0
        # goals / reference_path live on the habitat Episode, which only exists
        # in the sim subprocess — RemoteEnv's parent-side descriptor carries just
        # scene/id/instruction. Read them once, after the reset made them current.
        geom = env.episode_geom()

        frames = []                 # PIL, one per frame — fed to the policy
        rgbs = []                   # ndarray, one per frame — written to disk
        positions, rotations = [], []
        region_ids, region_cats, geo = [], [], []
        chunks = []                 # [n, K] expert plans
        actions = [-1]
        sources = ["reset"]

        pending = []
        phase = "policy"
        n_policy = n_tail = n_disagree = 0
        policy_stop_geo = None
        policy_end = "?"            # why the policy phase handed over
        end_reason = "?"            # why the episode itself ended
        start_geo = None
        t_gen = t_step = t_expert = 0.0
        n_gen = 0

        while True:
            # ---- record frame c = len(frames) --------------------------------
            rgb = np.asarray(obs["rgb"])
            rgbs.append(rgb)
            frames.append(Image.fromarray(rgb).convert("RGB"))

            fs = env.frame_state()
            positions.append(np.asarray(fs["position"], dtype=np.float32))
            rotations.append(np.asarray(fs["rotation"], dtype=np.float32))
            region_ids.append(fs["region_id"])
            region_cats.append(fs["region_category"])
            geo.append(float(fs["geodesic"]))

            if start_geo is None:
                start_geo = geo[0]
                if not np.isfinite(start_geo):
                    return None, {"drop": "start_geodesic_inf"}

            _e0 = time.perf_counter()
            # raises -> caller drops the episode
            chunk = env.expert_chunk(self.chunk_size, stop_dist=a.goal_stop_dist)
            t_expert += time.perf_counter() - _e0
            chunks.append([int(x) for x in chunk])

            # Checked HERE, after recording, not after stepping: habitat ends the
            # episode at max_episode_steps (500), and breaking straight out of the
            # step would leave the post-step frame unrecorded — actions would be
            # one longer than positions and the episode would fail the alignment
            # check. Neither expert_chunk nor frame_state touches the task's
            # over-state, so both are still valid on the final frame.
            if env.episode_over:
                end_reason = "episode_over"
                break

            # ---- choose the action at frame c --------------------------------
            if phase == "policy":
                if n_policy >= a.max_policy_steps:
                    policy_end, phase, pending = "policy_budget", "expert", []
                elif geo[-1] > start_geo + a.max_geo_regress:
                    policy_end, phase, pending = "diverged", "expert", []

            if phase == "policy":
                use_expert = a.beta >= 1.0 or self.rng.random_sample() < a.beta
                if use_expert:
                    act, src = chunk[0], "expert"
                    pending = []       # the model's queued plan is now stale
                else:
                    if not pending:
                        _g0 = time.perf_counter()
                        text = self.ev._generate_chunk(instruction, frames)
                        t_gen += time.perf_counter() - _g0
                        n_gen += 1
                        acts = self.ev.parse_actions(text) or [0]
                        acts = acts[: self.ev.num_future_steps]
                        if self.ev.num_exec > 0:
                            acts = acts[: self.ev.num_exec]
                        pending = list(acts)
                    act, src = pending.pop(0), "policy"
                    if act != chunk[0]:
                        n_disagree += 1

                if act == 0:
                    # Any policy STOP ends the policy phase — stray or correct,
                    # the tail takes over and finishes the route. The frame itself
                    # is kept with its expert relabel: a state where the policy
                    # wrongly said STOP and the expert says "keep going" is the
                    # single most valuable label in this dataset. The STOP is
                    # neither executed nor recorded, so `actions` never gains a
                    # 0 the policy chose.
                    policy_stop_geo = geo[-1]
                    policy_end, phase, pending = "policy_stop", "expert", []
                else:
                    n_policy += 1

            if phase == "expert":
                # `chunk` was computed from THIS state, so chunk[0] is already
                # the expert's action here — no second query.
                if a.tail_goal_only:
                    chunk = [int(x) for x in
                             env.expert_chunk(self.chunk_size, goal_only=True,
                                              stop_dist=a.goal_stop_dist)]
                    chunks[-1] = chunk
                act, src = chunk[0], "expert"
                if act == 0:
                    end_reason = "expert_stop"
                    break
                if n_tail >= a.max_tail_steps:
                    end_reason = "tail_budget"
                    break
                n_tail += 1

            # ---- execute -----------------------------------------------------
            _s0 = time.perf_counter()
            obs = env.step(act)
            t_step += time.perf_counter() - _s0
            actions.append(int(act))
            sources.append(src)
            # No episode_over check here — it is done at the top of the next
            # iteration, after the resulting frame has been recorded.

        n = len(frames)
        stats = {
            "frames": n, "n_policy": n_policy, "n_tail": n_tail,
            "n_disagree": n_disagree,
            "policy_end": policy_end, "end_reason": end_reason,
            "start_geo": round(float(start_geo), 3),
            "final_geo": round(float(geo[-1]), 3),
            "policy_stop_geo": (None if policy_stop_geo is None
                                else round(float(policy_stop_geo), 3)),
            "t_reset": round(t_reset, 2), "t_gen": round(t_gen, 2),
            "t_step": round(t_step, 2), "t_expert": round(t_expert, 2),
            "n_gen": n_gen,
            "ms_per_step": (round(t_step / max(len(actions) - 1, 1) * 1000, 1)),
        }

        if len(actions) != n or len(sources) != n:
            stats["drop"] = f"misaligned actions={len(actions)} frames={n}"
            return None, stats
        if n > 498:
            # Same cap as gt_following_rollout: config/vln_r2r.yaml sets
            # max_episode_steps 500 and a truncated trajectory has no STOP.
            stats["drop"] = f"too_long({n})"
            return None, stats
        if end_reason != "expert_stop" and a.require_stop:
            stats["drop"] = f"no_stop({end_reason})"
            return None, stats

        record = {
            "rgbs": rgbs,
            "positions": np.stack(positions),
            "rotations": np.stack(rotations),
            "region_ids": np.array(region_ids),
            "region_categories": np.array(region_cats),
            "actions": np.asarray(actions, dtype=np.int8),
            "geodesic_to_goal": np.asarray(geo, dtype=np.float32),
            "goal_positions": np.asarray(geom["goals"], dtype=np.float32),
            "reference_path": np.asarray(geom["reference_path"], dtype=np.float32),
            "expert_chunks": np.asarray(chunks, dtype=np.int8),
            "action_sources": np.asarray(sources, dtype="<U8"),
        }
        return record, stats

    # -- the run loop -------------------------------------------------------
    def collect(self, rank=0, world_size=1, limit=0):
        os.makedirs(self.output_path, exist_ok=True)
        os.makedirs(os.path.join(self.output_path, "traj"), exist_ok=True)

        env = self.ev.config_env()

        by_scene = {}
        for ep in env.episodes:
            by_scene.setdefault(ep.scene_id, []).append(ep)

        shard_path = os.path.join(self.output_path, f"annotations_{rank}.json")
        annotations = json.load(open(shard_path)) if os.path.exists(shard_path) else []
        done_ids = {int(a["id"]) for a in annotations}
        # ALSO skip anything already in the merged annotations.json. --merge
        # deletes the shards by default, so without this a merge performed
        # between two runs (e.g. after a pilot) silently makes the next run
        # re-collect every merged episode. Sharding is deterministic
        # (by_scene[scene][rank::world_size]), so a merged episode this rank
        # owns was collected by this rank; the ones it does not own never reach
        # the loop below and cost nothing to carry here.
        merged_path = os.path.join(self.output_path, "annotations.json")
        n_merged = 0
        if os.path.exists(merged_path):
            merged_ids = {int(a["id"]) for a in json.load(open(merged_path))}
            n_merged = len(merged_ids - done_ids)
            done_ids |= merged_ids
        if done_ids:
            print(f"[rank {rank}] resuming: {len(done_ids)} episodes already done "
                  f"({n_merged} of them from the merged annotations.json)",
                  flush=True)
        summary_path = os.path.join(self.output_path, f"summary_{rank}.json")

        # Crash-loop guard. The launcher restarts a worker that dies, and the
        # resume skips only FINISHED episodes — so an episode that reliably
        # segfaults habitat-sim would be retried forever and the shard would
        # never advance past it. One line per attempt, written BEFORE the
        # episode runs (a crash leaves no other trace), and an id that burned
        # --max_attempts without finishing is skipped.
        attempts_path = os.path.join(self.output_path, f"attempts_{rank}.log")
        tried = {}
        if os.path.exists(attempts_path):
            with open(attempts_path) as f:
                for line in f:
                    line = line.strip()
                    if line.isdigit():
                        tried[int(line)] = tried.get(int(line), 0) + 1
        blocked = {i for i, c in tried.items()
                   if c >= self.args.max_attempts and i not in done_ids}
        if blocked:
            print(f"[rank {rank}] skipping {len(blocked)} episodes that failed "
                  f"{self.args.max_attempts}x: {sorted(blocked)[:20]}", flush=True)

        n_done = n_drop = 0
        step_ms = []
        for scene in sorted(by_scene.keys()):
            if limit and n_done >= limit:
                break
            scan = scene.split("/")[-2]
            for ep_desc in by_scene[scene][rank::world_size]:
                if limit and n_done >= limit:
                    break
                episode_id = int(ep_desc.episode_id)
                if episode_id in done_ids or episode_id in blocked:
                    continue

                with open(attempts_path, "a") as f:
                    f.write(f"{episode_id}\n")
                env.current_episode = ep_desc
                try:
                    _t0 = time.perf_counter()
                    record, stats = self.collect_episode(env, ep_desc)
                    wall = time.perf_counter() - _t0
                except Exception as e:
                    print(f"[rank {rank}] {scan}_{episode_id} FAILED: {e!r}",
                          flush=True)
                    n_drop += 1
                    continue

                stats["wall"] = round(wall, 1)
                if record is None:
                    n_drop += 1
                    print(f"[rank {rank}] {scan}_{episode_id} drop: "
                          f"{stats.get('drop')} {stats}", flush=True)
                    self._log(summary_path, {"id": episode_id, "scan": scan,
                                             "dropped": True, **stats})
                    continue

                basename = f"{scan}_{self.dataset}_{episode_id:06d}"
                self._write_episode(basename, record)

                instr = ep_desc.instruction.instruction_text
                ann = {
                    "id": episode_id,
                    "video": os.path.join("images", basename),
                    "pose_file": os.path.join("traj", basename + ".npz"),
                    "scan": scan,
                    "instructions": instr if isinstance(instr, list) else [instr],
                    "actions": [int(x) for x in record["actions"]],
                }
                annotations.append(ann)
                done_ids.add(episode_id)
                # Dumped every episode, not once per scene: at ~25 s/episode a
                # mid-scene crash would otherwise throw away hours of work.
                with open(shard_path, "w") as f:
                    json.dump(annotations, f)
                self._log(summary_path, {**ann, "dropped": False, **stats})

                n_done += 1
                step_ms.append(stats["ms_per_step"])
                print(f"[rank {rank}] {basename} frames={stats['frames']} "
                      f"policy={stats['n_policy']} tail={stats['n_tail']} "
                      f"disagree={stats['n_disagree']} "
                      f"end={stats['policy_end']}/{stats['end_reason']} "
                      f"geo {stats['start_geo']}->{stats['final_geo']} "
                      f"wall={stats['wall']}s "
                      f"(gen {stats['t_gen']}s step {stats['t_step']}s "
                      f"expert {stats['t_expert']}s)", flush=True)

                self._throughput_gate(rank, step_ms)

        env.close()
        print(f"[rank {rank}] done: {n_done} kept, {n_drop} dropped", flush=True)

    def _write_episode(self, basename, record):
        rgb_dir = os.path.join(self.output_path, "images", basename, "rgb")
        os.makedirs(rgb_dir, exist_ok=True)
        for c, rgb in enumerate(record["rgbs"]):
            Image.fromarray(rgb).convert("RGB").save(
                os.path.join(rgb_dir, f"{c + 1:03d}.jpg"))
        np.savez(
            os.path.join(self.output_path, "traj", basename + ".npz"),
            **{k: v for k, v in record.items() if k != "rgbs"},
        )

    @staticmethod
    def _log(path, obj):
        with open(path, "a") as f:
            f.write(json.dumps(obj, default=float) + "\n")

    def _throughput_gate(self, rank, step_ms):
        """Abort loudly if the sim landed on the model's card.

        Healthy on this box: env_step ~10 ms. Poisoned (CUDA compute preempting
        the GL render): 3700-5100 ms. There is no middle ground, so a single
        threshold is enough — and failing after 20 episodes beats discovering a
        400x slowdown 12 hours in.
        """
        if self.args.gate_ms <= 0 or len(step_ms) != self.args.gate_after:
            return
        med = statistics.median(step_ms)
        if med > self.args.gate_ms:
            raise SystemExit(
                f"[rank {rank}] THROUGHPUT GATE FAILED: median env_step "
                f"{med:.0f} ms > {self.args.gate_ms} ms over "
                f"{self.args.gate_after} episodes.\n"
                f"  Model and sim are almost certainly on the SAME physical "
                f"card (model_gpu={self.args.model_gpu} "
                f"sim_gpu={self.args.sim_gpu}); check that "
                f"CUDA_VISIBLE_DEVICES exposes all cards so these absolute "
                f"indices mean what they say. Healthy is ~10 ms."
            )
        print(f"[rank {rank}] throughput gate OK: median env_step {med:.0f} ms",
              flush=True)


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model_path', default='', help='Your completed Stage 1 full-model export; not a released final adapter')
    p.add_argument('--dataset', choices=('r2r_dagger', 'rxr_dagger'), default='r2r_dagger')
    p.add_argument('--habitat_config_path', default='', help='Default: config/vln_r2r_train.yaml or config/vln_rxr_train.yaml')
    p.add_argument('--data_path', default='', help='Official train/train.json.gz or train/train_guide.json.gz')
    p.add_argument('--scenes_dir', default='', help='Scene root containing mp3d/<scan>/<scan>.glb')
    p.add_argument('--split', choices=('train',), default='train')
    p.add_argument('--output_path', required=True)
    p.add_argument('--rank', type=int, default=0)
    p.add_argument('--world_size', type=int, default=1)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--merge', action='store_true')
    p.add_argument('--keep_shards', action='store_true')
    p.add_argument('--model_gpu', type=int, default=1)
    p.add_argument('--sim_gpu', type=int, default=0)
    p.add_argument('--chunk_size', type=int, default=8, help='Expert plans; at least num_future_steps actions per state')
    p.add_argument('--beta', type=float, default=0.0, help='Per-step expert execution probability; 0 rolls out the Stage 1 policy, 1 is an expert-only diagnostic')
    p.add_argument('--max_policy_steps', type=int, default=300)
    p.add_argument('--max_tail_steps', type=int, default=200)
    p.add_argument('--goal_stop_dist', type=float, default=0.5, help='Match the expert data goal-stop distance')
    p.add_argument('--max_geo_regress', type=float, default=5.0)
    p.add_argument('--tail_goal_only', action='store_true', help='Optional goal-only tail; default follows the described reference path')
    p.add_argument('--require_stop', action='store_true', default=True, help='Only keep trajectories ending at an expert STOP')
    p.add_argument('--max_attempts', type=int, default=3)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--gate_after', type=int, default=20)
    p.add_argument('--gate_ms', type=float, default=0.0, help='Optional simulator-step speed threshold; disabled by default')
    p.add_argument('--num_history', type=int, default=8)
    p.add_argument('--num_future_steps', type=int, default=4)
    p.add_argument('--num_exec', type=int, default=4)
    p.add_argument('--max_new_tokens', type=int, default=64)
    p.add_argument('--max_pixels', type=int, default=28 * 28 * 256)
    p.add_argument('--min_pixels', type=int, default=28 * 28 * 16)
    args = p.parse_args(argv)
    args.save_video, args.video_filter, args.max_episodes = False, 'all', 0
    if args.world_size < 1 or not 0 <= args.rank < args.world_size or args.limit < 0:
        p.error('Require world_size >= 1, 0 <= rank < world_size and limit >= 0')
    if not 0 <= args.beta <= 1 or args.max_attempts < 1:
        p.error('Require beta in [0,1] and max_attempts >= 1')
    if (args.num_history, args.num_future_steps, args.num_exec, args.seed) != (8, 4, 4, 42):
        p.error('Use history8, future4, exec4 and seed42 to match the released training recipe')
    if args.chunk_size < args.num_future_steps:
        p.error('chunk_size must be at least num_future_steps')
    if args.max_policy_steps < 1 or args.max_tail_steps < 1 or args.goal_stop_dist <= 0:
        p.error('Policy/tail budgets and goal_stop_dist must be positive')
    if not args.merge:
        if not args.data_path or not args.scenes_dir:
            p.error('--data_path and --scenes_dir are required for train-only collection')
        if args.beta < 1.0 and not args.model_path:
            p.error('--model_path is required unless using the beta=1 expert-only diagnostic')
        if not args.habitat_config_path:
            dataset = 'rxr' if args.dataset == 'rxr_dagger' else 'r2r'
            args.habitat_config_path = str(_HERE.parents[2] / 'config' / f'vln_{dataset}_train.yaml')
    return args


if __name__ == '__main__':
    _main()

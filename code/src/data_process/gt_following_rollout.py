import habitat
import json
import glob
import numpy as np
import argparse
import sys
import os
import torch

from omegaconf import OmegaConf
from PIL import Image

from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat_baselines.config.default import get_config as get_habitat_config
from habitat.config import read_write

DATASET = "rxr"
CONFIG_PATH = "./config/vln_r2r.yaml"
OUTPUT_PATH = f"./data/trajectory_data/{DATASET}"
DATA_PATH = None  # Set to None to use default dataset path


def region_of(semantic_scene, pos):
    """Return (region_id, region_category) for a world-space position.

    AABB containment (center +/- sizes/2); first matching region wins. Returns
    ("", "none") when the point is in no region's box, or when the scene has no
    semantic annotations (e.g. HM3D without the semantics add-on). Identical to
    src/replay.py so rollout and replay produce the same region fields.
    """
    p = np.asarray(pos, dtype=np.float64)
    for region in semantic_scene.regions:
        aabb = region.aabb
        c = np.asarray(aabb.center, dtype=np.float64)
        h = np.asarray(aabb.sizes, dtype=np.float64) / 2.0
        if np.all(p >= c - h) and np.all(p <= c + h):
            try:
                cat = region.category.name()
            except Exception:
                cat = "unknown"
            return region.id, cat
    return "", "none"


def merge_shards(output_path, keep_shards=False, include_existing=False):
    """Merge annotations_<rank>.json shards into a single annotations.json.

    Per-rank shards exist because workers write in parallel; this collapses
    them into one annotations.json (sorted by id, deduped). Mirrors the merge
    in src/replay.py so both pipelines produce a single annotation file.

    ``include_existing`` folds an already-present annotations.json into the
    union instead of overwriting it. Off by default so a re-merge still means
    "rebuild from the shards on disk" (the GT pipeline relies on that to drop
    episodes it has deleted). Turn it on for an incrementally-collected dataset
    where an earlier batch was already merged and its shards removed — see
    dagger_rollout.py.
    """
    shards = sorted(glob.glob(os.path.join(output_path, "annotations_*.json")))
    # guard against matching the merged file itself on re-runs
    shards = [s for s in shards
              if os.path.basename(s) != "annotations.json"]
    out_path = os.path.join(output_path, "annotations.json")
    merged = {}
    n_existing = 0
    if include_existing and os.path.exists(out_path):
        for rec in json.load(open(out_path)):
            merged[int(rec["id"])] = rec
        n_existing = len(merged)
    if not shards:
        raise SystemExit(f"no shards to merge in {output_path}")
    for s in shards:
        for rec in json.load(open(s)):
            merged[int(rec["id"])] = rec
    out = sorted(merged.values(), key=lambda r: int(r["id"]))
    with open(out_path, "w") as f:
        json.dump(out, f)
    extra = f" (+{n_existing} carried over from the previous merge)" if n_existing else ""
    print(f"merged {len(shards)} shards -> {out_path} ({len(out)} episodes){extra}")
    if not keep_shards:
        for s in shards:
            os.remove(s)
        print(f"removed {len(shards)} shard files")


def _resolve_sim_gpu(sim_gpu: int, rank: int) -> int:
    """Explicit --sim_gpu wins; otherwise spread workers over the visible cards."""
    if sim_gpu >= 0:
        return sim_gpu
    try:
        n = torch.cuda.device_count()
    except Exception:
        n = 0
    return rank % n if n > 0 else 0


class VLNHabitatRunner:
    def __init__(self, dataset: str, config_path: str, output_path: str, data_path: str = None,
                 split: str = None, follow_to_goal: bool = False,
                 goal_stop_dist: float = 0.5, sim_gpu: int = 0):
        self.device = torch.device("cuda")
        self.dataset = dataset.lower()
        self.config_path = config_path
        self.output_path = output_path
        self.data_path = data_path
        self.split = split
        self.follow_to_goal = bool(follow_to_goal)
        self.goal_stop_dist = float(goal_stop_dist)
        self.sim_gpu = int(sim_gpu)

        self.config = get_habitat_config(self.config_path)

    def config_env(self, scene: str = None) -> habitat.Env:
        with read_write(self.config):
            if self.split is not None:
                self.config.habitat.dataset.split = self.split
            if self.data_path is not None:
                self.config.habitat.dataset.update(
                    {
                        "data_path": self.data_path,
                    }
                )
            # Rollout only needs the ShortestPathFollower + agent transforms;
            # the configured measurements (success/spl/oracle_*) are unused here
            # and some aren't registered in this habitat-lab, so drop them.
            self.config.habitat.task.measurements = {}
            # Pin this worker's renderer to its own GPU. The habitat config hard-
            # codes gpu_device_id: 0, so without this every sharded worker would
            # open its GL context on card 0 and contend for memory there while the
            # rest of the box idles. Rollout is GL-only (no CUDA compute, no
            # model), so one worker per card is the natural mapping.
            self.config.habitat.simulator.habitat_sim_v0.gpu_device_id = self.sim_gpu
        print(OmegaConf.to_yaml(self.config))
        return habitat.Env(config=self.config)

    def generate(self, rank: int = 0, world_size: int = 1, limit: int = 0) -> None:
        os.makedirs(os.path.join(self.output_path), exist_ok=True)
        os.makedirs(os.path.join(self.output_path, "traj"), exist_ok=True)
        env = self.config_env()

        scene_episode_dict = {}
        for episode in env.episodes:
            if episode.scene_id not in scene_episode_dict:
                scene_episode_dict[episode.scene_id] = []
            scene_episode_dict[episode.scene_id].append(episode)

        # Resume: reload this rank's prior annotations so a restart (e.g. after
        # a habitat-sim scene-reload segfault) keeps finished episodes and the
        # merged annotations.json stays complete.
        shard_path = os.path.join(self.output_path, f"annotations_{rank}.json")
        annotations = json.load(open(shard_path)) if os.path.exists(shard_path) else []
        done_ids = {int(a["id"]) for a in annotations}
        if done_ids:
            print(f"[rank {rank}] resuming: {len(done_ids)} episodes already done")
        # Track ids already in the per-episode summary log so a resume after a
        # mid-scene crash does not append duplicate lines (annotations_{rank}.json
        # is dumped per-scene, so done_ids can miss episodes finished mid-scene).
        summary_path = os.path.join(self.output_path, f"summary_{rank}.json")
        summary_done = set()
        if os.path.exists(summary_path):
            with open(summary_path) as f:
                for line in f:
                    try:
                        summary_done.add(int(json.loads(line)["id"]))
                    except Exception:
                        pass
        n_done = 0
        for scene_id in sorted(scene_episode_dict.keys()):
            if limit and n_done >= limit:
                break
            scan = scene_id.split("/")[-2]
            episodes = scene_episode_dict[scene_id]
            print(f"scene_id: {scene_id}, scan: {scan}")

            for episode in episodes[rank::world_size]:
                if limit and n_done >= limit:
                    break                      # smoke-test cap
                if int(episode.episode_id) in done_ids:
                    continue  # already rolled out in a previous run
                env.current_episode = episode
                agent = ShortestPathFollower(
                    sim=env.sim, goal_radius=0.5, return_one_hot=False)

                instructions = episode.instruction.instruction_text
                trajectory_id = episode.trajectory_id
                scene_id = episode.scene_id.split('/')[-2]
                episode_id = int(episode.episode_id)
                ref_path = episode.reference_path
                if ref_path is None or len(ref_path) < 2:
                    print(f"[skip] episode {episode_id}: reference_path too short "
                          f"({0 if ref_path is None else len(ref_path)} waypoints)")
                    continue

                # The EVALUATED goal, which is NOT where this rollout ends. The
                # follower is chained between reference_path waypoints and stops at
                # the last one, but on R2R that final waypoint sits a median 1.66 m
                # from goals[0] — so ~15% of expert trajectories terminate already
                # outside the 3 m success radius. Anything that needs "am I at the
                # goal" (the arrive/stop labels) must use THIS, not positions[-1].
                goal_positions = [
                    np.asarray(g.position, dtype=np.float32) for g in episode.goals
                ]

                observation = env.reset()
                semantic_scene = env.sim.semantic_scene

                # episode initialization
                rgb_list = []
                depth_list = []
                positions = []
                rotations = []
                region_ids = []
                region_cats = []
                geo_dists = []
                actions = [-1]
                next_waypoint_id = 1

                rgb_dir = os.path.join(
                    self.output_path, "images", f"{scene_id}_{self.dataset}_{episode_id:06d}", "rgb")
                os.makedirs(rgb_dir, exist_ok=True)
                while not env.episode_over:
                    rgb = observation["rgb"]
                    rgb_list.append(rgb)
                    Image.fromarray(rgb).convert("RGB").save(
                        os.path.join(rgb_dir, f"{len(rgb_list):03d}.jpg"))

                    # Pose + region for THIS frame, captured at the same agent
                    # state that produced the rgb just saved (world frame, Y-up).
                    st = env.sim.get_agent_state()
                    positions.append(np.asarray(st.position, dtype=np.float32))
                    q = st.rotation
                    rotations.append(
                        np.array([q.w, q.x, q.y, q.z], dtype=np.float32))
                    rid, rcat = region_of(semantic_scene, st.position)
                    region_ids.append(rid)
                    region_cats.append(rcat)
                    # GEODESIC distance from this frame to the nearest goal —
                    # the same quantity habitat's distance_to_goal measure (and
                    # therefore success/SPL) is computed from. Euclidean distance
                    # is a poor proxy here: it ignores walls, so a pose 1 m from
                    # the goal through a wall would look arrived. Recorded per
                    # frame so the offline label builder never has to guess.
                    # inf (unreachable / off-navmesh) is stored as-is; consumers
                    # must treat non-finite as "no label".
                    geo_dists.append(
                        float(env.sim.geodesic_distance(st.position, goal_positions))
                    )

                    next_action = agent.get_next_action(
                        ref_path[next_waypoint_id])

                    force_episode_over = False
                    while next_action == 0:
                        next_waypoint_id += 1
                        if next_waypoint_id == len(ref_path) - 1:
                            agent = ShortestPathFollower(
                                sim=env.sim, goal_radius=0.25, return_one_hot=False)
                        if next_waypoint_id >= len(ref_path):
                            force_episode_over = True
                            break
                        next_action = agent.get_next_action(
                            ref_path[next_waypoint_id])

                    if force_episode_over:
                        break

                    observation = env.step(next_action)
                    actions.append(next_action)

                # Optionally keep walking to the EVALUATED goal after the
                # reference path is exhausted. R2R's last reference waypoint sits a
                # median 1.66 m from goals[0], which leaves ~15% of expert
                # trajectories terminating outside the 3 m success radius — i.e.
                # teaching STOP at a pose the benchmark scores as a failure. With
                # --follow_to_goal the trajectory ends where success is actually
                # measured, so the arrive labels and the metric agree.
                if self.follow_to_goal and not env.episode_over:
                    goal_agent = ShortestPathFollower(
                        sim=env.sim, goal_radius=0.25, return_one_hot=False)
                    guard = 0
                    while not env.episode_over and guard < 200:
                        st = env.sim.get_agent_state()
                        if float(env.sim.geodesic_distance(
                                st.position, goal_positions)) <= self.goal_stop_dist:
                            break
                        act = goal_agent.get_next_action(goal_positions[0])
                        if act == 0 or act is None:
                            break
                        rgb = observation["rgb"]
                        rgb_list.append(rgb)
                        Image.fromarray(rgb).convert("RGB").save(
                            os.path.join(rgb_dir, f"{len(rgb_list):03d}.jpg"))
                        positions.append(np.asarray(st.position, dtype=np.float32))
                        q = st.rotation
                        rotations.append(
                            np.array([q.w, q.x, q.y, q.z], dtype=np.float32))
                        rid, rcat = region_of(semantic_scene, st.position)
                        region_ids.append(rid)
                        region_cats.append(rcat)
                        geo_dists.append(float(env.sim.geodesic_distance(
                            st.position, goal_positions)))
                        observation = env.step(act)
                        actions.append(act)
                        guard += 1

                if len(actions) > 498:
                    continue  # Skip episodes with too many actions

                if len(actions) != len(rgb_list) or len(positions) != len(actions):
                    print(f"[skip] episode {episode_id}: misaligned "
                          f"actions={len(actions)} rgb={len(rgb_list)} "
                          f"pose={len(positions)}")
                    continue

                basename = f"{scene_id}_{self.dataset}_{episode_id:06d}"
                pose_file = os.path.join("traj", basename + ".npz")
                np.savez(
                    os.path.join(self.output_path, pose_file),
                    positions=np.stack(positions),
                    rotations=np.stack(rotations),
                    region_ids=np.array(region_ids),
                    region_categories=np.array(region_cats),
                    actions=np.asarray(actions, dtype=np.int8),
                    # per-frame geodesic distance to the nearest goal (metres)
                    geodesic_to_goal=np.asarray(geo_dists, dtype=np.float32),
                    # the evaluated goal(s) and the reference path, so the offline
                    # label builder does not have to re-open the habitat dataset
                    goal_positions=np.stack(goal_positions),
                    reference_path=np.asarray(ref_path, dtype=np.float32),
                )

                annotations.append({
                    "id": episode_id,
                    "video": os.path.join("images", basename),
                    "pose_file": pose_file,
                    "scan": scene_id,
                    "instructions": instructions if isinstance(instructions, list) else [instructions],
                    "actions": actions,
                })

                n_done += 1
                if episode_id not in summary_done:
                    with open(summary_path, "a") as f:
                        result = {
                            "id": episode_id,
                            "video": os.path.join("images", basename),
                            "pose_file": pose_file,
                            "scan": scene_id,
                            "instructions": instructions if isinstance(instructions, list) else [instructions],
                            "actions": actions,
                            "trajectory_id": trajectory_id,
                            "scene_id": scene_id,
                        }
                        f.write(json.dumps(result) + "\n")
                    summary_done.add(episode_id)

            with open(os.path.join(self.output_path, f"annotations_{rank}.json"), "w") as f:
                json.dump(annotations, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=DATASET)
    parser.add_argument("--config_path", type=str, default=CONFIG_PATH)
    parser.add_argument("--output_path", type=str, default=OUTPUT_PATH)
    parser.add_argument("--data_path", type=str, default=DATA_PATH)
    parser.add_argument("--split", type=str, default=None,
                        help="dataset split; fills {split} in data_path")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--merge", action="store_true",
                        help="merge annotations_<rank>.json shards into a "
                             "single annotations.json, then exit")
    parser.add_argument("--sim_gpu", type=int, default=-1,
                        help="GPU for this worker's habitat renderer. -1 (default) = "
                             "rank %% (visible GPU count), so sharded workers spread "
                             "across cards instead of all landing on card 0.")
    parser.add_argument("--limit", type=int, default=0,
                        help="stop after this many episodes per rank (0 = whole split); smoke tests")
    parser.add_argument("--follow_to_goal", action="store_true",
                        help="after the reference path is exhausted, keep following the "
                             "shortest path to goals[0] until within --goal_stop_dist. "
                             "R2R's last reference waypoint is a median 1.66 m from the "
                             "goal, leaving ~15%% of expert trajectories ending outside "
                             "the 3 m success radius; this makes the trajectory end where "
                             "success is actually measured.")
    parser.add_argument("--goal_stop_dist", type=float, default=0.5,
                        help="with --follow_to_goal, stop this many metres (geodesic) from the goal")
    parser.add_argument("--keep_shards", action="store_true",
                        help="with --merge, keep the per-rank shard files")
    args = parser.parse_args()

    if args.merge:
        merge_shards(args.output_path, keep_shards=args.keep_shards)
        sys.exit(0)

    runner = VLNHabitatRunner(
        dataset=args.dataset,
        config_path=args.config_path,
        output_path=args.output_path,
        data_path=args.data_path,
        split=args.split,
        follow_to_goal=args.follow_to_goal,
        goal_stop_dist=args.goal_stop_dist,
        sim_gpu=_resolve_sim_gpu(args.sim_gpu, args.rank),
    )
    runner.generate(rank=args.rank, world_size=args.world_size, limit=args.limit)

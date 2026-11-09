#!/usr/bin/env python3
"""build_vln_samples.py — offline per-step sample expansion for VLN training.

Turns each VLN episode (``annotations.json`` + ``traj/*.npz`` poses) into
**sample-level** records, one per decision timestep ``c`` (per-step / stride 1),
so the discrete-action and continuous-waypoint trainers can read the SAME
records and differ ONLY in which label they consume. This removes the
sampling-vs-head confound in the A/B and moves all index math offline.

Each record carries BOTH supervision targets:
  * ``action_text`` / ``action_ids`` — the next ``num_future_steps`` discrete
    actions (arrow glyphs, terminal STOP synthesised), for the discrete head.
    On a DAgger directory pass ``--action_source expert_chunks`` instead: there
    the executed actions are the *policy's*, and the label must be the per-frame
    expert plan the collector stored alongside them.
  * ``waypoints`` — ``predict_size`` future waypoints in frame ``c``'s local
    frame ``[x_fwd, y_lat, sin(yaw), cos(yaw)]``, for the continuous head. Near
    the end of an episode the chunk is TRUNCATED to the real remaining future and
    hold-last padded; ``wp_n_valid`` carries the unpadded length so the loss can
    mask per step. Only the terminal frame itself has ``wp_n_valid == 0``.
  * ``arrive_list`` / ``dist_goal`` / ``dist_future`` — one arrive flag PER
    waypoint step: flag ``k`` is 1 when the pose at future step ``k`` is within
    ``--stop_radius`` metres of the evaluated goal. The distance is the rollout's
    stored per-frame GEODESIC distance (``geodesic_to_goal`` in the npz), i.e. the
    same quantity habitat's success measure uses. ``dist_goal`` / ``dist_future``
    keep the raw metres so a softer label can be derived without a rebuild.
    ``stop`` is the derived ``all(arrive_list)``, which is what the rollout's stop
    rule tests.

This replaces an earlier scheme that gated whole frames on
``c <= (n_use-1) - predict_size`` and labelled ``stop = (c == n_use-1)``. Since
the dataset keeps a waypoint-mode row only if it has waypoints or is a stop row,
that gate silently deleted the last ``predict_size - 1`` frames of every episode
— the approach zone — leaving the stop head with one positive per episode and no
negatives anywhere near it. See ``padded_future_waypoints`` for the details.

Prefers trajectories rolled out WITH geodesic distances (and normally
``--follow_to_goal``); ``gt_following_rollout.py`` writes them. Episodes whose
npz lacks ``geodesic_to_goal`` raise by default; ``--allow_euclidean_fallback``
labels them from poses alone instead, using euclidean distance to
``positions[-1]``. That proxy is wall-blind AND aimed at the trajectory endpoint
rather than the evaluated goal, so it is opt-in and every record carries
``dist_source`` (``"geodesic"`` / ``"euclidean_endpoint"``). Endpoint distances
are systematically smaller than geodesic-to-goal ones, so a radius calibrated on
geodesic data does not transfer — re-run ``diagnose_stop_radius.py`` per dataset.

Sampling spec (identical for both heads):
  * per-step: one record per (episode, instruction, c), c in [0, n_frames-1].
  * current observation: 1 frame = frame ``c`` = ``rgb/{c+1:03d}.jpg``.
  * history: frames uniformly in ``[0, c)`` via ``linspace(0, c, memory_size)``,
    de-duplicated (variable length when ``c < memory_size``), current frame
    excluded so history and current are disjoint. Empty at ``c == 0``.

Index convention (verified on R2R; n_actions == n_positions == n_rgb):
  ``actions = [-1, a0, ..., a_{K-1}]`` (index 0 is the reset placeholder).
  Real action ``a_m`` = ``actions[m+1]``; its pre-action observation is frame
  ``m`` = ``rgb/{m+1:03d}.jpg`` = ``positions[m]``. So a decision at frame ``c``
  predicts actions starting at ``a_c`` and its current view is frame ``c``.

Sharding: ``--rank/--world_size`` split episodes; each writes
``samples_perstep_<rank>.jsonl``; ``--merge`` concatenates into
``samples_perstep.jsonl`` (mirrors the rollout scripts' pattern).
"""

import argparse
import glob
import json
import os
import random
import sys
from typing import List, Optional

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

# The single source of truth for <memory> frame selection, shared with both
# evaluators — see the module docstring for why this must not be re-implemented.
from qwenvl.data.vln_sampling import history_indices  # noqa: E402


# --- action glyphs (arrow-glyph mapping replayed by VLNUnifiedDataset) ------
IDX2ACTION = {0: "STOP", 1: "↑", 2: "←", 3: "→"}

# --- prompt template (baked into each sample; replayed by VLNUnifiedDataset) -
SYSTEM_PROMPT = (
    "You are an autonomous navigation assistant. Your task is to "
    "<instruction>. Devise an action sequence to follow the instruction "
    "using the four actions: TURN LEFT (←) or TURN RIGHT (→) by 15 "
    "degrees, MOVE FORWARD (↑) by 25 centimeters, or STOP."
)
CONJUNCTIONS = [
    "you can see ",
    "in front of you is ",
    "there is ",
    "you can spot ",
    "you are toward the ",
    "ahead of you is ",
    "in your sight is ",
]
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_MEMORY_TOKEN = "<memory>"


# --- waypoint geometry (identical to vln_unified_dataset) -------------------
def _quat_to_matrix(quat, quat_format="wxyz"):
    quat = np.asarray(quat, dtype=np.float32)
    if quat_format == "wxyz":
        w, x, y, z = quat
    elif quat_format == "xyzw":
        x, y, z, w = quat
    else:
        raise ValueError(f"Unsupported quat_format: {quat_format}")
    norm = np.sqrt(w * w + x * x + y * y + z * z)
    if norm < 1e-8:
        return np.eye(3, dtype=np.float32)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def compute_future_waypoints(positions, rotations, current_index, future_indices,
                             quat_format="wxyz"):
    """Future poses in the current pose's local frame:
    ``[x_forward, y_lateral, sin(yaw), cos(yaw)]`` per future index."""
    current_pos = positions[current_index]
    current_rot = _quat_to_matrix(rotations[current_index], quat_format)
    current_rot_inv = current_rot.T
    waypoints = []
    for future_index in future_indices:
        future_index = int(future_index)
        delta_world = positions[future_index] - current_pos
        delta_local = current_rot_inv @ delta_world
        future_rot = _quat_to_matrix(rotations[future_index], quat_format)
        future_forward_world = future_rot @ np.array([0.0, 0.0, -1.0], dtype=np.float32)
        future_forward_local = current_rot_inv @ future_forward_world
        yaw_local = np.arctan2(future_forward_local[0], -future_forward_local[2])
        waypoints.append(
            [-delta_local[2], delta_local[0], np.sin(yaw_local), np.cos(yaw_local)]
        )
    return np.asarray(waypoints, dtype=np.float32)


# --- sampling helpers -------------------------------------------------------
# ``history_indices`` now lives in qwenvl.data.vln_sampling (imported above) so
# the evaluators consume the exact same function instead of re-deriving it.


def padded_future_waypoints(positions, rotations, c, predict_size, n_use,
                            quat_format="wxyz"):
    """Waypoint chunk for frame ``c``, truncated to the real remaining future and
    hold-last padded to ``predict_size``.

    Returns ``(waypoints[predict_size, 4], n_valid, future_indices[predict_size])``.

    The build used to gate whole frames on ``c <= (n_use-1) - predict_size`` and
    emit ``null`` otherwise, which dropped the last ``predict_size - 1`` frames of
    every episode from training entirely (the dataset keeps a waypoint-mode row
    only if it has waypoints or is a stop row). Those are exactly the approach-zone
    frames, and their absence left the stop head with no supervision between
    "far away" negatives and the single terminal positive.

    Padding repeats the LAST REAL FUTURE INDEX, which is load-bearing twice over:
      * in the delta-action parameterization the padded steps become zero motion;
      * the arrive flags are keyed on each step's distance to the goal, and a
        padded step repeats the last real future index, so it inherits the
        trajectory endpoint's distance — inside stop_radius when the rollout used
        --follow_to_goal, hence legitimately flagged 1. Padding with zeros instead
        would put the padded waypoint at the agent's own position, contradicting
        the waypoints one step before the goal.
    ``n_valid`` travels with the record so the regression loss masks per step —
    the padded waypoint VALUE is never itself a regression target.
    """
    last = n_use - 1
    n_valid = max(0, min(predict_size, last - c))
    if n_valid == 0:
        # Terminal frame: nothing left to regress. Every future index is the
        # endpoint itself, so all arrive flags are 1 and the row is kept for the
        # stop head alone.
        fut = [last] * predict_size
        return np.zeros((predict_size, 4), dtype=np.float32), 0, fut
    fut = list(range(c + 1, c + 1 + n_valid))
    fut += [fut[-1]] * (predict_size - n_valid)      # hold-last padding
    wp = compute_future_waypoints(positions, rotations, c, fut, quat_format)
    return wp, n_valid, fut


def actions_to_text(action_ids: List[int]) -> str:
    return "".join(IDX2ACTION[int(a)] for a in action_ids)


def build_conversation(instruction: str, has_memory: bool, conj: str,
                       action_text: str) -> List[dict]:
    """Single-turn discrete conversation matching VLNUnifiedDataset action format:
    instruction + optional <memory> + one <image>, answer = arrow glyphs."""
    human = SYSTEM_PROMPT.replace("<instruction>.", instruction)
    if has_memory:
        human += f" These are your historical observations: {DEFAULT_MEMORY_TOKEN}."
    human += f" {conj}{DEFAULT_IMAGE_TOKEN}."
    return [
        {"from": "human", "value": human},
        {"from": "gpt", "value": action_text},
    ]


def expand_episode(rec: dict, dataset_dir: str, memory_size: int,
                   num_future_steps: int, predict_size: int, quat_format: str,
                   rng: random.Random, stop_radius: float = 3.0,
                   allow_euclidean_fallback: bool = False,
                   action_source: str = "actions") -> List[dict]:
    """Yield per-step sample records for one episode record."""
    actions = rec["actions"]                       # [-1, a0, ..., a_{K-1}]
    n_frames = len(actions)                        # == len(positions) == len(rgb)
    if n_frames < 2:
        return []

    video_rel = rec["video"]                       # e.g. images/<scan>_r2r_000001
    rgb_dir = os.path.join(dataset_dir, video_rel, "rgb")
    frame_files = sorted(os.listdir(rgb_dir))
    if len(frame_files) != n_frames:
        # Alignment guard: the +1 offset assumes one rgb file per frame.
        n_use = min(len(frame_files), n_frames)
    else:
        n_use = n_frames

    # local action array: strip reset placeholder, append synthetic terminal STOP
    actions_local = list(actions[1:]) + [0]        # index m -> a_m ; last -> STOP

    # poses. REQUIRED now (they were loaded lazily when waypoints were gated on a
    # full horizon): waypoints are truncated+padded per frame, and both the
    # arrive flags and dist_goal are geometric. An episode without an npz cannot
    # be labelled at all, so it is skipped rather than silently emitted with a
    # degraded label.
    pose_file = rec.get("pose_file")
    if not pose_file:
        # ScaleVLN annotations lack a pose_file field; the npz is named after
        # the video basename under traj/.
        pose_file = os.path.join("traj", os.path.basename(video_rel) + ".npz")
    pose_path = os.path.join(dataset_dir, pose_file)
    if not os.path.exists(pose_path):
        return []
    try:
        data = np.load(pose_path, allow_pickle=False)
        positions = data["positions"].astype(np.float32)
        rotations = data["rotations"].astype(np.float32)
    except Exception:
        return []
    if len(positions) < n_use or len(rotations) < n_use:
        return []

    # DAgger: the discrete label comes from the per-frame EXPERT plan, not from
    # what was executed. `actions` on a DAgger trajectory is the policy's own
    # (frequently wrong) action, so falling back to it here would be
    # self-imitation of mistakes — hence the hard failure rather than a default.
    expert_chunks = None
    if action_source == "expert_chunks":
        if "expert_chunks" not in data.files:
            raise SystemExit(
                f"--action_source expert_chunks but {pose_path} has no "
                f"'expert_chunks' array (fields: {list(data.files)}). This is a "
                f"gt_following_rollout directory, not a dagger_rollout one."
            )
        expert_chunks = data["expert_chunks"].astype(np.int8)
        if len(expert_chunks) < n_use:
            return []

    # Distance-to-goal, the quantity the arrive flags are keyed on. Preferred
    # source is the rollout's stored per-frame GEODESIC distance to the evaluated
    # goal — exactly what habitat's success measure uses.
    #
    # Legacy trajectories (rolled out before that field existed: R2R, RxR,
    # EnvDrop, ScaleVLN) fall back to EUCLIDEAN distance to ``positions[-1]``,
    # enabled by --allow_euclidean_fallback. Both degradations are real and the
    # labels are measurably worse, so the flag is opt-in and every record carries
    # ``dist_source`` so downstream code and audits can tell the two apart:
    #
    #   * euclidean ignores walls — a pose 1 m from the goal through a wall reads
    #     as arrived, so some arrive=1 flags mark poses the agent cannot stop at.
    #   * ``positions[-1]`` is not the evaluated goal. The follower stops at the
    #     last reference_path waypoint, which on R2R sits a median 1.66 m from
    #     goals[0], leaving ~15% of expert trajectories ending outside the 3 m
    #     success radius. Labelling against it teaches "stop where the expert
    #     stopped" rather than "stop at the goal".
    #
    # The second point matters for radius choice: distances measured to the
    # endpoint are systematically SMALLER than to the true goal (the endpoint is
    # on the path), so a radius calibrated on geodesic-to-goal data is not
    # transferable. Run diagnose_stop_radius.py per dataset instead of reusing
    # 1.75.
    if "geodesic_to_goal" in data.files:
        dist_to_goal = data["geodesic_to_goal"].astype(np.float64)[:n_use]
        dist_source = "geodesic"
    elif allow_euclidean_fallback:
        goal_xyz = positions[n_use - 1].astype(np.float64)
        dist_to_goal = np.linalg.norm(
            positions[:n_use].astype(np.float64) - goal_xyz[None, :], axis=1
        )
        dist_source = "euclidean_endpoint"
    else:
        raise ValueError(
            f"{pose_path} has no 'geodesic_to_goal' — it predates the geodesic "
            "rollout. Re-run data_process/gt_following_rollout.py (with "
            "--follow_to_goal) for this dataset, or pass "
            "--allow_euclidean_fallback to label it from poses alone (wall-blind, "
            "and aimed at the trajectory endpoint rather than the goal)."
        )
    if len(dist_to_goal) < n_use or not np.isfinite(dist_to_goal).all():
        # Off-navmesh or unreachable goal: no trustworthy arrive label anywhere in
        # this episode, so drop it rather than emit a silently-wrong one.
        return []

    instructions = rec["instructions"]
    if not isinstance(instructions, list):
        instructions = [instructions]

    out = []
    for ins_id, instruction in enumerate(instructions):
        for c in range(0, n_use):
            hist = history_indices(c, memory_size)
            if expert_chunks is None:
                # discrete label: next num_future_steps actions from c (auto-clip)
                act_ids = [int(a) for a in actions_local[c : c + num_future_steps]]
            else:
                # One coherent expert plan from frame c, 0-padded by the collector.
                # Truncate AT the first STOP (inclusive): the trailing zeros are
                # padding, not repeated STOPs. This reproduces the `actions` path's
                # convention exactly — there, the slice can only ever pick up the
                # single synthetic terminal 0.
                ch = [int(a) for a in expert_chunks[c][:num_future_steps]]
                act_ids = ch[: ch.index(0) + 1] if 0 in ch else ch
            if not act_ids:
                continue
            action_text = actions_to_text(act_ids)

            # waypoint label: truncated to the real remaining future + hold-last
            # padded, with n_valid travelling alongside so the loss masks per step.
            wp, wp_n_valid, fut = padded_future_waypoints(
                positions, rotations, c, predict_size, n_use, quat_format
            )
            waypoints = wp.tolist() if wp_n_valid > 0 else None
            dist_goal = float(dist_to_goal[c])
            # One arrive flag PER WAYPOINT STEP (OmniNav's arrive_list): flag k
            # asks "will the agent be within stop_radius of the goal at future
            # step k". Padded steps repeat the last real future index, so they
            # inherit its distance — at the trajectory end that is the endpoint's
            # own distance, making the flags legitimately 1 and producing the
            # all-positive pattern the rollout's stop rule keys on.
            dist_future = [float(dist_to_goal[j]) for j in fut]
            arrive_list = [int(d <= stop_radius) for d in dist_future]

            # Whether the rollout's stop rule ("every arrive flag positive")
            # would fire here. Derived, not independent — kept in the record for
            # cheap statistics and for the dataset's keep/drop filter.
            stop = int(all(arrive_list))

            images = [f"{video_rel}/rgb/{frame_files[j]}" for j in hist]
            images.append(f"{video_rel}/rgb/{frame_files[c]}")   # current last
            conj = rng.choice(CONJUNCTIONS)
            conv = build_conversation(instruction, bool(hist), conj, action_text)

            out.append({
                "dataset": os.path.basename(dataset_dir.rstrip("/")),
                "video": video_rel,
                "ep_id": rec.get("id"),
                "ins_id": ins_id,
                "c": c,
                "n_frames": n_use,
                "instruction": instruction,
                "images": images,          # [memory..., current]
                "n_memory": len(hist),
                "action_ids": act_ids,
                "action_text": action_text,
                "action_source": action_source,  # "actions" | "expert_chunks" (DAgger)
                "waypoints": waypoints,      # [predict_size,4] hold-last padded, or null
                "wp_n_valid": wp_n_valid,    # real (unpadded) steps; 0 = terminal frame
                "arrive_list": arrive_list,  # [predict_size] per-step arrive flag
                "dist_goal": dist_goal,      # metres from frame c to the goal
                "dist_future": dist_future,  # same, per waypoint step (for soft labels)
                "dist_source": dist_source,  # "geodesic" | "euclidean_endpoint" (degraded)
                "stop": stop,                # == all(arrive_list); derived, for stats
                "conversations": conv,
            })
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset_dir", required=True,
                    help="e.g. stage2_datasets/instruction_following/R2R")
    ap.add_argument("--out_name", default="samples_perstep",
                    help="base name for the output jsonl (per-rank suffix added)")
    ap.add_argument("--memory_size", type=int, default=8)
    ap.add_argument("--num_future_steps", type=int, default=4)
    ap.add_argument("--predict_size", type=int, default=8)
    ap.add_argument("--stop_radius", type=float, default=1.25,
                    help="a waypoint step's arrive flag is 1 when its GEODESIC distance "
                         "to the goal is within this many metres. Chosen from the real "
                         "distribution via data_process/diagnose_stop_radius.py, not from "
                         "the 3.0 m success radius: labelling everything inside 3.0 m as "
                         "'stop now' teaches the agent to halt at the edge of the success "
                         "region and gives away SPL.")
    ap.add_argument("--allow_euclidean_fallback", action="store_true",
                    help="for legacy trajectories with no 'geodesic_to_goal' in the npz "
                         "(R2R, RxR, EnvDrop, ScaleVLN), label from poses alone: EUCLIDEAN "
                         "distance to positions[-1]. Degraded twice over — wall-blind (a "
                         "pose 1 m from the goal through a wall reads as arrived) and aimed "
                         "at the trajectory endpoint, which on R2R sits a median 1.66 m from "
                         "the evaluated goal. Endpoint distances run SMALLER than "
                         "geodesic-to-goal ones, so re-run diagnose_stop_radius.py per "
                         "dataset; do not reuse a radius calibrated on geodesic data. Each "
                         "record carries dist_source so the two can be told apart later.")
    ap.add_argument("--quat_format", default="wxyz", choices=["wxyz", "xyzw"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world_size", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0,
                    help="debug: only process first N episodes (0 = all)")
    ap.add_argument("--action_source", default="actions",
                    choices=["actions", "expert_chunks"],
                    help="where the discrete label comes from. 'actions' (default) "
                         "slices the executed action array — correct for expert "
                         "rollouts. 'expert_chunks' reads the per-frame expert plan "
                         "written by dagger_rollout.py, which is the only correct "
                         "choice there: on a DAgger trajectory the executed action "
                         "is the POLICY's, so slicing it would train the model to "
                         "repeat its own mistakes.")
    ap.add_argument("--merge", action="store_true",
                    help="merge all per-rank shards into <out_name>.jsonl and exit")
    args = ap.parse_args()

    out_dir = args.dataset_dir
    merged = os.path.join(out_dir, f"{args.out_name}.jsonl")
    if args.merge:
        shards = sorted(glob.glob(os.path.join(out_dir, f"{args.out_name}_*.jsonl")))
        n = 0
        with open(merged, "w") as fout:
            for sh in shards:
                with open(sh) as fin:
                    for line in fin:
                        fout.write(line)
                        n += 1
        print(f"[merge] {len(shards)} shards -> {merged}  ({n} samples)")
        return

    anno = json.load(open(os.path.join(out_dir, "annotations.json")))
    if args.limit:
        anno = anno[: args.limit]
    # shard by episode index
    my = [(i, r) for i, r in enumerate(anno) if i % args.world_size == args.rank]
    rng = random.Random(args.seed + args.rank)

    out_path = os.path.join(out_dir, f"{args.out_name}_{args.rank}.jsonl")
    n_samples = n_wp = n_eps = n_stop = n_trunc = 0
    with open(out_path, "w") as fout:
        for _, rec in my:
            recs = expand_episode(
                rec, out_dir, args.memory_size, args.num_future_steps,
                args.predict_size, args.quat_format, rng, args.stop_radius,
                args.allow_euclidean_fallback, args.action_source,
            )
            for s in recs:
                fout.write(json.dumps(s, ensure_ascii=False) + "\n")
                n_samples += 1
                if s["waypoints"] is not None:
                    n_wp += 1
                    if s["wp_n_valid"] < args.predict_size:
                        n_trunc += 1     # approach-zone rows the old gate dropped
                n_stop += int(s["stop"])
            n_eps += 1
    print(f"[rank {args.rank}/{args.world_size}] {n_eps} episodes -> "
          f"{n_samples} samples ({n_wp} with waypoints, of which {n_trunc} "
          f"truncated/approach-zone; {n_stop} stop-positive "
          f"[{100.0 * n_stop / max(n_samples, 1):.1f}%]) -> {out_path}")


if __name__ == "__main__":
    main()

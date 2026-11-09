"""Subprocess host for a habitat ``Env`` used by ``RemoteEnv``.

This module deliberately imports **no torch / CUDA** — it only ever touches
OpenGL/EGL (via habitat-sim). Running the simulator here, in a spawned
subprocess pinned to a *different physical GPU* than the model, is the only
configuration that keeps ``env.step`` fast on this box: when habitat's GL
rendering shares a GPU with the model's CUDA compute (even in a separate
process, or same process on another GPU), ``env.step`` balloons from ~4 ms to
~5 s because CUDA compute preempts the GL graphics engine. Separate process
AND separate GPU → ~4 ms (measured).

Protocol (parent <-> child over two Queues):
  parent sends: ("reset", ep_idx) | ("step", action) | ("metrics",)
                | ("oracle", predict_size, spacing) | ("close",)
                | ("expert_chunk", K, goal_only) | ("frame_state",)
                | ("episode_geom",) | ("agent_state",)
                | ("follow", dx, dy, goal_radius, max_actions)
  child replies: on start -> list of (idx, scene_id, episode_id, instruction)
                 reset/step -> (rgb_uint8_ndarray, episode_over_bool)
                 metrics      -> scalar metrics requested by the task config
                 oracle       -> local-frame waypoints ndarray [predict_size,4] or None
                 expert_chunk -> list[int] of length K (the expert's plan; 0-padded)
                 frame_state  -> {position, rotation(wxyz), geodesic, region_id,
                                  region_category}
                 episode_geom -> {goals, reference_path, start}
                 agent_state  -> {position, rotation(wxyz)}
                 follow       -> (actions, info)
                 unknown op   -> ("__unknown_op__", op)   [never silently hang]
"""

import os

import numpy as np


# ---------------------------------------------------------------------------
# Oracle-waypoint geometry (numpy only; used only by the continuous eval's
# --oracle validation mode). Mirrors vln_unified_dataset.compute_future_waypoints
# so the controller sees waypoints in the exact training convention:
#   [x_forward = -delta_local_z, y_lateral = delta_local_x, sin(yaw), cos(yaw)]
# ---------------------------------------------------------------------------
def _quat_to_matrix(quat, quat_format="wxyz"):
    quat = np.asarray(quat, dtype=np.float32)
    if quat_format == "wxyz":
        w, x, y, z = quat
    else:  # xyzw
        x, y, z, w = quat
    n = np.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-8:
        return np.eye(3, dtype=np.float32)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def _resample_polyline(pts, n, spacing):
    """Walk a world-space polyline, emitting ``n`` (position, forward_dir) pairs at
    arc-lengths spacing, 2*spacing, ... Past the end, clamp to the last point with
    the last direction."""
    if len(pts) < 2:
        d = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        p = pts[-1] if pts else np.zeros(3, dtype=np.float32)
        return [(p.copy(), d) for _ in range(n)]
    out = []
    seg = 0
    pos = pts[0].copy()
    v = pts[1] - pts[0]
    L = float(np.linalg.norm(v))
    cur_dir = (v / L) if L > 1e-6 else np.array([0.0, 0.0, -1.0], dtype=np.float32)
    rem = L
    for _ in range(n):
        need = spacing
        while need > 0 and seg < len(pts) - 1:
            if rem >= need:
                pos = pos + cur_dir * need
                rem -= need
                need = 0.0
            else:
                need -= rem
                seg += 1
                pos = pts[seg].copy()
                if seg < len(pts) - 1:
                    v = pts[seg + 1] - pts[seg]
                    L = float(np.linalg.norm(v))
                    cur_dir = (v / L) if L > 1e-6 else cur_dir
                    rem = L
                else:
                    rem = 0.0
        out.append((pos.copy(), cur_dir.copy()))
    return out


def _oracle_local_waypoints(env, predict_size, spacing, quat_format="wxyz"):
    import habitat_sim

    sim = env.sim
    st = sim.get_agent_state()
    cur_pos = np.asarray(st.position, dtype=np.float32)
    q = st.rotation  # np.quaternion (w,x,y,z)
    cur_rot = _quat_to_matrix([q.w, q.x, q.y, q.z], quat_format)
    cur_rot_inv = cur_rot.T

    goal = np.asarray(env.current_episode.goals[0].position, dtype=np.float32)
    sp = habitat_sim.ShortestPath()
    sp.requested_start = cur_pos
    sp.requested_end = goal
    if not sim.pathfinder.find_path(sp):
        return None
    pts = [np.asarray(p, dtype=np.float32) for p in sp.points]
    wps = []
    for fpos, fdir in _resample_polyline(pts, predict_size, spacing):
        d_local = cur_rot_inv @ (fpos - cur_pos)
        fdir_local = cur_rot_inv @ fdir
        yaw = float(np.arctan2(fdir_local[0], -fdir_local[2]))
        wps.append([-float(d_local[2]), float(d_local[0]), np.sin(yaw), np.cos(yaw)])
    return np.asarray(wps, dtype=np.float32)


# ---------------------------------------------------------------------------
# DAgger support: expert relabelling of policy-visited states.
# ---------------------------------------------------------------------------
def _region_of(semantic_scene, pos):
    """(region_id, region_category) for a world position, by AABB containment.

    A verbatim copy of ``data_process.gt_following_rollout.region_of``, which is
    NOT imported: that module imports torch at the top level, and this file's
    whole reason for existing is that the sim subprocess must never touch CUDA
    (see the module docstring). ``replay.py`` already keeps its own copy for the
    same reason — keep all three in sync if the rule ever changes.
    """
    if semantic_scene is None:
        return "", "none"
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


class _Expert:
    """The expert DAgger relabels with — the SAME one the BC checkpoint imitated.

    ``gt_following_rollout.py`` does not run a shortest-path follower straight at
    ``goals[0]``; it chains the follower through ``reference_path`` waypoints
    (radius 0.5, switching to 0.25 on the last one) and only then, under
    ``--follow_to_goal``, walks to ``goals[0]``. That distinction is not cosmetic:
    R2R instructions describe a *specific route*, and the geodesic shortest path
    to the goal is frequently a different one. Relabelling with a goal-only
    follower would therefore supervise a route the instruction does not describe.

    So the cursor into ``reference_path`` is reproduced here. It is state-, not
    time-, dependent: it advances whenever the follower reports the current target
    as already reached, which is exactly the behaviour wanted when the policy
    strays and then recovers. Once the chain is exhausted the expert falls through
    to ``goals[0]`` (the ``--follow_to_goal`` tail).

    ``stop_dist`` is the last piece of that parity, and it is not cosmetic either.
    gt_following_rollout's goal walk breaks on ``geodesic <= --goal_stop_dist``
    (0.5 by default in rollout.sh) *before* asking the follower, so R2R_v2 ends a
    median 0.63 m from the goal. Letting the follower run to its own 0.25 m radius
    instead ends ~0.12 m out — measured over the same 50 episodes — i.e. 1-3 extra
    FORWARDs past where BC was taught to stop. Both are far inside the 3 m success
    radius, so neither is "wrong", but mixing the two in one retrain puts two
    different STOP geometries on this model's single weakest behaviour. Applying
    the rule at every lookahead step (not just at termination) is strictly better
    than gt_following_rollout's one synthetic terminal STOP: a policy that wanders
    into the goal region now gets a real STOP label there, which is exactly the
    supervision DAgger is being run to collect. ``stop_dist <= 0`` disables it.
    """

    def __init__(self, env, stop_dist=0.5):
        from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower

        ep = env.current_episode
        ref = ep.reference_path if ep.reference_path is not None else []
        self.ref = [np.asarray(p, dtype=np.float32) for p in ref]
        self.goals = [np.asarray(g.position, dtype=np.float32) for g in ep.goals]
        self.stop_dist = float(stop_dist)
        # gt_following_rollout starts at index 1 — ref[0] is the start pose.
        self.cursor = 1
        self._f05 = ShortestPathFollower(sim=env.sim, goal_radius=0.5,
                                         return_one_hot=False)
        self._f025 = ShortestPathFollower(sim=env.sim, goal_radius=0.25,
                                          return_one_hot=False)

    def arrived(self, sim):
        """Whether the expert should STOP here on the distance rule alone.

        Measured against ALL goals, matching gt_following_rollout's
        ``geodesic_distance(st.position, goal_positions)``."""
        if self.stop_dist <= 0:
            return False
        d = sim.geodesic_distance(sim.get_agent_state().position, self.goals)
        return bool(np.isfinite(d) and d <= self.stop_dist)

    def _follower(self, cursor):
        # Mirrors the radius swap gt_following_rollout performs once the cursor
        # reaches the LAST reference waypoint.
        return self._f025 if cursor >= len(self.ref) - 1 else self._f05

    def path_action(self, cursor):
        """Expert action at the sim's CURRENT agent state, following the reference
        chain from ``cursor``. Returns ``(action, new_cursor)``; action 0 means the
        chain is exhausted and the caller should fall through to the goal."""
        c = cursor
        while c < len(self.ref):
            a = self._follower(c).get_next_action(self.ref[c])
            a = 0 if a is None else int(a)
            if a != 0:
                return a, c
            c += 1
        return 0, c

    def goal_action(self):
        a = self._f025.get_next_action(self.goals[0])
        return 0 if a is None else int(a)


def _expert_chunk(env, expert, K, goal_only=False, stop_dist=None):
    """A coherent K-step expert plan from the CURRENT state, without rendering.

    habitat-sim only rasterises inside ``get_sensor_observations()``, which
    ``Agent.act()`` does not call — ``env.step()`` and ``sim.step()`` DO. So
    save-state -> act() x K -> restore-state is a true multi-step lookahead at
    pathfinder cost (sub-ms), and it is what lets the DAgger label be one real
    plan instead of the first actions of K different plans from K different
    states (which is what a per-frame 1-step relabel degenerates into once
    build_vln_samples.py slices a chunk out of the flat action array).
    """
    sim = env.sim
    agent = sim.get_agent(0)
    st = sim.get_agent_state()
    saved_pos = np.array(st.position, dtype=np.float64, copy=True)
    saved_rot = st.rotation
    if stop_dist is not None:
        expert.stop_dist = float(stop_dist)

    cursor = expert.cursor
    committed = None
    acts = []
    try:
        for i in range(K):
            # The distance rule is checked FIRST, before the follower, and at
            # every lookahead step — the agent moves during the lookahead, so a
            # plan may cross the radius partway through and must terminate there.
            if expert.arrived(sim):
                a = 0
            elif goal_only:
                a = expert.goal_action()
            else:
                a, cursor = expert.path_action(cursor)
                if a == 0:
                    a = expert.goal_action()  # chain exhausted -> walk to goals[0]
            if i == 0:
                # Only the first step reflects the agent's REAL state; the rest are
                # hypothetical, so their cursor drift must not be committed.
                committed = cursor
            acts.append(int(a))
            if a == 0:
                break
            agent.act(int(a))
    finally:
        sim.set_agent_state(saved_pos, saved_rot)
        expert.cursor = cursor if committed is None else committed
    return acts + [0] * (K - len(acts))


def _local_to_world(sim, dx, dy):
    """Local metric offset -> world point. ``dx`` = metres forward, ``dy`` = metres
    left, in the agent's current body frame — the same convention as
    ``_oracle_local_waypoints`` (x_forward = -d_local[2], y_lateral = d_local[0]),
    inverted."""
    st = sim.get_agent_state()
    cur = np.asarray(st.position, dtype=np.float32)
    q = st.rotation
    R = _quat_to_matrix([q.w, q.x, q.y, q.z], "wxyz")
    return cur + R @ np.asarray([dy, 0.0, -dx], dtype=np.float32)


def _follow_local_goal(env, dx, dy, goal_radius, max_actions):
    """Plan navmesh-geodesic actions to a local metric goal (pointing eval).

    Returns ``(actions, info)``. Empty actions means no path was found and the
    caller should fall back to its own controller. The plan is computed by
    lookahead (``Agent.act()``, no render) and the agent is restored afterwards,
    so this op is a pure query — the caller executes the actions through ``step``.
    """
    from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower

    sim = env.sim
    agent = sim.get_agent(0)
    world = _local_to_world(sim, dx, dy)
    snapped = np.asarray(sim.pathfinder.snap_point(world), dtype=np.float32)
    off_navmesh = not bool(np.all(np.isfinite(snapped)))
    if off_navmesh:
        return [], {"path_found": False, "off_navmesh": True,
                    "goal_world": world.tolist(), "snapped": None}

    st = sim.get_agent_state()
    saved_pos = np.array(st.position, dtype=np.float64, copy=True)
    saved_rot = st.rotation
    follower = ShortestPathFollower(sim=sim, goal_radius=goal_radius,
                                    return_one_hot=False)
    actions = []
    try:
        for _ in range(max_actions):
            a = follower.get_next_action(snapped)
            a = 0 if a is None else int(a)
            if a == 0:
                break
            actions.append(a)
            agent.act(a)
    finally:
        sim.set_agent_state(saved_pos, saved_rot)
    return actions, {"path_found": bool(actions), "off_navmesh": False,
                     "goal_world": world.tolist(), "snapped": snapped.tolist()}


def sim_worker(gpu_device_id, config, cmd_q, res_q):
    os.environ.setdefault("MAGNUM_LOG", "quiet")
    os.environ.setdefault("HABITAT_SIM_LOG", "quiet")
    os.environ.setdefault("GLOG_minloglevel", "2")

    import habitat
    from habitat import Env
    # Registers oracle_success / oracle_navigation_error before Env is built.
    from habitat_extensions import measures  # noqa: F401

    with habitat.config.read_write(config):
        config.habitat.simulator.habitat_sim_v0.gpu_device_id = gpu_device_id

    env = Env(config=config)

    eps = [
        (i, ep.scene_id, ep.episode_id, ep.instruction.instruction_text)
        for i, ep in enumerate(env.episodes)
    ]
    res_q.put(eps)

    try:
        # Per-episode state, (re)built on every reset. `expert` is only used by the
        # DAgger collector; building it costs two ShortestPathFollower objects per
        # episode, the same as gt_following_rollout.py.
        expert = None
        semantic_scene = None
        goal_positions = None

        while True:
            cmd = cmd_q.get()
            op = cmd[0]
            if op == "reset":
                env.current_episode = env.episodes[cmd[1]]
                obs = env.reset()
                semantic_scene = env.sim.semantic_scene
                goal_positions = [
                    np.asarray(g.position, dtype=np.float32)
                    for g in env.current_episode.goals
                ]
                try:
                    expert = _Expert(env)
                except Exception:
                    expert = None  # e.g. an episode with no reference_path
                res_q.put((np.asarray(obs["rgb"]), bool(env.episode_over)))
            elif op == "step":
                obs = env.step(cmd[1])
                res_q.put((np.asarray(obs["rgb"]), bool(env.episode_over)))
            elif op == "diagnostics":
                try:
                    from trace_support import diagnostic_state
                    res_q.put(diagnostic_state(env))
                except Exception as exc:
                    res_q.put({"error": "diagnostics: " + repr(exc)})
            elif op == "metrics":
                m = env.get_metrics()
                res_q.put(
                    {
                        k: float(m[k])
                        for k in (
                            "success",
                            "spl",
                            "oracle_success",
                            "distance_to_goal",
                            "ndtw",
                            "sdtw",
                            "path_length",
                            "steps_taken",
                        )
                        if k in m
                    }
                )
            elif op == "oracle":
                try:
                    res_q.put(_oracle_local_waypoints(env, int(cmd[1]), float(cmd[2])))
                except Exception as e:  # never crash the sim on an oracle hiccup
                    res_q.put(None)
            elif op == "expert_chunk":
                # ("expert_chunk", K, goal_only, stop_dist) -> list[int] length K
                K = int(cmd[1])
                goal_only = bool(cmd[2]) if len(cmd) > 2 else False
                stop_dist = float(cmd[3]) if len(cmd) > 3 else None
                if expert is None:
                    res_q.put([0] * K)
                else:
                    try:
                        res_q.put(_expert_chunk(env, expert, K, goal_only,
                                                stop_dist))
                    except Exception as e:
                        # A label we cannot trust is worse than none: surface it so
                        # the collector drops the episode rather than writing zeros
                        # (which would read as "STOP here").
                        res_q.put({"error": f"expert_chunk: {e!r}"})
            elif op == "frame_state":
                st = env.sim.get_agent_state()
                pos = np.asarray(st.position, dtype=np.float32)
                q = st.rotation
                rid, rcat = _region_of(semantic_scene, pos)
                res_q.put({
                    "position": pos,
                    "rotation": np.array([q.w, q.x, q.y, q.z], dtype=np.float32),
                    "geodesic": float(
                        env.sim.geodesic_distance(pos, goal_positions)),
                    "region_id": rid,
                    "region_category": rcat,
                })
            elif op == "episode_geom":
                ep = env.current_episode
                ref = ep.reference_path if ep.reference_path is not None else []
                res_q.put({
                    "goals": np.asarray(
                        [g.position for g in ep.goals], dtype=np.float32),
                    "reference_path": np.asarray(ref, dtype=np.float32),
                    "start": np.asarray(ep.start_position, dtype=np.float32),
                })
            elif op == "agent_state":
                st = env.sim.get_agent_state()
                q = st.rotation
                res_q.put({
                    "position": np.asarray(st.position, dtype=np.float32).tolist(),
                    "rotation": [float(q.w), float(q.x), float(q.y), float(q.z)],
                })
            elif op == "follow":
                # ("follow", dx, dy, goal_radius, max_actions) -> (actions, info)
                res_q.put(_follow_local_goal(
                    env, float(cmd[1]), float(cmd[2]), float(cmd[3]), int(cmd[4])))
            elif op == "close":
                break
            else:
                # Without this branch the loop falls through, the parent blocks
                # forever on res_q.get(), and the whole rank hangs with no output.
                res_q.put(("__unknown_op__", op))
    finally:
        try:
            env.close()
        except Exception:
            pass

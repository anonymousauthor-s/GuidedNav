"""Shared primitives for the discrete and continuous Habitat evaluators.

Both ``eval_discrete_action`` and ``eval_continuous_action`` need the same
distributed bootstrap, the same parent-side habitat ``Env`` stand-ins, and the
same sim-GPU resolution. They used to carry byte-identical copies of all of it;
this module is the single source so the two evaluators can't drift apart.

Deliberately light imports: only ``os``/``torch``/``dist``/``timedelta`` at
module load. ``RemoteEnv`` imports ``multiprocessing`` and ``sim_worker`` lazily
inside ``__init__`` (the sim subprocess must stay free of torch/CUDA), so this
module does NOT pull in habitat — callers construct the real ``Env`` themselves.
"""
import os
from datetime import timedelta

import torch
import torch.distributed as dist


# ---------------------------------------------------------------------------
# Minimal distributed helpers (replaces StreamVLN's utils.dist). Works both
# under torchrun (multi-proc) and as a plain single-process run.
# ---------------------------------------------------------------------------
def get_rank() -> int:
    return int(os.environ.get("RANK", 0))


def get_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", 1))


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))


def init_distributed() -> None:
    if torch.cuda.is_available():
        _n = torch.cuda.device_count()
        # Model device: explicit MODEL_GPUS list (rank i -> MODEL_GPUS[i]) wins;
        # else MODEL_GPU_BASE + local_rank. Default base 0 = original behavior.
        _mg = os.environ.get("MODEL_GPUS")
        if _mg:
            _devs = [int(x) for x in _mg.split(",") if x != ""]
            torch.cuda.set_device(_devs[get_local_rank() % len(_devs)])
        else:
            _base = int(os.environ.get("MODEL_GPU_BASE", "0"))
            torch.cuda.set_device((_base + get_local_rank()) % max(_n, 1))
    if get_world_size() > 1 and not dist.is_initialized():
        # Oversubscription (>1 rank per GPU): NCCL forbids two ranks in a
        # communicator sharing a device -> use gloo for the tiny end-of-run
        # collective when world_size > #GPUs.
        _oversub = torch.cuda.is_available() and get_world_size() > torch.cuda.device_count()
        backend = "gloo" if (_oversub or not torch.cuda.is_available()) else "nccl"
        # Long timeout: the only collective is the final aggregation barrier, and a
        # straggler rank grinding a non-stopping (max-step) episode must not trip
        # the default 600s wait while the others idle at the barrier. (Helps
        # stragglers only — a genuinely hung rank still needs a per-episode
        # sim-step watchdog.)
        dist.init_process_group(backend=backend, timeout=timedelta(hours=2))


def resolve_sim_gpu() -> int:
    """Sim render device for this rank.

    Explicit ``SIM_GPUS`` list (rank i -> SIM_GPUS[i % len]) wins; else
    ``SIM_GPU_BASE + local_rank``.

    **One sim rank per GPU.** Keeping the model off the sim card is necessary
    but NOT sufficient: multiple habitat EGL contexts on one card do not
    coexist. 7/GPU pegs it at 100% util with zero completions, 3/GPU crashes 4
    of 6 sim subprocesses, and 2/GPU is the nastiest because it does not fail
    loudly — it just runs at 1700 ms/step instead of ~10 ms (measured
    2026-08-05 on 157, model on separate cards throughout). Sizing a run is
    therefore capped by the sim cards: N sim GPUs -> N workers.
    """
    n = max(torch.cuda.device_count(), 1) if torch.cuda.is_available() else 1
    sg = os.environ.get("SIM_GPUS")
    if sg:
        devs = [int(x) for x in sg.split(",") if x != ""]
        return devs[get_local_rank() % len(devs)]
    base = int(os.environ.get("SIM_GPU_BASE", "0"))
    return (base + get_local_rank()) % n


# ---------------------------------------------------------------------------
# RemoteEnv — run habitat in a subprocess on a *different* GPU than the model.
#
# On this box, habitat's OpenGL render and the model's CUDA compute contend for
# the GPU: interleaving model.generate with env.step on the SAME physical GPU
# blows env.step from ~4 ms to ~5 s (CUDA preempts the graphics engine). This
# persists even with the sim in a separate process on the same GPU, or same
# process on a different GPU. Only "separate process AND separate GPU" restores
# ~4 ms. RemoteEnv hosts the real Env in a spawned subprocess (see
# ``sim_worker.py`` — no torch/CUDA there) pinned to ``sim_gpu``, and proxies
# just the subset the evaluators need. Enable via env var REMOTE_SIM=1.
# ---------------------------------------------------------------------------
class _EpDesc:
    """Lightweight parent-side stand-in for a habitat Episode."""

    __slots__ = ("_idx", "scene_id", "episode_id", "instruction")

    class _Instr:
        __slots__ = ("instruction_text",)

        def __init__(self, text):
            self.instruction_text = text

    def __init__(self, idx, scene_id, episode_id, instruction_text):
        self._idx = idx
        self.scene_id = scene_id
        self.episode_id = episode_id
        self.instruction = _EpDesc._Instr(instruction_text)


class RemoteEnv:
    """Drop-in stand-in for habitat ``Env`` (only the methods the evaluators
    use) that runs the real Env in a spawned subprocess on ``sim_gpu``."""

    def __init__(self, config, sim_gpu):
        import multiprocessing as mp
        from qwenvl.eval.sim_worker import sim_worker

        ctx = mp.get_context("spawn")
        self._cmd_q = ctx.Queue()
        self._res_q = ctx.Queue()
        self._p = ctx.Process(
            target=sim_worker,
            args=(sim_gpu, config, self._cmd_q, self._res_q),
            daemon=True,
        )
        self._p.start()
        eps = self._res_q.get()  # episode descriptors (metadata only)
        self.episodes = [_EpDesc(*e) for e in eps]
        self._cur = None
        self._pending_idx = None
        self._episode_over = True

    @property
    def current_episode(self):
        return self._cur

    @current_episode.setter
    def current_episode(self, ep):
        self._cur = ep
        self._pending_idx = ep._idx

    def _rpc(self, *cmd):
        """Send one op and return its reply, turning the worker's error markers
        into exceptions here. Without this an op the worker does not implement
        used to fall through its dispatch chain with no reply at all, and the
        parent blocked forever on ``res_q.get()`` — a silent, output-free hang."""
        self._cmd_q.put(cmd)
        res = self._res_q.get()
        if (isinstance(res, tuple) and len(res) == 2
                and isinstance(res[0], str) and res[0] == "__unknown_op__"):
            raise RuntimeError(f"sim_worker does not implement op {res[1]!r}")
        if isinstance(res, dict) and "error" in res and len(res) == 1:
            raise RuntimeError(f"sim_worker: {res['error']}")
        return res

    def reset(self):
        rgb, over = self._rpc("reset", self._pending_idx)
        self._episode_over = over
        return {"rgb": rgb}

    @property
    def episode_over(self):
        return self._episode_over

    def step(self, action):
        rgb, over = self._rpc("step", int(action))
        self._episode_over = over
        return {"rgb": rgb}

    def get_metrics(self):
        return self._rpc("metrics")

    def oracle_waypoints(self, predict_size, spacing):
        # Continuous evaluator only (oracle controller self-check); harmless here.
        return self._rpc("oracle", int(predict_size), float(spacing))

    def expert_chunk(self, k, goal_only=False, stop_dist=None):
        """A coherent ``k``-step expert plan from the agent's current state, as a
        list of habitat action ids (0-padded). DAgger's label. Pure query — the
        worker restores the agent afterwards and does not render.

        ``stop_dist`` (metres, geodesic to the goals) is where the expert calls
        STOP; ``None`` leaves the worker's per-episode default (0.5, matching
        ``gt_following_rollout.py --goal_stop_dist``). Pass the same value the
        rest of the training mix was rolled out with, or the DAgger STOP labels
        land at a different distance than every other dataset's."""
        cmd = ("expert_chunk", int(k), bool(goal_only))
        if stop_dist is not None:
            cmd += (float(stop_dist),)
        return self._rpc(*cmd)

    def frame_state(self):
        """Everything the DAgger npz needs for the current frame:
        ``{position, rotation(wxyz), geodesic, region_id, region_category}``."""
        return self._rpc("frame_state")

    def follow_local_goal(self, dx, dy, goal_radius=0.25, max_actions=200):
        """Plan navmesh-geodesic actions to a local metric goal (pointing eval).

        The goal->world conversion and the ShortestPathFollower both need the live
        ``sim`` (agent pose + pathfinder), which only exists in the subprocess, so
        this is an RPC. Returns ``(actions, info)``; empty actions means no path
        (off-navmesh goal) and the caller should fall back to the open-loop
        controller."""
        return self._rpc("follow", float(dx), float(dy),
                         float(goal_radius), int(max_actions))

    def agent_state(self):
        """Current agent world pose ``{"position":[x,y,z], "rotation":[w,x,y,z]}``.
        Viz only — no mutation. Lets the parent place emitted point-goals in world
        coordinates to compare against the GT reference_path."""
        return self._rpc("agent_state")

    def episode_geom(self):
        """Current episode's ``{"goals":[[x,y,z]...], "reference_path":[...],
        "start":[x,y,z]}`` in world coords. Viz only (called once per episode)."""
        return self._rpc("episode_geom")

    def close(self):
        try:
            self._cmd_q.put(("close",))
            self._p.join(timeout=30)
        except Exception:
            pass
        if self._p.is_alive():
            self._p.terminate()

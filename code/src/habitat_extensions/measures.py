r"""VLN-CE measures used by the R2R and RxR evaluators.

The nDTW implementation follows the official RxR-Habitat evaluator: FastDTW
with radius 1 by default (or exact DTW when configured), Euclidean point
distance, consecutive duplicate agent positions removed, and normalization by
``len(reference) * success_distance``.  Alignment is deferred until the metric
is read so the final score is unchanged without recomputing it after each step.
"""

import gzip
import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from habitat.config.default_structured_configs import MeasurementConfig
from habitat.core.embodied_task import EmbodiedTask, Measure
from habitat.core.registry import registry
from habitat.tasks.nav.nav import DistanceToGoal, Success
from hydra.core.config_store import ConfigStore


@dataclass
class NDTWMeasurementConfig(MeasurementConfig):
    type: str = "NDTW"
    gt_path: str = ""
    split: str = "val_unseen"
    success_distance: float = 3.0
    fdtw: bool = True


# ``measures`` is imported before Habitat composes the YAML, so registering the
# structured node here lets the config carry the RxR-specific GT path fields.
ConfigStore.instance().store(
    package="habitat.task.measurements.ndtw",
    group="habitat/task/measurements",
    name="ndtw",
    node=NDTWMeasurementConfig,
)


def _config_value(config: Any, name: str, default: Any = None) -> Any:
    """Read both Hydra/lower-case and legacy Habitat/yacs config fields."""
    for key in (name, name.lower(), name.upper()):
        try:
            value = config.get(key, None)
        except (AttributeError, TypeError):
            value = getattr(config, key, None)
        if value is not None:
            return value
    return default


def euclidean_distance(a: Sequence[float], b: Sequence[float]) -> float:
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b), ord=2))


def _dtw(
    x: Sequence[Sequence[float]],
    y: Sequence[Sequence[float]],
    window=None,
):
    """Return classic DTW distance/path with the recurrence used by fastdtw."""
    len_x, len_y = len(x), len(y)
    if window is None:
        window = ((i, j) for i in range(len_x) for j in range(len_y))
    window = ((i + 1, j + 1) for i, j in window)
    matrix = defaultdict(lambda: (float("inf"),))
    matrix[0, 0] = (0.0, 0, 0)
    for i, j in window:
        cost = euclidean_distance(x[i - 1], y[j - 1])
        matrix[i, j] = min(
            (matrix[i - 1, j][0] + cost, i - 1, j),
            (matrix[i, j - 1][0] + cost, i, j - 1),
            (matrix[i - 1, j - 1][0] + cost, i - 1, j - 1),
            key=lambda value: value[0],
        )
    path = []
    i, j = len_x, len_y
    while not (i == j == 0):
        path.append((i - 1, j - 1))
        i, j = matrix[i, j][1], matrix[i, j][2]
    path.reverse()
    return float(matrix[len_x, len_y][0]), path


def _expand_window(path, len_x: int, len_y: int, radius: int):
    expanded_path = set(path)
    for i, j in path:
        for delta_i in range(-radius, radius + 1):
            for delta_j in range(-radius, radius + 1):
                expanded_path.add((i + delta_i, j + delta_j))
    candidates = set()
    for i, j in expanded_path:
        candidates.update(
            ((2 * i, 2 * j), (2 * i, 2 * j + 1),
             (2 * i + 1, 2 * j), (2 * i + 1, 2 * j + 1))
        )
    window = []
    start_j = 0
    for i in range(len_x):
        new_start_j = None
        for j in range(start_j, len_y):
            if (i, j) in candidates:
                window.append((i, j))
                if new_start_j is None:
                    new_start_j = j
            elif new_start_j is not None:
                break
        start_j = new_start_j
    return window


def _fastdtw(x, y, radius: int = 1):
    """Pure-Python equivalent of ``fastdtw.fastdtw(..., radius=1)``."""
    min_time_size = radius + 2
    if len(x) < min_time_size or len(y) < min_time_size:
        return _dtw(x, y)
    x_half = [(x[i] + x[i + 1]) / 2 for i in range(0, len(x) - len(x) % 2, 2)]
    y_half = [(y[i] + y[i + 1]) / 2 for i in range(0, len(y) - len(y) % 2, 2)]
    _, path = _fastdtw(x_half, y_half, radius=radius)
    return _dtw(x, y, _expand_window(path, len(x), len(y), radius))


@registry.register_measure
class PathLength(Measure):
    """Total metric length of the executed trajectory (TL/PL)."""

    cls_uuid: str = "path_length"

    def __init__(self, sim, *args: Any, **kwargs: Any):
        self._sim = sim
        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, *args: Any, **kwargs: Any):
        self._previous_position = np.asarray(
            self._sim.get_agent_state().position, dtype=np.float64
        ).copy()
        self._metric = 0.0

    def update_metric(self, *args: Any, **kwargs: Any):
        current = np.asarray(
            self._sim.get_agent_state().position, dtype=np.float64
        ).copy()
        self._metric += euclidean_distance(self._previous_position, current)
        self._previous_position = current


@registry.register_measure
class StepsTaken(Measure):
    """Number of executed actions; STOP is included, matching VLN-CE."""

    cls_uuid: str = "steps_taken"

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, *args: Any, **kwargs: Any):
        self._metric = 0.0

    def update_metric(self, *args: Any, **kwargs: Any):
        self._metric += 1.0


@registry.register_measure
class OracleNavigationError(Measure):
    """Minimum geodesic distance to the goal reached during the episode."""

    cls_uuid: str = "oracle_navigation_error"

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, episode, task: EmbodiedTask, *args: Any, **kwargs: Any):
        task.measurements.check_measure_dependencies(
            self.uuid, [DistanceToGoal.cls_uuid]
        )
        self._metric = float("inf")
        self.update_metric(episode=episode, task=task, *args, **kwargs)

    def update_metric(self, episode, task: EmbodiedTask, *args: Any, **kwargs: Any):
        distance_to_target = task.measurements.measures[
            DistanceToGoal.cls_uuid
        ].get_metric()
        self._metric = min(self._metric, distance_to_target)


@registry.register_measure
class OracleSuccess(Measure):
    """Whether the trajectory ever enters the success radius."""

    cls_uuid: str = "oracle_success"

    def __init__(self, sim, config, *args: Any, **kwargs: Any):
        self._sim = sim
        self._success_distance = float(
            _config_value(config, "success_distance", 3.0)
        )
        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, episode, task: EmbodiedTask, *args: Any, **kwargs: Any):
        task.measurements.check_measure_dependencies(
            self.uuid, [DistanceToGoal.cls_uuid]
        )
        self._metric = 0.0
        self.update_metric(episode=episode, task=task, *args, **kwargs)

    def update_metric(self, episode, task: EmbodiedTask, *args: Any, **kwargs: Any):
        distance_to_target = task.measurements.measures[
            DistanceToGoal.cls_uuid
        ].get_metric()
        if distance_to_target < self._success_distance:
            self._metric = 1.0


@registry.register_measure
class NDTW(Measure):
    """Official RxR normalized Dynamic Time Warping score."""

    cls_uuid: str = "ndtw"

    def __init__(self, sim, config, *args: Any, **kwargs: Any):
        self._sim = sim
        gt_path = str(_config_value(config, "gt_path"))
        split = str(_config_value(config, "split", "val_unseen"))
        self._success_distance = float(
            _config_value(config, "success_distance", 3.0)
        )
        self._use_fastdtw = bool(_config_value(config, "fdtw", True))
        if not gt_path or gt_path == "None":
            raise ValueError("NDTW requires a gt_path in the Habitat config")
        with gzip.open(gt_path.format(split=split), "rt", encoding="utf-8") as handle:
            self._gt_json = json.load(handle)
        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, episode, *args: Any, **kwargs: Any):
        episode_id = str(episode.episode_id)
        if episode_id not in self._gt_json:
            raise KeyError(f"Episode {episode_id!r} missing from nDTW ground truth")
        self._gt_locations = np.asarray(
            self._gt_json[episode_id]["locations"], dtype=np.float64
        )
        if self._gt_locations.ndim != 2 or len(self._gt_locations) == 0:
            raise ValueError(f"Invalid nDTW reference for episode {episode_id}")
        self._locations = []
        self._previous_position = None
        self._dirty = True
        self._metric = 0.0
        self.update_metric()

    def update_metric(self, *args: Any, **kwargs: Any):
        current = np.asarray(
            self._sim.get_agent_state().position, dtype=np.float64
        ).copy()
        # This equality test intentionally mirrors the official RxR code's
        # exact list comparison for consecutive duplicate locations.
        if self._previous_position is not None and np.array_equal(
            current, self._previous_position
        ):
            return

        self._locations.append(current)
        self._previous_position = current
        self._dirty = True

    def get_metric(self):
        # The evaluator reads nDTW at episode end. Delaying DTW until that read
        # preserves the official final value and avoids recomputing the full
        # alignment after every action.
        if self._dirty:
            locations = np.asarray(self._locations, dtype=np.float64)
            if self._use_fastdtw:
                dtw_distance, _ = _fastdtw(locations, self._gt_locations, radius=1)
            else:
                dtw_distance, _ = _dtw(locations, self._gt_locations)
            self._metric = float(
                np.exp(
                    -dtw_distance
                    / (len(self._gt_locations) * self._success_distance)
                )
            )
            self._dirty = False
        return self._metric


@registry.register_measure
class SDTW(Measure):
    """Success-weighted nDTW: ``Success * nDTW``."""

    cls_uuid: str = "sdtw"

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, *args: Any, task: EmbodiedTask, **kwargs: Any):
        task.measurements.check_measure_dependencies(
            self.uuid, [NDTW.cls_uuid, Success.cls_uuid]
        )
        self.update_metric(task=task)

    def update_metric(self, *args: Any, task: EmbodiedTask, **kwargs: Any):
        success = task.measurements.measures[Success.cls_uuid].get_metric()
        ndtw = task.measurements.measures[NDTW.cls_uuid].get_metric()
        self._metric = float(success * ndtw)

"""vln_sampling.py — the SINGLE source of truth for VLN history-frame sampling.

``history_indices`` defines which past frames become the ``<memory>`` block of a
VLN prompt. It is imported by, and must stay identical across:

  * ``data_process/build_vln_samples.py``  — bakes the indices into each offline
    per-step record (training).
  * ``qwenvl/eval/eval_discrete_action.py``   — rebuilds them during rollout.
  * ``qwenvl/eval/eval_continuous_action.py`` — likewise.

This module exists because those three drifted apart once already and it cost a
full eval cycle: the continuous evaluator sampled the last N *consecutive*
frames (~1.75 m of trajectory) while training sampled uniformly over the whole
trajectory. The mismatch erased global instruction progress from the memory
block and made the stop head look untrained — the same checkpoint went from
SR 0 to SR 0.35 once the rollout fed the training distribution, with no
retraining. Any change here must be made ONCE, here, and re-run through
``build_vln_samples.py`` (the offline records cache the indices).

Deliberately numpy-only: ``build_vln_samples.py`` runs 48-way sharded with no
torch/transformers in the process, so this must not pull in the heavy stack.
"""

from typing import List

import numpy as np


def history_indices(c: int, memory_size: int) -> List[int]:
    """History frame indices for a decision at frame ``c``: uniform over the
    whole traversed trajectory, current frame excluded.

    ``linspace(0, c, memory_size)``, rounded to ints and de-duplicated, keeping
    only ``j < c``. Consequences worth knowing, since prompts are variable-length
    and every consumer must reproduce the same ramp:

    * ``c == 0`` -> ``[]``. The caller must then omit the whole "These are your
      historical observations: <memory>." clause, not emit an empty one.
    * ``1 <= c <= memory_size - 1`` -> ``[0, 1, ..., c-1]``: EVERY traversed
      frame, no subsampling. Early-episode history is lossless.
    * ``c >= memory_size`` -> saturates at ``memory_size - 1`` frames (7 for the
      default ``memory_size=8``), because the ``linspace`` endpoint ``c`` is the
      current frame and gets filtered out. The name says 8, the block holds 7 —
      intentional, and identical on both sides.

    Note the span keeps growing with ``c`` while the count stays fixed, so late
    in a long rollout consecutive memory frames are tens of steps apart. Training
    only ever sees ``c`` up to the GT trajectory length, so a rollout that wanders
    far past it drifts out of distribution here — a known limitation, not a bug.
    """
    if c <= 0:
        return []
    raw = np.linspace(0, c, memory_size)
    idxs = sorted({int(round(x)) for x in raw})
    return [j for j in idxs if j < c]

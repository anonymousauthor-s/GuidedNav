"""Parameter-free action-state/object-region alignment for SOMA.

The controller observes one intermediate language-decoder layer without
modifying its output.  It compares the state immediately before the first
supervised action token with current-image patch states and assigns a training
loss to the subtask-selected mask.  There are no query tokens, projectors,
residuals, gates, or inference-time parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class SomaConfig:
    student_layer: int = 27
    temperature: float = 0.07
    epsilon: float = 1e-8

    def __post_init__(self):
        if self.student_layer < 1:
            raise ValueError("SOMA student_layer counts decoder outputs from one")
        if not self.temperature > 0:
            raise ValueError("SOMA temperature must be positive")
        if not 0 < self.epsilon < 1:
            raise ValueError("SOMA epsilon must lie in (0,1)")


@dataclass
class SomaCapture:
    """Lazy view of one decoder output captured without tensor operations.

    Decoder hooks run inside PyTorch's checkpointed function.  Performing the
    position indexing in that hook changes how many tensors checkpoint saves;
    skipping it on recomputation then raises ``CheckpointError``.  The hook
    therefore stores only the raw output reference and integer metadata.  All
    differentiable indexing happens lazily, after the full policy forward has
    returned and outside the checkpointed block.
    """

    hidden_states: torch.Tensor
    action_positions: torch.Tensor
    current_patch_positions: torch.Tensor
    _action_states: Optional[torch.Tensor] = field(default=None, init=False, repr=False)
    _patch_states: Optional[list[torch.Tensor]] = field(default=None, init=False, repr=False)

    def _materialize(self):
        if self._action_states is not None:
            return
        hidden = self.hidden_states
        actions = self.action_positions.to(hidden.device)
        patches = self.current_patch_positions.to(hidden.device)
        if hidden.shape[0] != actions.numel():
            raise RuntimeError("Stale SOMA batch context")
        if bool((actions < 0).any()) or int(actions.max()) >= hidden.shape[1]:
            raise ValueError("Action-state position lies outside the sequence")
        rows = torch.arange(hidden.shape[0], device=hidden.device)
        action_states = hidden[rows, actions]
        patch_states = []
        for row in range(hidden.shape[0]):
            positions = patches[row]
            positions = positions[positions >= 0]
            if positions.numel() == 0:
                raise ValueError("Every SOMA row requires current-image patches")
            if int(positions.max()) >= hidden.shape[1]:
                raise ValueError("Current-patch position lies outside the sequence")
            if not bool(torch.all(positions[1:] > positions[:-1])):
                raise ValueError("Current-patch positions must be strictly increasing")
            if int(positions[-1]) >= int(actions[row]):
                raise ValueError("Current patches must causally precede the action state")
            patch_states.append(hidden[row, positions])
        self._action_states = action_states
        self._patch_states = patch_states

    @property
    def action_states(self):
        self._materialize()
        return self._action_states

    @property
    def patch_states(self):
        self._materialize()
        return self._patch_states


def unwrap_policy(model):
    """Resolve DDP/DeepSpeed/PEFT wrappers without unwrapping Qwen internals."""

    current = model
    seen = set()
    while id(current) not in seen:
        seen.add(id(current))
        wrapped = getattr(current, "module", None)
        if wrapped is not None and wrapped is not current:
            current = wrapped
            continue
        getter = getattr(current, "get_base_model", None)
        if callable(getter):
            base = getter()
            if base is not current:
                current = base
                continue
        return current
    raise RuntimeError("Cyclic model wrapper")


def decoder_layers(policy):
    candidates = [getattr(policy, "language_model", None)]
    inner = getattr(policy, "model", None)
    candidates.append(getattr(inner, "language_model", None))
    candidates.extend((inner, policy))
    for candidate in candidates:
        layers = getattr(candidate, "layers", None)
        if layers is not None:
            return layers
    raise ValueError("Cannot locate Qwen language decoder layers")


class SomaController:
    """Read-only decoder hook and mask-attention loss.

    ``set_context`` must be called immediately before the action forward.  A
    later subtask-only forward in the same loss call cannot overwrite the first
    capture.  ``set_context`` at the next step resets it after the preceding
    backward has completed.
    """

    def __init__(self, model, config: SomaConfig):
        self.policy = unwrap_policy(model)
        self.config = config
        layers = decoder_layers(self.policy)
        if config.student_layer > len(layers):
            raise ValueError(
                f"SOMA layer {config.student_layer} exceeds {len(layers)} decoder blocks"
            )
        self.capture: Optional[SomaCapture] = None
        self._context = None
        self._handle = layers[config.student_layer - 1].register_forward_hook(
            self._after_block
        )
        object.__setattr__(self.policy, "_soma_controller", self)

    def set_context(self, *, action_state_positions, current_patch_positions):
        actions = torch.as_tensor(action_state_positions, dtype=torch.long)
        patches = torch.as_tensor(current_patch_positions, dtype=torch.long)
        if actions.ndim != 1:
            raise ValueError("action_state_positions must have shape [batch]")
        if patches.ndim != 2 or patches.shape[0] != actions.numel():
            raise ValueError("current_patch_positions must have shape [batch,max_patches]")
        for row in range(actions.numel()):
            positions = patches[row]
            positions = positions[positions >= 0]
            if positions.numel() == 0:
                raise ValueError("Every SOMA row requires current-image patches")
            if not bool(torch.all(positions[1:] > positions[:-1])):
                raise ValueError("Current-patch positions must be strictly increasing")
            if int(positions[-1]) >= int(actions[row]):
                raise ValueError("Current patches must causally precede the action state")
        self._context = {"actions": actions, "patches": patches}
        self.capture = None

    def _after_block(self, module, args, output):
        # A second (subtask) main-model forward deliberately leaves the action
        # capture untouched.  The hook is observational and always returns the
        # original output object.
        if self._context is None or self.capture is not None:
            return output
        hidden = output[0] if isinstance(output, tuple) else output
        if not torch.is_tensor(hidden) or hidden.ndim != 3:
            raise TypeError("SOMA expected decoder hidden states [batch,sequence,hidden]")
        if self.policy.training and not torch.is_grad_enabled():
            raise RuntimeError("SOMA requires nonreentrant gradient checkpointing")

        actions = self._context["actions"]
        patches = self._context["patches"]
        if hidden.shape[0] != actions.numel():
            raise RuntimeError("Stale SOMA batch context")
        self.capture = SomaCapture(
            hidden_states=hidden,
            action_positions=actions,
            current_patch_positions=patches,
        )
        return output

    def _require_capture(self) -> SomaCapture:
        if self.capture is None:
            raise RuntimeError("Run the action forward before requesting SOMA loss")
        return self.capture

    def attention_probabilities(self) -> list[torch.Tensor]:
        """Return per-row distributions over current-image patches."""

        capture = self._require_capture()
        distributions = []
        for query, patches in zip(capture.action_states, capture.patch_states):
            query = F.normalize(query.float(), dim=-1)
            keys = F.normalize(patches.float(), dim=-1)
            logits = (keys @ query) / self.config.temperature
            distributions.append(logits.softmax(dim=0))
        return distributions

    def grounding_loss(self, mask_targets, eligible):
        """Return summed eligible-frame losses and the eligible-frame count.

        The loss maximizes the action state's probability mass inside the soft
        mask.  The trainer is responsible for division by the globally reduced
        eligible count across ranks and gradient-accumulation microbatches.
        """

        capture = self._require_capture()
        eligible = torch.as_tensor(
            eligible, device=capture.action_states.device, dtype=torch.bool
        )
        if eligible.ndim != 1 or eligible.numel() != len(capture.patch_states):
            raise ValueError("eligible must contain one flag per batch row")
        if len(mask_targets) != len(capture.patch_states):
            raise ValueError("mask_targets must contain one row per batch sample")

        probabilities = self.attention_probabilities()
        total = capture.action_states.sum().float() * 0.0
        total = total + sum(states.sum().float() * 0.0 for states in capture.patch_states)
        for row, probability in enumerate(probabilities):
            if not bool(eligible[row]):
                continue
            target = torch.as_tensor(
                mask_targets[row], device=probability.device, dtype=torch.float32
            ).flatten()
            if target.numel() < probability.numel():
                raise ValueError("Mask target is shorter than current-image patches")
            target = target[: probability.numel()]
            if not bool(torch.isfinite(target).all()) or bool(
                ((target < 0) | (target > 1)).any()
            ):
                raise ValueError("Mask target must be finite soft occupancy in [0,1]")
            if not bool(target.sum() > 0):
                raise ValueError("Eligible masks must contain a visible positive region")
            mass = (probability * target).sum()
            total = total - torch.log(mass.clamp_min(self.config.epsilon))
        return total, eligible.sum()

    def remove_hook(self):
        self._handle.remove()
        self._context = None
        self.capture = None


def attach_soma(model, config: SomaConfig):
    policy = unwrap_policy(model)
    if getattr(policy, "_soma_controller", None) is not None:
        raise ValueError("SOMA is already attached to this policy")
    return SomaController(model, config)


def get_soma(model):
    controller = getattr(unwrap_policy(model), "_soma_controller", None)
    if controller is None:
        raise ValueError("SOMA has not been attached to this policy")
    return controller

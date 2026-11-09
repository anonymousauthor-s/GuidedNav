"""Training-only, non-mutating capture of native Qwen3-VL Q/K attention."""
import torch
from transformers.models.qwen3_vl.modeling_qwen3_vl import rotate_half
from qwenvl.train.soma_modules import unwrap_policy, decoder_layers


class NativeAttention:
    def __init__(self, model, layer=27, heads=(0, 16)):
        self.attn = decoder_layers(unwrap_policy(model))[layer - 1].self_attn
        self.heads = tuple(heads)
        assert self.attn.attention_dropout == 0
        assert len(set(h // self.attn.num_key_value_groups for h in heads)) == len(heads)
        self.active = False
        self.q = self.k = self.rope = self.mask = None
        self.handles = [
            self.attn.register_forward_pre_hook(self._before, with_kwargs=True),
            self.attn.q_norm.register_forward_hook(self._query),
            self.attn.k_norm.register_forward_hook(self._key),
        ]

    # Hooks only store references. Tensor operations here would change the
    # non-reentrant checkpoint saved-tensor contract on recomputation.
    def _before(self, module, args, kwargs):
        if self.active and self.rope is None:
            self.rope = kwargs.get('position_embeddings', args[1] if len(args) > 1 else None)
            self.mask = kwargs.get('attention_mask', args[2] if len(args) > 2 else None)
            assert kwargs.get('past_key_values') is None

    def _query(self, module, args, output):
        if self.active and self.q is None:
            self.q = output

    def _key(self, module, args, output):
        if self.active and self.k is None:
            self.k = output

    def begin(self):
        self.q = self.k = self.rope = self.mask = None
        self.active = True

    def clear(self):
        self.active = False
        self.q = self.k = self.rope = self.mask = None

    def row_logits(self, action_position):
        assert self.q is not None and self.k is not None and self.rope is not None
        self.active = False
        assert self.q.shape[0] == 1, 'Locked microbatch is one'
        pos = int(action_position.reshape(-1)[0])
        assert 0 <= pos < self.q.shape[1]
        heads = list(self.heads)
        kv = [h // self.attn.num_key_value_groups for h in heads]
        # Use already computed Q/K normalization and the exact mRoPE cos/sin
        # from the action forward; never rerun LoRA projections/dropout.
        q = self.q[0, pos, heads, :]
        k = self.k[0, :pos + 1, kv, :].transpose(0, 1)
        cos, sin = self.rope
        q = q * cos[0, pos] + rotate_half(q) * sin[0, pos]
        k = k * cos[0, :pos + 1] + rotate_half(k) * sin[0, :pos + 1]
        logits = torch.einsum('hd,hkd->hk', q.float(), k.float()) * self.attn.scaling
        # Causal clipping above excludes every ground-truth answer token.
        if self.mask is not None:
            if self.mask.ndim == 2:
                logits = logits.masked_fill(~self.mask[0, :pos + 1].bool(), -torch.inf)
            elif self.mask.ndim == 4:
                logits = logits + self.mask[0, 0, pos, :pos + 1].float()
            else:
                raise ValueError('Unexpected native attention mask')
        return logits

    @staticmethod
    def distribution(logits, positions):
        positions = positions.reshape(-1)
        positions = positions[positions >= 0]
        assert positions.numel() > 0 and int(positions.max()) < logits.shape[-1]
        selected = logits[:, positions]
        log_p = torch.logsumexp(selected.log_softmax(-1), dim=0) - torch.log(
            logits.new_tensor(float(logits.shape[0])))
        with torch.no_grad():
            modality_mass = (torch.logsumexp(selected, -1) - torch.logsumexp(logits, -1)).exp().mean()
        return log_p, modality_mass

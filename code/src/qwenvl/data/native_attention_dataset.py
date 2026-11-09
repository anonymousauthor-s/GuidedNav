"""Teacher metadata is added after the unchanged SOMA action encoding."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from qwenvl.data.soma_dataset import SOMADataset, SOMACollator


def token_span(tokenizer, ids, instruction, span):
    text = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    if encoded['input_ids'] != ids:
        raise ValueError('Token roundtrip changed original action prefix')
    if text.count(instruction) != 1:
        raise ValueError('Original instruction must occur once in action prefix')
    start = text.index(instruction)
    end = start + len(instruction)
    a, b = map(int, span)
    positions, target = [], []
    for i, (lo, hi) in enumerate(encoded['offset_mapping']):
        if hi > start and lo < end and hi > lo:
            positions.append(i)
            target.append(a >= 0 and hi > start + a and lo < start + b)
    if not positions or (a >= 0 and not any(target)):
        raise ValueError('Empty instruction or valid span token mapping')
    return torch.tensor(positions), torch.tensor(target, dtype=torch.bool)


class NativeDataset(SOMADataset):
    def __init__(self, *args, native_arm, native_sidecar, **kwargs):
        super().__init__(*args, **kwargs)
        self.native_arm = native_arm
        self.spans = self.weights = None
        if native_arm == 'S':
            root = Path(native_sidecar)
            audit = json.loads((root / 'audit.json').read_text())
            if audit['source_sha256'] != self.manifest['source']['sha256']:
                raise ValueError('Sidecar source mismatch')
            for name in ('spans', 'weights'):
                p = root / (name + '.npy')
                if hashlib.sha256(p.read_bytes()).hexdigest() != audit['files'][name]:
                    raise ValueError('Sidecar hash mismatch')
                setattr(self, name, np.load(p, mmap_mode='r', allow_pickle=False))

    def __getitem__(self, index):
        item = super().__getitem__(index)
        positions = torch.empty(0, dtype=torch.long)
        target = torch.empty(0, dtype=torch.bool)
        weight = 0.0
        if self.native_arm == 'S':
            source = self.source_index_for_sample(index)
            raw = self._read_record(index)
            prefix = item['input_ids'].reshape(-1)[:int(item['action_state_position']) + 1].tolist()
            positions, target = token_span(self.tokenizer, prefix, raw['instruction'], self.spans[source])
            weight = float(self.weights[source])
            if (weight > 0) != bool(target.any()):
                raise ValueError('Span eligibility/weight mismatch')
        item.update(span_positions=positions, span_target=target, span_weight=torch.tensor(weight))
        return item


class NativeCollator(SOMACollator):
    def __call__(self, instances):
        batch = super().__call__(instances)
        batch['span_positions'] = pad_sequence([i['span_positions'] for i in instances], batch_first=True, padding_value=-1)
        batch['span_target'] = pad_sequence([i['span_target'] for i in instances], batch_first=True, padding_value=False)
        batch['span_weight'] = torch.stack([i['span_weight'] for i in instances])
        return batch

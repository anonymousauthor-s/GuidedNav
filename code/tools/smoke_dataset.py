"""Decode and collate annotated R2R/RxR examples without loading an 8B model."""
import argparse
import json
import numpy as np
from transformers import AutoProcessor
from qwenvl.data.full_vln_dataset import load_full_dataset, FullVLNCollator
from qwenvl.train.argument import DataArguments
from qwenvl.train.train_discrete_action_temporal_spatial import TemporalSpatialArguments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--processor', required=True)
    parser.add_argument('--per-source', type=int, default=4)
    args = parser.parse_args()
    processor = AutoProcessor.from_pretrained(args.processor, local_files_only=True)
    processor.tokenizer.model_max_length = 8192
    data = DataArguments()
    data.model_type, data.min_pixels, data.max_pixels = 'qwen3vl', 12544, 200704
    temporal = TemporalSpatialArguments(current_merge_ratio=0.25, current_temporal_decay='0.2,0.35,0.5',
                                        current_fusion_mode='gated_residual', current_matching_mode='mutual')
    dataset, manifest = load_full_dataset(processor, data, temporal, args.manifest)
    collator = FullVLNCollator(processor.tokenizer)
    results = []
    for source in dataset.datasets:
        eligible = np.flatnonzero(source.arrays['o_eligible'] & (source.arrays['weights'] > 0))
        if len(eligible) < args.per_source:
            raise ValueError('Not enough positive S/O samples for the decode audit')
        chosen = np.random.RandomState(42).choice(eligible, args.per_source, replace=False)
        for index in chosen:
            batch = collator([source[int(index)]])
            if not bool(batch['mask_valid'][0]) or not bool(batch['span_target'].any()):
                raise ValueError('Expected positive mask and subtask span')
            results.append({'source': source.source['name'], 'index': int(index),
                            'sequence_length': int(batch['input_ids'].shape[-1]),
                            'current_patch_count': int((batch['current_patch_positions'] >= 0).sum())})
    print(json.dumps({'rows': len(dataset), 'examples': results, 'passed': True}, indent=2))


if __name__ == '__main__':
    main()

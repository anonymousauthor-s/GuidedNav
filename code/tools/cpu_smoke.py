"""CPU-only import, CLI, history and fail-closed identity smoke; never loads model weights."""
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    modules = [
        'qwenvl.train.train_full_e6', 'qwenvl.data.full_vln_dataset',
        'qwenvl.train.full_e6_trainer', 'qwenvl.train.native_attention',
        'qwenvl.inference', 'qwenvl.reference_policy', 'qwenvl.infer_frames',
        'vggt.models.vggt',
    ]
    for name in modules:
        importlib.import_module(name)
    from qwenvl.data.vln_sampling import history_indices
    for current in (0, 1, 7, 8, 30, 500):
        indices = history_indices(current, 8)
        if list(indices) != sorted(set(indices)) or any(index < 0 or index >= current for index in indices):
            raise AssertionError('History must be sorted, unique and exclude the current frame')
        if len(indices) > 7:
            raise AssertionError('The final history protocol has at most seven memory observations')
    from collections import OrderedDict
    from qwenvl.reference_policy import ReferencePolicy
    policy = ReferencePolicy()
    policy.actions2idx = OrderedDict({'STOP': [0], '↑': [1], '←': [2], '→': [3]})
    if policy.parse_actions('→ ↑ STOP') != [3, 1, 0]:
        raise AssertionError('Final action parser changed')
    from qwenvl.model_identity import verify_model_identity
    with tempfile.TemporaryDirectory(prefix='guidednav-identity-negative-') as temporary:
        path = Path(temporary)
        (path / 'model-00001-of-00004.safetensors').write_bytes(b'incorrect checkpoint')
        try:
            verify_model_identity(path, path)
        except ValueError:
            pass
        else:
            raise AssertionError('Incorrect base must fail closed')
    commands = [
        [sys.executable, '-m', 'qwenvl.train.train_full_e6', '--help'],
        [sys.executable, '-m', 'qwenvl.infer_frames', '--help'],
        [sys.executable, str(Path(__file__).with_name('prepare_data_manifest.py')), '--help'],
        [sys.executable, str(Path(__file__).with_name('download_teacher.py')), '--help'],
        [sys.executable, str(Path(__file__).with_name('build_rxr_navmesh.py')), '--help'],
        [sys.executable, str(Path(__file__).with_name('render_training_rgb.py')), '--help'],
    ]
    for command in commands:
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode or 'usage:' not in result.stdout.lower():
            raise RuntimeError(f'CLI failed: {command}: {result.stderr}')
    import torch
    if torch.cuda.is_available():
        raise RuntimeError('Run this smoke without visible CUDA devices')
    print(json.dumps({'passed': True, 'imported_modules': modules,
                      'cli_help_checks': len(commands), 'history_cases': 6,
                      'identity_negative_check': True, 'cuda_visible': False}, indent=2))


if __name__ == '__main__':
    main()

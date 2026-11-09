"""Relocate a frozen final-stage manifest after verifying immutable payload hashes."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 ** 2), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--array-root', required=True)
    parser.add_argument('--output', required=True)
    for name in ('r2r', 'rxr'):
        parser.add_argument(f'--{name}-jsonl', required=True)
        parser.add_argument(f'--{name}-rgb', required=True)
        parser.add_argument(f'--{name}-masks', required=True)
    args = parser.parse_args()
    original = Path(args.manifest).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    manifest = json.loads(original.read_text(encoding='utf-8'))
    if (manifest['format'] != 'full_vln_expert25_so_v1' or manifest['selected_rows'] != 629788
            or [s['name'] for s in manifest['sources']] != ['r2r', 'rxr']):
        raise ValueError('Expected the final expert25 manifest')
    for source in manifest['sources']:
        name = source['name']
        path = Path(getattr(args, f'{name}_jsonl')).resolve()
        if sha256(path) != source['sha256'] or path.stat().st_size != source['size_bytes']:
            raise ValueError(f'{name} JSONL payload differs from the frozen export')
        source.update(path=str(path), mtime_ns=path.stat().st_mtime_ns,
                      rgb_root=str(Path(getattr(args, f'{name}_rgb')).resolve()),
                      mask_root=str(Path(getattr(args, f'{name}_masks')).resolve()))
        for array_name, meta in source['arrays'].items():
            relative = Path(meta['path'])
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('Input array paths must be safe release-relative paths')
            array_path = (Path(args.array_root) / relative).resolve()
            if sha256(array_path) != meta['sha256']:
                raise ValueError(f'{name}/{array_name} hash mismatch')
            array = np.load(array_path, mmap_mode='r', allow_pickle=False)
            if list(array.shape) != meta['shape'] or str(array.dtype) != meta['dtype']:
                raise ValueError(f'{name}/{array_name} shape or dtype mismatch')
            meta['path'] = str(array_path)
        if source['selected_rows'] != source['source_rows'] // 4 or not source['annotation_complete']:
            raise ValueError('Invalid final-stage selection metadata')
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest['relocation'] = {'verified_input_manifest_sha256': sha256(original),
                               'payloads_verified': True}
    output.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'output': str(output), 'rows': manifest['selected_rows'], 'passed': True}))


if __name__ == '__main__':
    main()

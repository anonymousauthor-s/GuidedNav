"""Verify the exact final RxR benchmark, GT and navigation-mesh audit before loading the VLM."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path

DATA_SHA = '93abddce58e8232867a8738daa8107eb82953693a9d5e70779d1eb00386b222a'
GT_SHA = 'bdfbb624132501bf297eb0ea16a99e38682668ba5e9ce1554ca5cdcde6b65a0d'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--gt', required=True)
    parser.add_argument('--audit', required=True)
    args = parser.parse_args()
    data_path, gt_path, audit_path = Path(args.dataset), Path(args.gt), Path(args.audit)
    audit = json.loads(audit_path.read_text())
    if (not audit.get('complete') or audit.get('count') != 3669 or audit.get('unreachable')
            or audit.get('invalid_gt') or not math.isclose(audit['height'], 0.88)
            or not math.isclose(audit['radius'], 0.18)):
        raise ValueError('Incomplete or incorrect RxR geometry audit')
    for path, expected, key in ((data_path, DATA_SHA, 'input_sha256'), (gt_path, GT_SHA, 'gt_sha256')):
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected or audit.get(key) != expected:
            raise ValueError(f'RxR payload/audit SHA mismatch: {path.name}')
    with gzip.open(data_path, 'rt', encoding='utf-8') as stream:
        episodes = [e for e in json.load(stream)['episodes']
                    if e['instruction']['language'] in ('en-US', 'en-IN')]
    if Counter(e['instruction']['language'] for e in episodes) != {'en-US': 1223, 'en-IN': 2446}:
        raise ValueError('Unexpected English guide episode set')
    for scene in {Path(e['scene_id']).stem for e in episodes}:
        mesh = audit_path.parent / f'{scene}.navmesh'
        if hashlib.sha256(mesh.read_bytes()).hexdigest() != audit['scenes'][scene]['navmesh_sha256']:
            raise ValueError(f'Navigation mesh SHA mismatch: {scene}')
    print(json.dumps({'passed': True, 'english_episodes': len(episodes),
                      'scenes': len(audit['scenes']), 'collision_body': [0.88, 0.18]}))


if __name__ == '__main__':
    main()

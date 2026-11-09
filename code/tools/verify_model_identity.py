"""Verify exact tensors of the released final evaluation model."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from qwenvl.model_identity import verify_model_identity, verify_training_backbone


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True)
    parser.add_argument('--checkpoint')
    parser.add_argument('--base-only', action='store_true',
                        help='Verify only the author-released backbone SHA256')
    parser.add_argument('--trained-base', action='store_true',
                        help='Validate an independently trained full backbone by architecture and tensor structure, without author hashes')
    parser.add_argument('--stage', choices=['stage3-epoch1'], default='stage3-epoch1')
    parser.add_argument('--force', action='store_true', help='Rehash even if size/mtime cache matches')
    args = parser.parse_args()
    if sum((args.base_only, args.trained_base, bool(args.checkpoint))) != 1:
        parser.error('Choose one of --checkpoint PATH, --base-only, or --trained-base')
    if args.trained_base:
        verified = verify_training_backbone(args.base)
        source = 'independently-trained architecture validation; not paper-model identity'
    else:
        verified = verify_model_identity(args.base, args.checkpoint, args.stage, args.force)
        source = 'released tensor SHA256'
    print(json.dumps({'source': source, 'verified': verified}, indent=2))


if __name__ == '__main__':
    main()

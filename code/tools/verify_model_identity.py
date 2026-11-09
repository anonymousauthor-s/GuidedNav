"""Verify exact base and selected-stage tensors before a distributed launch."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from qwenvl.model_identity import verify_model_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--stage', choices=['final-2461', 'initialization-6656'], default='final-2461')
    parser.add_argument('--force', action='store_true', help='Rehash even if size/mtime cache matches')
    args = parser.parse_args()
    print(json.dumps({'verified': verify_model_identity(args.base, args.checkpoint, args.stage, args.force)}, indent=2))


if __name__ == '__main__':
    main()

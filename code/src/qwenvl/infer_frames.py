"""Infer a four-action chunk from chronological RGB observations."""
import argparse
import json
from pathlib import Path
from PIL import Image
from qwenvl.inference import create_policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--base', required=True)
    parser.add_argument('--instruction', required=True)
    parser.add_argument('--frames-list', required=True,
                        help='UTF-8 text: one chronological RGB path per line, current frame last; relative to this list')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--attention', choices=['sdpa', 'flash_attention_2'], default='sdpa')
    parser.add_argument('--weights-source', choices=['released', 'trained'], default='released',
                        help='Use trained for your own Stage 3 output; released verifies reference weight hashes')
    args = parser.parse_args()
    frame_list = Path(args.frames_list).resolve()
    paths = [frame_list.parent / line.strip() for line in
             frame_list.read_text(encoding='utf-8').splitlines() if line.strip()]
    if not paths:
        parser.error('The chronological frame list must contain at least the current image')
    frames = []
    for path in paths:
        with Image.open(path) as image:
            frames.append(image.convert('RGB'))
    policy = create_policy(args.checkpoint, args.base, args.device, args.attention, args.weights_source)
    text = policy._generate_chunk(args.instruction, frames)
    print(json.dumps({'raw_text': text, 'actions': policy.parse_actions(text)[:4],
                      'action_mapping': {'0': 'STOP', '1': 'FORWARD_0.25m',
                                         '2': 'LEFT_15deg', '3': 'RIGHT_15deg'}},
                     ensure_ascii=False))


if __name__ == '__main__':
    main()

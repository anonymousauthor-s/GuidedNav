"""Download the official VGGT teacher and verify the exact training payload hash."""
import argparse
import hashlib
from pathlib import Path
import urllib.request

URL = 'https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt'
EXPECTED_SHA256 = 'd15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0'
EXPECTED_BYTES = 5026874952


def check(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 ** 2), b''):
            digest.update(block)
    if path.stat().st_size != EXPECTED_BYTES or digest.hexdigest() != EXPECTED_SHA256:
        raise ValueError('Teacher size or SHA256 differs from the final-stage teacher')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='../assets/vggt/vggt.pt')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    destination = Path(args.output).resolve()
    if destination.is_file():
        check(destination)
    elif args.check_only:
        raise FileNotFoundError(destination)
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_name(destination.name + '.partial')
        with urllib.request.urlopen(URL) as response, partial.open('wb') as stream:
            while True:
                block = response.read(8 * 1024 ** 2)
                if not block:
                    break
                stream.write(block)
        check(partial)
        partial.replace(destination)
    print(f'Verified VGGT teacher: {destination}')


if __name__ == '__main__':
    main()

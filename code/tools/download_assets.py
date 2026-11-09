#!/usr/bin/env python3
"""Download, verify and install the public GuidedNav model or Stage 3 data assets.

Python 3.9+ standard library only. No GitHub account or access token is needed.
Interrupted downloads resume; rerunning verifies and reuses completed files.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import time
import urllib.error
import urllib.request

BLOCK = 8 * 1024 * 1024


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(BLOCK), b''):
            h.update(block)
    return h.hexdigest()


def matches(path, row):
    return path.is_file() and path.stat().st_size == row['bytes'] and sha(path) == row['sha256']


def destination(root, name):
    relative = PurePosixPath(name)
    if (relative.is_absolute() or not relative.parts or
            any(p in ('.', '..') or ':' in p or '\\' in p for p in relative.parts)):
        raise ValueError('Unsafe asset path: ' + name)
    target = root.joinpath(*relative.parts)
    for parent in (target, *target.parents):
        if parent == root.parent:
            break
        if parent.is_symlink():
            raise ValueError('Refusing a symlink destination: ' + str(parent))
    if root not in target.resolve().parents:
        raise ValueError('Asset escapes the output directory')
    return target


def download(row, cache):
    target = destination(cache, row['name'])
    if target.exists():
        if matches(target, row):
            return target
        raise ValueError('Existing cached asset failed SHA256; move it aside: ' + str(target))
    partial = target.with_name(target.name + '.part')
    if partial.is_symlink():
        raise ValueError('Refusing a symlink partial file')
    for attempt in range(3):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset == row['bytes']:
            if sha(partial) != row['sha256']:
                partial.unlink()
                raise ValueError('Downloaded asset failed SHA256: ' + row['name'] + '; rerun to retry')
            partial.replace(target)
            return target
        if offset > row['bytes']:
            raise ValueError('Oversized partial download; move it aside: ' + str(partial))
        headers = {'User-Agent': 'GuidedNav-public-downloader', 'Accept-Encoding': 'identity'}
        if offset:
            headers['Range'] = 'bytes=' + str(offset) + '-'
        try:
            request = urllib.request.Request(row['url'], headers=headers)
            with urllib.request.urlopen(request, timeout=120) as response:
                if response.status == 206:
                    expected = 'bytes ' + str(offset) + '-'
                    if not response.headers.get('Content-Range', '').startswith(expected):
                        raise ValueError('Server returned the wrong download range')
                elif response.status == 200:
                    offset = 0
                else:
                    raise ValueError('Unexpected download response')
                print('Downloading ' + row['name'] + ' from byte ' + str(offset), flush=True)
                last = time.monotonic()
                with partial.open('ab' if offset else 'wb') as out:
                    while True:
                        block = response.read(BLOCK)
                        if not block:
                            break
                        offset += len(block)
                        if offset > row['bytes']:
                            raise ValueError('Server sent more bytes than the manifest permits')
                        out.write(block)
                        if time.monotonic() - last > 15:
                            print('  {:.1f}%'.format(100 * offset / row['bytes']), flush=True)
                            last = time.monotonic()
            if matches(partial, row):
                partial.replace(target)
                return target
            if partial.stat().st_size == row['bytes']:
                partial.unlink()
                raise ValueError('Downloaded asset failed SHA256: ' + row['name'])
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            if attempt == 2:
                raise RuntimeError('Download interrupted; rerun the same command to resume: ' + row['name']) from error
        time.sleep(2 * (attempt + 1))
    raise RuntimeError('Incomplete asset; rerun to resume: ' + row['name'])


def extract(archive, root, allowed):
    count = 0
    with tarfile.open(archive, 'r|gz') as tar:
        for member in tar:
            if not member.isfile():
                raise ValueError('Only regular files are allowed in release archives')
            if PurePosixPath(member.name).parts[0] not in allowed:
                raise ValueError('Unexpected archive root: ' + member.name)
            target = destination(root, member.name)
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + '.guidednav-part')
            if temporary.is_symlink():
                raise ValueError('Refusing a symlink temporary file')
            h = hashlib.sha256()
            with tar.extractfile(member) as source, temporary.open('wb') as out:
                for block in iter(lambda: source.read(BLOCK), b''):
                    h.update(block)
                    out.write(block)
            if target.exists():
                if target.is_file() and target.stat().st_size == member.size and sha(target) == h.hexdigest():
                    temporary.unlink()
                else:
                    temporary.unlink()
                    raise FileExistsError('Refusing to overwrite a different existing file: ' + str(target))
            else:
                temporary.replace(target)
            count += 1
    print('Installed {} verified files from {}'.format(count, archive.name), flush=True)


def install(manifest, group, root, keep_cache=False):
    if manifest.get('schema') != 'guidednav_downloads_v1':
        raise ValueError('Unsupported release manifest')
    root.mkdir(parents=True, exist_ok=True)
    root = root.resolve()
    cache = destination(root, '.guidednav-downloads')
    cache.mkdir(exist_ok=True)
    assets = {a['name']: a for a in manifest['assets']}
    selected = {a['name'] for a in manifest['assets'] if group == 'all' or group in a['groups']}
    for row in manifest['files']:
        if not set(row['parts']) <= selected:
            continue
        target = destination(root, row['path'])
        if target.exists():
            if matches(target, row):
                print('Verified ' + row['path'], flush=True)
                continue
            raise ValueError('Existing model file differs: ' + str(target))
        parts = [download(assets[name], cache) for name in row['parts']]
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + '.guidednav-part')
        if temporary.is_symlink():
            raise ValueError('Refusing a symlink temporary file')
        with temporary.open('wb') as out:
            for part in parts:
                with part.open('rb') as source:
                    shutil.copyfileobj(source, out, BLOCK)
        if not matches(temporary, row):
            raise ValueError('Reassembled model failed SHA256: ' + row['path'])
        temporary.replace(target)
        if not keep_cache:
            for part in parts:
                part.unlink()
        print('Installed ' + row['path'], flush=True)
    for row in manifest['archives']:
        if row['asset'] not in selected:
            continue
        archive = download(assets[row['asset']], cache)
        allowed = {'weights'} if row['root'] == 'weights' else {'annotations', 'data', 'prompt-metadata', 'DATA_TERMS.md'}
        extract(archive, root, allowed)
        # Keep the small archives for fast, fully verified reruns.
    print('Done. Original scenes/instructions and the training-only VGGT teacher are obtained separately.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group', choices=('inference', 'training', 'data', 'all'), default='inference')
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[2], help='Repository root, not code/')
    parser.add_argument('--manifest', type=Path, default=Path(__file__).resolve().parents[1] / 'config/download-manifest.json')
    parser.add_argument('--keep-cache', action='store_true', help='Keep transport parts after tensor reassembly')
    args = parser.parse_args()
    install(json.loads(args.manifest.read_text(encoding='utf-8')), args.group, args.output, args.keep_cache)


if __name__ == '__main__':
    main()

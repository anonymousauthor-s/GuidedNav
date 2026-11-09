"""Fail-closed tensor-payload identity checks, with a local size/mtime hash cache."""
import hashlib
import json
import os
from pathlib import Path
import tempfile


def verify_model_identity(base, checkpoint=None, stage='stage3-epoch1', force=False):
    """Verify the released base, plus a released adapter when one is supplied.

    A user-trained adapter has different tensor hashes. Its loader explicitly
    requests base-only verification and separately validates its architecture,
    recipe, adapter tensors and temporal state.
    """
    identity_path = Path(__file__).resolve().parents[2] / 'config' / 'model_identity.json'
    identity = json.loads(identity_path.read_text())
    cache_path = Path(os.environ.get('GUIDEDNAV_HASH_CACHE',
                      str(Path(tempfile.gettempdir()) / 'guidednav-model-hashes-v1.json')))
    try:
        cache = json.loads(cache_path.read_text())
    except (OSError, ValueError):
        cache = {}
    verified = []
    payloads = [(Path(base), 'navigation-base')]
    if checkpoint is not None:
        payloads.append((Path(checkpoint), stage))
    for folder, name in payloads:
        for filename, expected in identity[name].items():
            path = (folder / filename).resolve()
            stat = path.stat()
            if stat.st_size != expected['bytes']:
                raise ValueError(f'Incorrect {name} payload size: {filename}')
            record = {'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns,
                      'sha256': expected['sha256']}
            if force or cache.get(str(path)) != record:
                digest = hashlib.sha256()
                with path.open('rb') as stream:
                    for block in iter(lambda: stream.read(8 * 1024 ** 2), b''):
                        digest.update(block)
                if digest.hexdigest() != expected['sha256']:
                    raise ValueError(f'Incorrect {name} tensor SHA256: {filename}')
                after = path.stat()
                if (after.st_size, after.st_mtime_ns) != (stat.st_size, stat.st_mtime_ns):
                    raise ValueError(f'Payload changed while verifying: {filename}')
                cache[str(path)] = record
            verified.append(f'{name}/{filename}')
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        partial = cache_path.with_name(cache_path.name + f'.{os.getpid()}.partial')
        partial.write_text(json.dumps(cache, indent=2) + '\n')
        partial.replace(cache_path)
    except OSError:
        # A read-only cache location may slow future loads; verification above
        # still succeeded. Never skip a payload check because caching failed.
        pass
    return verified

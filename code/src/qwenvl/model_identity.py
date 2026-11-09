"""Released-model hash verification and independent training-backbone validation."""
import hashlib
import json
import os
import math
import struct
from pathlib import Path
import tempfile


def verify_model_identity(base, checkpoint=None, stage='stage3-epoch1', force=False):
    """Verify the released base, plus a released adapter when one is supplied.

    This function authenticates the released evaluation model only. Independently
    trained full backbones use verify_training_backbone instead; no author
    weight hash is required for user-produced training outputs.
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


def verify_training_backbone(base):
    """Validate a complete independently trained Qwen3-VL-8B full model.

    Compare architecture, exact tensor names/shapes, shard sizes and index
    mappings without comparing tensor values or author model hashes. A normal
    strict Transformers load still runs afterwards. No model tensor is loaded
    into memory by this metadata check.
    """
    base = Path(base).resolve()
    architecture_path = Path(__file__).resolve().parents[2] / 'config' / 'backbone_architecture.json'
    expected = json.loads(architecture_path.read_text(encoding='utf-8'))
    config = json.loads((base / 'config.json').read_text(encoding='utf-8'))
    if config.get('model_type') != expected['model_type']:
        raise ValueError('Training backbone must be Qwen3-VL-8B')
    for section in ('text_config', 'vision_config'):
        actual = config.get(section, {})
        for key, value in expected[section].items():
            if actual.get(key) != value:
                raise ValueError('Incorrect training backbone architecture: ' + section + '.' + key)
    for filename in ('preprocessor_config.json', 'tokenizer_config.json', 'tokenizer.json'):
        if not (base / filename).is_file():
            raise FileNotFoundError(base / filename)
    index_path = base / 'model.safetensors.index.json'
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding='utf-8'))
        weight_map = index.get('weight_map', {})
        if not weight_map or set(weight_map) != set(expected['tensor_shapes']):
            raise ValueError('Training backbone index has incomplete or unexpected tensors')
        shard_names = sorted(set(weight_map.values()))
    elif (base / 'model.safetensors').is_file():
        weight_map = None
        shard_names = ['model.safetensors']
    else:
        raise FileNotFoundError('Expected a full safetensors backbone, not a LoRA-only adapter: ' + str(base))
    observed = {}
    dtype_bytes = {'BF16': 2, 'F16': 2, 'F32': 4}
    for name in shard_names:
        if not isinstance(name, str) or Path(name).name != name or ':' in name or chr(92) in name:
            raise ValueError('Unsafe backbone shard path')
        path = base / name
        with path.open('rb') as stream:
            prefix = stream.read(8)
            if len(prefix) != 8:
                raise ValueError('Truncated safetensors shard: ' + name)
            header_size = struct.unpack('<Q', prefix)[0]
            if header_size <= 0 or header_size > 100 * 1024 ** 2:
                raise ValueError('Invalid safetensors header length: ' + name)
            raw_header = stream.read(header_size)
            if len(raw_header) != header_size:
                raise ValueError('Truncated safetensors header: ' + name)
            header = json.loads(raw_header)
        payload_size = path.stat().st_size - 8 - header_size
        spans = []
        for tensor, spec in header.items():
            if tensor == '__metadata__':
                continue
            if tensor in observed or tensor not in expected['tensor_shapes']:
                raise ValueError('Duplicate or unexpected backbone tensor: ' + tensor)
            if weight_map is not None and weight_map.get(tensor) != name:
                raise ValueError('Backbone shard/index mapping differs: ' + tensor)
            if spec.get('shape') != expected['tensor_shapes'][tensor]:
                raise ValueError('Incorrect backbone tensor shape: ' + tensor)
            width = dtype_bytes.get(spec.get('dtype'))
            offsets = spec.get('data_offsets', [])
            if (width is None or len(offsets) != 2 or
                    any(not isinstance(v, int) for v in offsets) or
                    not 0 <= offsets[0] <= offsets[1] <= payload_size or
                    offsets[1] - offsets[0] != math.prod(spec['shape']) * width):
                raise ValueError('Invalid backbone tensor payload: ' + tensor)
            spans.append(tuple(offsets))
            observed[tensor] = spec['shape']
        cursor = 0
        for start, end in sorted(spans):
            if start != cursor:
                raise ValueError('Gapped or overlapping safetensors data: ' + name)
            cursor = end
        if cursor != payload_size:
            raise ValueError('Unindexed or truncated safetensors data: ' + name)
    if observed != expected['tensor_shapes']:
        raise ValueError('Incomplete training backbone tensor inventory')
    return [str(base / name) for name in shard_names]

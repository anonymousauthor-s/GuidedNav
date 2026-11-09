"""Print versions and check the APIs required by training or frame inference."""
import argparse
import importlib.metadata
import json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training', action='store_true')
    args = parser.parse_args()
    names = ['torch', 'transformers', 'peft', 'accelerate', 'safetensors', 'numpy', 'Pillow']
    if args.training:
        names += ['deepspeed', 'flash-attn', 'av']
    versions = {}
    for name in names:
        versions[name] = importlib.metadata.version(name)
    import torch
    from transformers import Qwen3VLForConditionalGeneration
    from transformers.models.qwen3_vl.modeling_qwen3_vl import rotate_half
    from peft import PeftModel
    if not torch.cuda.is_available():
        raise RuntimeError('A CUDA GPU is required for the released BF16 recipe')
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError('This GPU/runtime does not support BF16')
    print(json.dumps({'versions': versions, 'cuda': torch.version.cuda,
                      'gpu': torch.cuda.get_device_name(), 'required_apis_imported': True}, indent=2))


if __name__ == '__main__':
    main()

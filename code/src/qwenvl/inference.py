"""Strict loading for the selected final adapter and its learned temporal gate."""
import json
from pathlib import Path
import torch
from transformers import AutoProcessor, AutoTokenizer, Qwen3VLForConditionalGeneration
from qwenvl.temporal_token_merge import TemporalMergeConfig, TemporalResidualGate, parse_temporal_decay


def validate_final_recipe(checkpoint):
    recipe = json.loads((Path(checkpoint) / 'experiment_recipe.json').read_text())
    expected = dict(current_merge_enable=True, current_window_size=4, current_merge_ratio=0.25,
                    current_temporal_decay='0.2,0.35,0.5', current_anchor_weight=1.0,
                    current_spatial_bias=0.05, current_fusion_mode='gated_residual',
                    current_matching_mode='mutual', current_gate_init=0.05, current_gate_max=0.2)
    temporal = recipe.get('temporal_spatial_arguments', {})
    if (recipe.get('recipe') != 'gated_spatial_e6'
            or recipe.get('full_variant') != 'current_sf_native_so'
            or any(temporal.get(key) != value for key, value in expected.items())):
        raise ValueError('Checkpoint recipe differs from the selected final model')
    return recipe


def load_final_model(checkpoint, base, device, min_pixels=12544, max_pixels=200704,
                     attention='sdpa'):
    from qwenvl.model_identity import verify_model_identity
    verify_model_identity(base, checkpoint, 'final-2461')
    validate_final_recipe(checkpoint)
    checkpoint, base = Path(checkpoint), Path(base)
    for filename in ('adapter_model.safetensors', 'adapter_config.json',
                     'experiment_recipe.json', 'temporal_spatial_projectors.pt',
                     'preprocessor_config.json', 'tokenizer_config.json'):
        if not (checkpoint / filename).is_file():
            raise FileNotFoundError(checkpoint / filename)
    config = json.loads((checkpoint / 'adapter_config.json').read_text())
    if config['r'] != 32 or config['lora_alpha'] != 16:
        raise ValueError('Expected final GuidedNav rank-32, alpha-16 LoRA')
    processor = AutoProcessor.from_pretrained(checkpoint, local_files_only=True)
    base_tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    if processor.tokenizer.get_vocab() != base_tokenizer.get_vocab():
        raise ValueError('Checkpoint/base tokenizer vocabularies differ')
    model, loading = Qwen3VLForConditionalGeneration.from_pretrained(
        base, dtype=torch.bfloat16, attn_implementation=attention,
        low_cpu_mem_usage=True, local_files_only=True, output_loading_info=True)
    if model.config.model_type != 'qwen3_vl' or any(loading.get(key) for key in
            ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
        raise RuntimeError(f'Incomplete or incorrect base model: {loading}')
    try:
        import peft.tuners.lora.torchao as peft_torchao
        peft_torchao.is_torchao_available = lambda: False
    except ImportError:
        pass
    import peft.utils.save_and_load as peft_io
    original = getattr(peft_io, '_maybe_shard_state_dict_for_tp', None)
    if original is not None:
        def shard_if_tensor_parallel(model, state_dict, adapter_name):
            if any(getattr(m, '_hf_tp_plan', None) is not None and
                   getattr(m, '_hf_device_mesh', None) is not None for m in model.modules()):
                return original(model, state_dict, adapter_name)
        peft_io._maybe_shard_state_dict_for_tp = shard_if_tensor_parallel
    from peft import PeftModel, get_peft_model_state_dict
    from safetensors.torch import load_file
    model = PeftModel.from_pretrained(model, checkpoint, local_files_only=True)
    stored = load_file(str(checkpoint / 'adapter_model.safetensors'))
    loaded = get_peft_model_state_dict(model)
    if set(stored) != set(loaded) or not all(
            torch.equal(stored[k], loaded[k].detach().cpu()) for k in stored):
        raise RuntimeError('Loaded adapter tensors differ from the checkpoint')
    del stored, loaded
    model = model.merge_and_unload()
    image_processor = processor.image_processor
    image_processor.min_pixels, image_processor.max_pixels = min_pixels, max_pixels
    if isinstance(getattr(image_processor, 'size', None), dict):
        image_processor.size.update(shortest_edge=min_pixels, longest_edge=max_pixels)
    model.to(device).eval()
    model.config.use_cache = True
    return model, processor, processor.tokenizer


def create_policy(checkpoint, base, device='cuda', attention='sdpa'):
    from collections import OrderedDict
    from qwenvl.reference_policy import ReferencePolicy
    checkpoint = Path(checkpoint)
    recipe = json.loads((checkpoint / 'experiment_recipe.json').read_text())
    if recipe.get('full_variant') != 'current_sf_native_so':
        raise ValueError('This release supports the final native S/O recipe only')
    temporal = recipe['temporal_spatial_arguments']
    if (not temporal['current_merge_enable'] or temporal['current_fusion_mode'] != 'gated_residual'
            or temporal['current_matching_mode'] != 'mutual'):
        raise ValueError('Incorrect temporal recipe')
    decay = parse_temporal_decay(temporal['current_temporal_decay'])
    if len(decay) != temporal['current_window_size'] - 1:
        raise ValueError('Temporal decay/window mismatch')
    gate = TemporalResidualGate(initial_gate=temporal['current_gate_init'],
                                max_gate=temporal['current_gate_max'])
    auxiliary = torch.load(checkpoint / 'temporal_spatial_projectors.pt',
                           map_location='cpu', weights_only=True)
    if set(auxiliary) != {'sf8_sf_projector', 'delta_sf_projector', 'temporal_residual_gate'}:
        raise ValueError('Incomplete auxiliary state')
    gate.load_state_dict(auxiliary['temporal_residual_gate'], strict=True)
    gate.to(device=device, dtype=torch.float32).eval()
    if not torch.isfinite(gate()):
        raise ValueError('Nonfinite temporal gate')
    model, processor, tokenizer = load_final_model(checkpoint, base, device, attention=attention)
    policy = ReferencePolicy()
    settings = dict(model=model, processor=processor, tokenizer=tokenizer,
                    device=torch.device(device), num_history=8,
                    current_window_size=temporal['current_window_size'],
                    temporal_merge_config=TemporalMergeConfig(
                        merge_ratio=temporal['current_merge_ratio'],
                        spatial_bias=temporal['current_spatial_bias'],
                        anchor_weight=temporal['current_anchor_weight'],
                        temporal_decay=tuple(reversed(decay)),
                        fusion_mode=temporal['current_fusion_mode'],
                        matching_mode=temporal['current_matching_mode']),
                    temporal_residual_gate=gate, scob_query_tokens=[], future_query_tokens=[],
                    conjunction='you can see ', max_new_tokens=64,
                    actions2idx=OrderedDict({'STOP': [0], '↑': [1], '←': [2], '→': [3]}))
    for name, value in settings.items():
        setattr(policy, name, value)
    return policy

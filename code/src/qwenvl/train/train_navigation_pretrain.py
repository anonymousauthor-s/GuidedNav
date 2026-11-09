"""Paper Stages 1 and 2: full-parameter navigation and visual instruction tuning.

Stage 1 consumes expert navigation + VQA; Stage 2 consumes DAgger + VQA.
The prepared manifest freezes source versions and sample selections before launch.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import torch
import torch.distributed as dist
import transformers
from transformers import AutoConfig, AutoProcessor, Qwen3VLForConditionalGeneration, Trainer

from qwenvl.data.data_processor import update_processor_pixels
from qwenvl.data.early_stage_data import EarlyStageVLNDataset, EarlyStageVQADataset
from qwenvl.data.vln_unified_dataset import VLNActionDataCollator
from qwenvl.train.argument import ModelArguments, DataArguments, TrainingArguments


@dataclass
class NavigationStageArguments:
    stage: int = 1
    training_manifest: str = ''
    resume_checkpoint: str = ''
    smoke: bool = False


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_training_manifest(path, stage):
    manifest_path = Path(path)
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('format') != 'guidednav_navigation_training_v1' or not manifest.get('prepared'):
        raise ValueError('Run tools/prepare_training_manifest.py before training')
    if manifest.get('stage') != stage or manifest.get('seed') != 42:
        raise ValueError('Manifest stage or seed differs from the paper recipe')
    sources = manifest.get('sources', [])
    expected_role = 'expert_navigation' if stage == 1 else 'dagger_navigation'
    if not sources or {source.get('role') for source in sources} != {expected_role, 'visual_qa'}:
        raise ValueError('Stage 1 requires experts + VQA; Stage 2 requires DAgger + VQA')
    names = set()
    for source in sources:
        if source.get('name') in names:
            raise ValueError('Dataset names must be unique')
        names.add(source.get('name'))
        kind = source.get('type')
        if kind not in ('vqa', 'vln_unified'):
            raise ValueError('Unsupported manifest source type')
        if (kind == 'vqa') != (source.get('role') == 'visual_qa'):
            raise ValueError('Dataset type and role disagree')
        if source.get('sampling_rate') != 1.0 or int(source.get('selected_rows', 0)) <= 0:
            raise ValueError('Every source must have a nonempty frozen selection')
        data_file = source['annotation_path'] if kind == 'vqa' else source['samples_path']
        if not Path(data_file).is_absolute() or not Path(data_file).is_file():
            raise ValueError('Prepared data paths must be existing absolute files')
        if sha256_file(data_file) != source['sha256']:
            raise ValueError('Prepared dataset hash differs: ' + source['name'])
    return manifest


def parameter_group(name):
    """Projector includes the visual merger and deepstack merger layers."""
    if 'visual.' in name:
        return 'projector' if 'merger' in name else 'vision'
    return 'language'


class NavigationPretrainTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        inputs = dict(inputs)
        for key in ('task_type', 'subtask_token_mask', 'phase_token_mask'):
            inputs.pop(key, None)
        return super().compute_loss(model, inputs, return_outputs=return_outputs,
                                    num_items_in_batch=num_items_in_batch)

    def create_optimizer(self):
        if self.optimizer is None:
            decay_names = set(self.get_decay_parameter_names(self.model))
            learning_rates = {'language': self.args.learning_rate,
                              'vision': self.args.vision_tower_lr,
                              'projector': self.args.mm_projector_lr}
            groups = []
            for kind, learning_rate in learning_rates.items():
                for decay in (True, False):
                    parameters = [parameter for name, parameter in self.model.named_parameters()
                                  if parameter.requires_grad and parameter_group(name) == kind
                                  and (name in decay_names) == decay]
                    if parameters:
                        groups.append({'params': parameters, 'lr': learning_rate,
                                       'weight_decay': self.args.weight_decay if decay else 0.0})
            optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args, self.model)
            self.optimizer = optimizer_cls(groups, **optimizer_kwargs)
        return self.optimizer


def build_data(processor, data_args, manifest):
    datasets = []
    for source in manifest['sources']:
        if source['type'] == 'vqa':
            dataset = EarlyStageVQADataset(processor, data_args, dataset_configs=[source])
        else:
            dataset = EarlyStageVLNDataset(processor=processor, data_args=data_args,
                                       vln_config=source, label_mode='action')
        if len(dataset) != source['selected_rows']:
            raise ValueError('Loaded sample count differs: ' + source['name'])
        datasets.append(dataset)
    return torch.utils.data.ConcatDataset(datasets)


def main():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments,
                                            NavigationStageArguments))
    model_args, data_args, args, run = parser.parse_args_into_dataclasses()
    if run.stage not in (1, 2):
        raise ValueError('Use --stage 1 or 2')
    if not run.training_manifest:
        raise ValueError('--training_manifest is required')
    expected_lr = 2e-5 if run.stage == 1 else 1e-5
    if args.lora_enable or not all((model_args.tune_mm_llm, model_args.tune_mm_mlp,
                                    model_args.tune_mm_vision)):
        raise ValueError('Stages 1 and 2 update every policy parameter without LoRA')
    if not run.smoke:
        if args.max_steps > 0 or args.num_train_epochs != 1:
            raise ValueError('The paper recipe uses one complete epoch')
        batch = args.world_size * args.per_device_train_batch_size * args.gradient_accumulation_steps
        if batch != 256 or args.learning_rate != expected_lr:
            raise ValueError('Use global batch 256 and the paper stage learning rate')
        if args.mm_projector_lr != 5e-6 or args.vision_tower_lr != 5e-6:
            raise ValueError('Vision and projector learning rates must both be 5e-6')
    if args.seed != 42 or args.data_seed != 42 or data_args.data_flatten or data_args.data_packing:
        raise ValueError('Use seed/data_seed 42 and non-packed mixed-task batches')
    transformers.set_seed(42)
    args.remove_unused_columns = False
    args.gradient_checkpointing_kwargs = {'use_reentrant': False}
    manifest = load_training_manifest(run.training_manifest, run.stage)
    from qwenvl.model_identity import verify_training_backbone
    verify_training_backbone(model_args.model_name_or_path)
    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    if config.model_type != 'qwen3_vl':
        raise ValueError('The paper uses Qwen3-VL-8B-Instruct')
    if run.stage == 2:
        previous = Path(model_args.model_name_or_path) / 'training_recipe.json'
        if not previous.is_file() or (json.loads(previous.read_text(encoding='utf-8')).get('stage') != 1 or
                                      json.loads(previous.read_text(encoding='utf-8')).get('smoke')):
            raise ValueError('Stage 2 must start from your completed Stage 1 export')
    output = Path(args.output_dir)
    error = None
    if args.process_index == 0:
        if output.exists() and any(output.iterdir()) and not run.resume_checkpoint:
            error = 'Choose an empty output directory or set resume_checkpoint'
    if dist.is_initialized():
        errors = [error]
        dist.broadcast_object_list(errors, src=0)
        error = errors[0]
    if error:
        raise FileExistsError(error)
    processor = AutoProcessor.from_pretrained(model_args.model_name_or_path)
    processor.tokenizer.model_max_length = args.model_max_length
    processor.tokenizer.padding_side = 'right'
    update_processor_pixels(processor, data_args)
    data_args.model_type = 'qwen3vl'
    dataset = build_data(processor, data_args, manifest)
    model, loading_info = Qwen3VLForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path, dtype=torch.bfloat16 if args.bf16 else None,
        attn_implementation='flash_attention_2', output_loading_info=True)
    if any(loading_info.get(key) for key in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
        raise ValueError('Full-model weights did not load exactly: ' + str(loading_info))
    model.requires_grad_(True)
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.enable_input_require_grads()
    recipe = {'stage': run.stage, 'epochs': 1, 'global_batch_size': 256,
              'policy_update': 'full_parameters', 'source_model': model_args.model_name_or_path,
              'manifest_sha256': sha256_file(run.training_manifest),
              'language_lr': args.learning_rate, 'vision_lr': args.vision_tower_lr,
              'projector_lr': args.mm_projector_lr, 'seed': 42, 'smoke': run.smoke}
    if run.resume_checkpoint:
        previous_recipe = output / 'training_recipe.json'
        if not previous_recipe.is_file() or json.loads(previous_recipe.read_text(encoding='utf-8')) != recipe:
            raise ValueError('Resume model, manifest or optimization recipe differs')
    if args.should_save:
        output.mkdir(parents=True, exist_ok=True)
        (output / 'training_recipe.json').write_text(json.dumps(recipe, indent=2) + '\n', encoding='utf-8')
    if dist.is_initialized():
        dist.barrier()
    trainer = NavigationPretrainTrainer(model=model, processing_class=processor.tokenizer,
        args=args, train_dataset=dataset,
        data_collator=VLNActionDataCollator(tokenizer=processor.tokenizer))
    trainer.train(resume_from_checkpoint=run.resume_checkpoint or None)
    trainer.save_state()
    export = output / ('smoke-export' if run.smoke else f'stage{run.stage}-epoch1')
    trainer.save_model(str(export))
    if args.should_save:
        processor.save_pretrained(export)
        (export / 'training_recipe.json').write_text(json.dumps(recipe, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()

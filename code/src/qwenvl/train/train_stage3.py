"""Stage 3 training with spatial, subtask and landmark guidance."""
from dataclasses import dataclass, asdict
import hashlib
import json
import math
import os
from pathlib import Path

import torch
import torch.distributed as dist
import transformers
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, TrainerCallback

from qwenvl.train.argument import ModelArguments, DataArguments, TrainingArguments
from qwenvl.train.train_discrete_action_temporal_spatial import TemporalSpatialArguments, load_vggt_teacher
from qwenvl.train.train_discrete_action_vggt_patch import AlignProjector, _hidden_size
from qwenvl.temporal_token_merge import TemporalResidualGate
from qwenvl.data.full_vln_dataset import load_full_dataset, FullVLNCollator
from qwenvl.train.stage3_trainer import Stage3Trainer


@dataclass
class FullRunArguments:
    full_manifest: str = ''
    full_init: str = ''
    full_resume: str = ''
    full_stop_step: int = 0
    full_smoke: bool = False
    full_async_eval: bool = False


class StageStop(TrainerCallback):
    def __init__(self,stop_step,expected_total):
        self.stop_step,self.expected_total = stop_step,expected_total

    def on_train_begin(self,args,state,control,**kwargs):
        if state.max_steps != self.expected_total:
            raise ValueError(f'Whole-epoch scheduler changed: {state.max_steps} != {self.expected_total}')
        if self.stop_step <= state.global_step or self.stop_step > state.max_steps:
            raise ValueError('Stage stop must lie after resume and within the full epoch')

    def on_step_end(self,args,state,control,**kwargs):
        if state.global_step >= self.stop_step:
            control.should_save = True
            control.should_training_stop = True
        return control


class SmokeGradientAudit(TrainerCallback):
    def __init__(self,model):
        self.sums = {}
        self.handles = []
        for name,parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            group = ('sf_projector' if 'sf8_sf_projector' in name else 'delta_projector' if 'delta_sf_projector' in name
                     else 'gate' if 'temporal_residual_gate' in name else 'vision_lora' if 'visual' in name else 'language_lora')
            def record(gradient,key=group):
                value = gradient.detach().float().square().sum()
                self.sums[key] = self.sums.get(key,0.)+value
                return gradient
            self.handles.append(parameter.register_hook(record))

    def on_train_begin(self,args,state,control,**kwargs):
        torch.cuda.reset_peak_memory_stats()

    def on_train_end(self,args,state,control,**kwargs):
        keys=('sf_projector','delta_projector','gate','vision_lora','language_lora')
        values=torch.stack([torch.as_tensor(self.sums.get(k,0.),device=args.device) for k in keys])
        if dist.is_initialized():dist.all_reduce(values)
        report=dict(zip(keys,values.sqrt().tolist()))
        required = ('sf_projector','delta_projector','gate','vision_lora','language_lora')
        if any(not math.isfinite(report.get(key,float('nan'))) or report.get(key,0.)<=0 for key in required):
            raise RuntimeError('Smoke lacks finite nonzero gradients in all intended branches: '+str(report))
        if args.should_save:Path(args.output_dir,f'gradient_audit_step{state.global_step}.json').write_text(json.dumps(report,indent=2)+'\n')
        memory = {'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                  'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
                  'device':torch.cuda.get_device_name()}
        if args.should_save:Path(args.output_dir,f'memory_audit_step{state.global_step}.json').write_text(json.dumps(memory,indent=2)+'\n')
        for handle in self.handles:
            handle.remove()


class CheckpointedFullTrainer(Stage3Trainer):
    def __init__(self,*args,full_processor,full_recipe,**kwargs):
        super().__init__(*args,**kwargs)
        self.full_processor,self.full_recipe = full_processor,full_recipe

    def _save(self,output_dir=None,state_dict=None):
        super()._save(output_dir,state_dict)
        if self.args.should_save:
            folder = Path(output_dir or self.args.output_dir)
            self.full_processor.save_pretrained(folder)
            (folder/'experiment_recipe.json').write_text(json.dumps(self.full_recipe,indent=2)+'\n')


def main():
    parser = transformers.HfArgumentParser((ModelArguments,DataArguments,TemporalSpatialArguments,TrainingArguments,FullRunArguments))
    model_args,data_args,spatial,args,run = parser.parse_args_into_dataclasses()
    from qwenvl.model_identity import verify_model_identity
    verify_model_identity(os.environ.get('CONTINUE_BASE',model_args.model_name_or_path),
                          run.full_init, 'stage3-initialization')
    if not run.full_init:
        raise ValueError('Stage 3 initialization weights are required')
    if not run.full_manifest:
        raise ValueError('A validated full mixed-source manifest is required')
    if args.per_device_train_batch_size != 1 or not args.lora_enable:
        raise ValueError('Recipe requires LoRA and per-device batch 1')
    if (spatial.current_spatial_weight,spatial.past_spatial_weight,spatial.sf8_spatial_weight,spatial.delta_spatial_weight) != (0.,0.,.5,.1):
        raise ValueError('Use long-horizon weight .5/current-only loss and local weight .1; standalone current/past SF must be disabled')
    if spatial.current_fusion_mode != 'gated_residual' or spatial.current_matching_mode != 'mutual':
        raise ValueError('Stage 3 requires mutual matching and gated residual fusion')
    if not run.full_smoke:
        if args.world_size != 32 or args.gradient_accumulation_steps != 8 or args.max_steps > 0 or args.num_train_epochs != 1:
            raise ValueError('Reference Stage 3 training requires 32 ranks, accumulation 8 and one whole epoch')
        if args.learning_rate != 2e-4 or args.seed != 42 or args.lora_r != 32 or args.lora_alpha != 16:
            raise ValueError('Formal optimizer/LoRA recipe mismatch')
    args.remove_unused_columns = False
    args.dataloader_drop_last = False
    args.gradient_checkpointing_kwargs = {'use_reentrant':False}
    transformers.set_seed(args.seed)
    output = Path(args.output_dir)
    error = None
    if args.process_index == 0:
        try:
            if output.exists() and any(output.iterdir()) and not run.full_resume:
                raise FileExistsError('Fresh full run refuses nonempty output directory')
            output.mkdir(parents=True,exist_ok=True)
        except Exception as exc:
            error = str(exc)
    if dist.is_initialized():
        errors = [error]
        dist.broadcast_object_list(errors,src=0)
        error = errors[0]
    if error:
        raise RuntimeError(error)
    if dist.is_initialized():
        dist.barrier()
    processor = AutoProcessor.from_pretrained(run.full_init,local_files_only=True)
    processor.tokenizer.model_max_length = args.model_max_length
    processor.tokenizer.padding_side = 'right'
    data_args.model_type = 'qwen3vl'
    dataset,manifest = load_full_dataset(processor,data_args,spatial,run.full_manifest,allow_smoke=run.full_smoke)
    if run.full_smoke:
        # Exercise eligible R2R and RxR subtask/landmark supervision.
        from torch.utils.data import Subset
        import numpy as np
        chosen,offset = [],0
        for part in dataset.datasets:
            source = part.source['name']
            eligible = np.flatnonzero(part.arrays['o_eligible'] & (part.arrays['weights']>0))
            indices = np.random.RandomState(42).choice(eligible,min(32,len(eligible)),replace=False)
            chosen.extend(offset+int(i) for i in indices)
            offset += len(part)
        dataset = Subset(dataset,chosen)
    total_steps = args.max_steps if args.max_steps > 0 else math.ceil(math.ceil(len(dataset)/args.world_size)/args.gradient_accumulation_steps)
    stop_step = run.full_stop_step or total_steps
    recipe = {'recipe':'gated_spatial_e6','full_variant':'current_sf_native_so',
              'temporal_spatial_arguments':asdict(spatial),'sf_supervised_frames':'current_only',
              'sf_teacher_context':'original_memory_plus_current','s_weight':.01,'o_weight':.01,
              'aux_warmup_ratio':.03,'full_manifest_sha256':hashlib.sha256(Path(run.full_manifest).read_bytes()).hexdigest(),
              'full_epoch_steps':total_steps,'world_size':args.world_size,
              'gradient_accumulation_steps':args.gradient_accumulation_steps,
              'global_batch_size':args.world_size*args.gradient_accumulation_steps*args.per_device_train_batch_size,
              'lora':{'enabled':True,'r':args.lora_r,'alpha':args.lora_alpha,'dropout':args.lora_dropout,'target_modules':'all-linear'},
              'base_model':model_args.model_name_or_path,'init_checkpoint':run.full_init,'stage':'stage3'}
    if run.full_resume:
        resume = Path(run.full_resume)
        previous = json.loads((resume/'experiment_recipe.json').read_text())
        if previous != recipe:
            raise ValueError('Resume recipe/manifest/schedule differs from the original full run')
        if not (resume/'trainer_state.json').is_file() or not (resume/'temporal_spatial_projectors.pt').is_file():
            raise ValueError('Resume checkpoint lacks trainer state or auxiliary trainable modules')
        if not list(resume.glob('global_step*')) and args.deepspeed:
            raise ValueError('ZeRO resume checkpoint lacks optimizer partitions')
    model = Qwen3VLForConditionalGeneration.from_pretrained(os.environ.get('CONTINUE_BASE',model_args.model_name_or_path),dtype=torch.bfloat16,
                                                           attn_implementation='flash_attention_2',local_files_only=True)
    model.requires_grad_(False)
    try:
        import peft.tuners.lora.torchao as peft_torchao
        peft_torchao.is_torchao_available = lambda:False
    except ImportError:
        pass
    # This runtime's PEFT eagerly imports a newer Transformers TP symbol even
    # for ordinary data-parallel LoRA. Its TP routine is a no-op without a mesh.
    import peft.utils.save_and_load as peft_io
    original_tp_shard=getattr(peft_io,'_maybe_shard_state_dict_for_tp',None)
    def shard_only_if_tensor_parallel(model,state_dict,adapter_name):
        if not any(getattr(m,'_hf_tp_plan',None) is not None and getattr(m,'_hf_device_mesh',None) is not None for m in model.modules()):
            return
        if original_tp_shard is None:
            raise RuntimeError('Tensor parallel checkpoint loading is unsupported')
        return original_tp_shard(model,state_dict,adapter_name)
    peft_io._maybe_shard_state_dict_for_tp=shard_only_if_tensor_parallel
    from peft import PeftModel, get_peft_model_state_dict
    init=Path(run.full_init)
    previous=json.loads((init/'experiment_recipe.json').read_text())
    assert previous['full_variant']=='current_sf_native_so' and previous['lora']==recipe['lora']
    model=PeftModel.from_pretrained(model,init,is_trainable=True)
    hidden = _hidden_size(model)
    for name in ('sf8_sf_projector','delta_sf_projector'):
        setattr(model,name,AlignProjector(hidden,1024).to(dtype=torch.bfloat16))
    model.temporal_residual_gate = TemporalResidualGate(initial_gate=spatial.current_gate_init,max_gate=spatial.current_gate_max)
    saved=torch.load(init/'temporal_spatial_projectors.pt',map_location='cpu',weights_only=True)
    assert set(saved)=={'sf8_sf_projector','delta_sf_projector','temporal_residual_gate'}
    for name,state in saved.items():
        getattr(model,name).load_state_dict(state,strict=True)
    if args.process_index==0:
        from safetensors.torch import load_file
        stored=load_file(str(init/'adapter_model.safetensors'))
        loaded=get_peft_model_state_dict(model)
        assert set(stored)==set(loaded)
        assert all(torch.equal(stored[k],loaded[k].detach().cpu()) for k in stored)
        for name,state in saved.items():
            actual=getattr(model,name).state_dict()
            assert all(torch.equal(state[k],actual[k].detach().cpu()) for k in state)
        (output/'initialization_audit.json').write_text(json.dumps({'exact_adapter_values':True,'exact_auxiliary_values':True,'source':str(init),'new_optimizer':not bool(run.full_resume)},indent=2))
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.enable_input_require_grads()
    if args.process_index == 0:
        trainable = {name:{'shape':list(p.shape),'count':p.numel()} for name,p in model.named_parameters() if p.requires_grad}
        if not any('visual' in name and 'lora_' in name for name in trainable):
            raise ValueError('Visual LoRA is missing')
        if any('lm_head' in name or 'embed_tokens' in name for name in trainable):
            raise ValueError('Unexpected trainable embedding or LM head')
        (output/'trainable_parameters.json').write_text(json.dumps(trainable,indent=2)+'\n')
        (output/'experiment_recipe.json').write_text(json.dumps(recipe,indent=2)+'\n')
    teacher = load_vggt_teacher(spatial,args.device)
    callbacks = [StageStop(stop_step,total_steps)]
    if run.full_async_eval:
        raise ValueError('Use full_async_eval=False for the standalone release')
    if run.full_smoke:
        callbacks.append(SmokeGradientAudit(model))
    trainer = CheckpointedFullTrainer(model=model,processing_class=processor.tokenizer,args=args,
                train_dataset=dataset,data_collator=FullVLNCollator(processor.tokenizer),
                spatial_args=spatial,vggt_teacher=teacher,full_processor=processor,full_recipe=recipe,
                callbacks=callbacks)
    trainer.train(resume_from_checkpoint=run.full_resume or None)
    # All ranks participate in the DeepSpeed save. The epoch-named export is
    # directly usable for inference; trainer checkpoints retain optimizer state.
    completed_epoch = (not run.full_smoke and trainer.state.global_step == total_steps)
    if completed_epoch:
        trainer.save_model(str(output / 'stage3-epoch1'))
    if dist.is_initialized():
        dist.barrier()
    if args.process_index == 0:
        checkpoint = output/f'checkpoint-{trainer.state.global_step}'
        if not checkpoint.is_dir():
            raise RuntimeError('Stage ended without its full checkpoint')
        ready = {'step':trainer.state.global_step,'full_epoch_steps':total_steps,'checkpoint':str(checkpoint),
                 'recipe_sha256':hashlib.sha256((checkpoint/'experiment_recipe.json').read_bytes()).hexdigest()}
        (output/f'STAGE_READY_{trainer.state.global_step}.json').write_text(json.dumps(ready,indent=2)+'\n')


if __name__ == '__main__':
    main()

"""Keep the GuidedNav model; supervise only current SF tokens and add native S/O."""
import math
import torch
import torch.distributed as dist
from qwenvl.train.train_discrete_action_temporal_spatial import TemporalSpatialTrainer
from qwenvl.train.native_attention import NativeAttention


class Stage3Trainer(TemporalSpatialTrainer):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.native = NativeAttention(self.model,layer=27,heads=(0,16))
        self.model_accepts_loss_kwargs = False

    def log(self,logs,*args,**kwargs):
        if 'grad_norm' in logs and not math.isfinite(float(logs['grad_norm'])):
            raise FloatingPointError('Stage 3 training produced a nonfinite gradient norm')
        return super().log(logs,*args,**kwargs)

    def get_batch_samples(self,epoch_iterator,num_batches,device):
        batches,_ = super().get_batch_samples(epoch_iterator,num_batches,device)
        if not batches:
            return batches,None
        counts = torch.tensor([len(batches),sum(int((b['span_weight']>0).sum()) for b in batches),
                               sum(int(b['mask_valid'].sum()) for b in batches)],device=device,dtype=torch.long)
        if dist.is_initialized():
            dist.all_reduce(counts)
        for batch in batches:
            batch['_full_counts'] = counts
            batch['_full_microbatches'] = len(batches)
        return batches,None

    def _alignment_loss(self,model,projector_name,student_parts,target_grids,teacher,vggt_hw):
        if projector_name != 'sf8_sf_projector':
            raise ValueError('Full recipe only permits the joint-context spatial guidance branch')
        # Teacher already processed the original history+current sequence. Slice
        # only its output; historical student tokens never enter this SF loss.
        return super()._alignment_loss(model,projector_name,student_parts[-1:],
                                       target_grids[-1:],teacher[:,-1:],vggt_hw)

    def _pop_additional_inputs(self,inputs):
        super()._pop_additional_inputs(inputs)
        names = ('action_state_position','current_patch_positions','mask_targets','mask_valid',
                 'span_positions','span_target','span_weight','source_indices','_full_counts','_full_microbatches')
        return {name:inputs.pop(name,None) for name in names}

    def _additional_loss(self,model,outputs,metadata):
        del model
        s_valid = bool(metadata['span_weight'][0]>0)
        o_valid = bool(metadata['mask_valid'][0])
        s_loss = o_loss = outputs.loss*0.
        if s_valid or o_valid:
            logits = self.native.row_logits(metadata['action_state_position'])
            if s_valid:
                log_p,_ = self.native.distribution(logits,metadata['span_positions'])
                target = metadata['span_target'][0,:log_p.numel()]
                s_loss = -torch.logsumexp(log_p[target],dim=0)*metadata['span_weight'][0]
            if o_valid:
                log_p,_ = self.native.distribution(logits,metadata['current_patch_positions'])
                target = metadata['mask_targets'][0,:log_p.numel()].float()
                if target.max() <= 0 or target.min() < 0 or target.max() > 1:
                    raise ValueError('Invalid area-pooled object mask')
                # Soft region mass preserves within-object attention freedom.
                o_loss = -torch.logsumexp(log_p + target.clamp_min(1e-30).log(),dim=0)
        counts = metadata['_full_counts']
        if counts is not None:
            s_loss = s_loss * counts[0] / counts[1].clamp_min(1)
            o_loss = o_loss * counts[0] / counts[2].clamp_min(1)
        return .01*(s_loss+o_loss), {'native_s_loss':float(s_loss.detach()),'native_o_loss':float(o_loss.detach()),
                                     'native_s_valid':float(s_valid),'native_o_valid':float(o_valid)}

    def compute_loss(self,model,inputs,return_outputs=False,num_items_in_batch=None):
        inputs = dict(inputs)
        counts = inputs.get('_full_counts')
        microbatches = int(inputs.get('_full_microbatches',1))
        self.native.begin()
        try:
            total,outputs = super().compute_loss(model,inputs,return_outputs=True,num_items_in_batch=None)
        finally:
            self.native.clear()
        warmup_steps = max(1,math.ceil(self.state.max_steps*.03))
        warmup = min(1.,(self.state.global_step+1)/warmup_steps)
        loss = outputs.loss + warmup*(total-outputs.loss)
        if counts is not None:
            loss = loss*(self.args.world_size*microbatches)/counts[0].clamp_min(1)
        if not torch.isfinite(loss):
            raise FloatingPointError('Stage 3 training produced a nonfinite loss')
        self._metric_sums['aux_warmup'] = self._metric_sums.get('aux_warmup',0.)+warmup
        return (loss,outputs) if return_outputs else loss

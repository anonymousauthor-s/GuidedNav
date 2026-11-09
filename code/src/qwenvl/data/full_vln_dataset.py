"""Manifest-pinned action inputs with temporal inputs and training-only S/O."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import ConcatDataset

from qwenvl.data.data_processor import update_processor_pixels
from qwenvl.data.soma_dataset import SOMADataset
from qwenvl.data.native_attention_dataset import NativeCollator, token_span
from qwenvl.train.train_discrete_action_temporal_spatial import TemporalSpatialVLNDataset
from qwenvl.train.train_discrete_action_vggt_patch import _preprocess_vggt_pil_images


class FullSourceDataset(SOMADataset):
    def __init__(self, processor, data_args, *, manifest_dir, source, spatial_args):
        self.source = source
        self.samples_path = source['path']
        stat = Path(self.samples_path).stat()
        if stat.st_size != source['size_bytes'] or stat.st_mtime_ns != source['mtime_ns']:
            raise ValueError('Frozen training source changed: '+self.samples_path)
        self.arrays = {}
        for name, meta in source['arrays'].items():
            path = manifest_dir/meta['path']
            if hashlib.sha256(path.read_bytes()).hexdigest() != meta['sha256']:
                raise ValueError('Mixed training array hash mismatch: '+str(path))
            self.arrays[name] = np.load(path, mmap_mode='r', allow_pickle=False)
        self.offsets = self.arrays['offsets']
        self.source_indices = self.arrays['source_indices']
        self.video_folder, self.mask_root = source['rgb_root'], source['mask_root']
        self.processor = update_processor_pixels(processor,data_args)
        self.tokenizer = processor.tokenizer
        self.merge_size = int(getattr(processor.image_processor,'merge_size',2))
        self.image_token_id = int(self.tokenizer.convert_tokens_to_ids('<|image_pad|>'))
        self.load_masks, self.load_subtasks = bool(source.get('object_masks_enabled',source['name']=='r2r')), False
        self.spatial_args = spatial_args
        self._base_images = self._base_paths = None
        self._last_row = None

    def _read_record(self,index):
        with open(self.samples_path,'rb') as f:
            f.seek(int(self.offsets[index]))
            row = json.loads(f.readline())
        self._last_row = row
        return row

    def _encode(self,record,images,n_memory):
        self._base_images = images
        self._base_paths = record['images']
        return super()._encode(record,images,n_memory)

    def __getitem__(self,index):
        item = super().__getitem__(index)
        raw, images = self._last_row, self._base_images
        auxiliary, ages = TemporalSpatialVLNDataset._load_current_auxiliary(self)
        current_index = int(Path(self._base_paths[-1]).stem)
        expected_ages = [age for age in (3,2,1) if current_index-age >= 1]
        if ages != expected_ages:
            raise ValueError('Missing non-boundary local four-frame RGB frame')
        if auxiliary:
            encoded = self.processor.image_processor(auxiliary,return_tensors='pt')
            item['current_aux_pixel_values'] = encoded['pixel_values']
            item['current_aux_grid_thw'] = encoded['image_grid_thw']
            item['current_aux_ages'] = torch.tensor(ages,dtype=torch.long)
        local = auxiliary + [images[-1]]
        item['current_vggt_images'] = _preprocess_vggt_pil_images(local,self.spatial_args.vggt_preprocess_mode)
        item['current_vggt_count'] = len(local)
        item['sf8_vggt_images'] = _preprocess_vggt_pil_images(images,self.spatial_args.vggt_preprocess_mode)
        item['sf8_vggt_count'] = len(images)
        weight = float(self.arrays['weights'][index])
        positions, target = torch.empty(0,dtype=torch.long), torch.empty(0,dtype=torch.bool)
        if weight > 0:
            prefix = item['input_ids'].reshape(-1)[:int(item['action_state_position'])+1].tolist()
            positions,target = token_span(self.tokenizer,prefix,raw['instruction'],self.arrays['spans'][index])
            if not bool(target.any()):
                raise ValueError('Valid S supervision became an empty token span')
        item.update(span_positions=positions,span_target=target,span_weight=torch.tensor(weight))
        self._base_images = self._base_paths = self._last_row = None
        return item


class FullVLNCollator(NativeCollator):
    def __call__(self,instances):
        if len(instances) != 1:
            raise ValueError('Stage 3 recipe requires microbatch one')
        batch = super().__call__(instances)
        item = instances[0]
        for name in ('current_aux_pixel_values','current_aux_grid_thw','current_aux_ages'):
            if name in item:
                batch[name] = item[name]
        for prefix in ('current','sf8'):
            batch[prefix+'_vggt_images'] = item[prefix+'_vggt_images'].unsqueeze(0)
            batch[prefix+'_vggt_count'] = torch.tensor([item[prefix+'_vggt_count']],dtype=torch.long)
        return batch


def load_full_dataset(processor,data_args,spatial_args,manifest_path,*,allow_smoke=False):
    path = Path(manifest_path)
    manifest = json.loads(path.read_text())
    formal = manifest['format'] == 'full_vln_mixed_v1' and manifest['selected_rows'] == 3262896
    smoke = allow_smoke and manifest['format'] == 'full_vln_mixed_smoke_v1'
    quarter=(manifest['format']=='full_vln_expert25_so_v1' and [x['name'] for x in manifest['sources']]==['r2r','rxr'] and manifest.get('all_rows_annotation_complete') is True and all(x['fraction']==.25 and x['selected_rows']==x['source_rows']//4 for x in manifest['sources']))
    if not formal and not quarter and not smoke:
        raise ValueError('Unexpected full training manifest')
    dataset = ConcatDataset([FullSourceDataset(processor,data_args,manifest_dir=path.parent,
                           source=source,spatial_args=spatial_args) for source in manifest['sources']])
    if len(dataset) != manifest['selected_rows']:
        raise ValueError('Mixed dataset length mismatch')
    return dataset,manifest

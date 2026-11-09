"""Rebuild stage-3 action-training rows from authorized public text and aligned metadata.

RGB is rendered separately from the exact collected agent poses. This tool does
not download, render, relabel, resample, or modify any input artifact.
"""
import argparse
import copy
import json
from pathlib import Path
import shutil
import numpy as np

from verify_annotation_alignment import (CONJUNCTIONS, GLYPHS, SYSTEM_PROMPT,
                                          history, public_index, sha_file, text_sha)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations', required=True)
    parser.add_argument('--r2r-source', required=True)
    parser.add_argument('--rxr-source', required=True)
    parser.add_argument('--trajectories', required=True, help='Contains r2r/traj/ and rxr/traj/')
    parser.add_argument('--pose-manifest', required=True, help='Frozen pose-manifest.json with per-video SHA256')
    parser.add_argument('--prompt-metadata', required=True, help='Contains conjunction manifest.json and uint8 arrays')
    parser.add_argument('--r2r-rgb', required=True)
    parser.add_argument('--rxr-rgb', required=True)
    parser.add_argument('--output', required=True, help='New local training-data directory; must not exist')
    args = parser.parse_args()
    addon = Path(args.annotations).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError('Refusing to overwrite output')
    manifest = json.loads((addon/'manifest.json').read_text(encoding='utf-8'))
    prompts_root = Path(args.prompt_metadata)
    prompt_manifest = json.loads((prompts_root/'manifest.json').read_text(encoding='utf-8'))
    if prompt_manifest['conjunction_order'] != list(CONJUNCTIONS):
        raise ValueError('Prompt conjunction order differs')
    poses = json.loads(Path(args.pose_manifest).read_text(encoding='utf-8'))
    pose_hashes = {(x['source'], x['video']):x['sha256'] for x in poses['files']}
    expected_counts = {'r2r':157465, 'rxr':472323}
    source_rows = {'r2r':629862, 'rxr':1889294}
    result = {'format':'full_vln_expert25_so_v1', 'selected_rows':629788, 'seed':42,
              'all_rows_annotation_complete':True, 'epochs':1, 'sources':[],
              'materialization':{'from':'public upstream text + frozen annotations/poses/actions/prompt IDs',
                  'annotation_manifest_sha256':sha_file(addon/'manifest.json'),
                  'pose_manifest_sha256':sha_file(args.pose_manifest),
                  'prompt_manifest_sha256':sha_file(prompts_root/'manifest.json'),
                  'no_new_sampling_or_labels':True, 'RGB_rendering':'separate'}}
    output.mkdir(parents=True)
    for source in manifest['sources']:
        name = source['name']
        if source['rows'] != expected_counts[name]:
            raise ValueError('Unexpected final-stage selection size')
        original = addon/source['path']
        if sha_file(original) != source['sha256']:
            raise ValueError('Annotation bytes differ from frozen manifest')
        prompt = prompt_manifest['sources'][name]
        prompt_path = prompts_root/prompt['path']
        if (prompt['annotation_jsonl_sha256'] != source['sha256']
                or prompt['source_indices_sha256'] != source['arrays']['source_indices']['sha256']
                or sha_file(prompt_path) != prompt['sha256']):
            raise ValueError('Prompt IDs belong to a different selection')
        conjunctions = np.load(prompt_path, allow_pickle=False)
        if conjunctions.shape != (source['rows'],) or conjunctions.dtype != np.uint8 or (conjunctions>6).any():
            raise ValueError('Invalid prompt-ID array')
        public = public_index(Path(getattr(args,name+'_source')), name)
        target_dir = output/name
        target_dir.mkdir()
        path = target_dir/'samples_perstep.jsonl'
        offsets = np.empty(source['rows'], dtype=np.int64)
        actions_by_video = {}
        with original.open('rb') as incoming, path.open('wb') as target:
            for index,line in enumerate(incoming):
                row = json.loads(line)
                source_id = int(row['fgr_path_id'] if name=='r2r' else row['rxr_instruction_id'])
                key = (source_id,row['instruction_sha256'])
                candidates = public.get(key if name=='r2r' else source_id,[])
                candidates = [x for x in candidates if text_sha(x[0])==row['instruction_sha256']]
                if len(candidates)!=1:
                    raise ValueError('Cannot uniquely match exact upstream instruction')
                instruction, scene, _, language = candidates[0]
                video = row['video'].split('/')[-1]
                if scene != video.split('_'+name+'_')[0] or (name=='rxr' and language not in ('en-US','en-IN')):
                    raise ValueError('Upstream scan/language mismatch')
                if video not in actions_by_video:
                    pose_path = Path(args.trajectories)/name/'traj'/(video+'.npz')
                    if sha_file(pose_path) != pose_hashes[(name,video)]:
                        raise ValueError('Collected trajectory hash mismatch')
                    with np.load(pose_path,allow_pickle=False) as pose:
                        actions = [int(x) for x in pose['actions']]
                        if len(actions)!=row['n_frames'] or actions[0]!=-1 or pose['positions'].shape!=(len(actions),3) or pose['rotations'].shape!=(len(actions),4):
                            raise ValueError('Trajectory/action dimensions differ')
                    actions_by_video[video] = actions[1:]+[0]
                c = int(row['c'])
                hist = history(c)
                images = [row['video']+'/rgb/'+format(j+1,'03d')+'.jpg' for j in hist+[c]]
                action_ids = actions_by_video[video][c:c+4]
                action = ''.join(GLYPHS[a] for a in action_ids)
                human = SYSTEM_PROMPT.replace('<instruction>.',instruction)
                if hist:
                    human += ' These are your historical observations: <memory>.'
                human = (human+' '+CONJUNCTIONS[int(conjunctions[index])]+'<image>.').rstrip()
                rebuilt = {key:row[key] for key in ('video','ep_id','ins_id','c','n_frames')}
                rebuilt.update(dataset=('R2R_v2' if name=='r2r' else 'RxR_v2'),instruction=instruction,
                    images=images,n_memory=len(hist),action_ids=action_ids,action_text=action,
                    conversations=[{'from':'human','value':human},{'from':'gpt','value':action}],
                    subtask_phase_index=row['phase'],object_supervision=copy.deepcopy(row['object_supervision']))
                offsets[index] = target.tell()
                target.write((json.dumps(rebuilt,ensure_ascii=False,separators=(',',':'))+'\n').encode('utf-8'))
        np.save(target_dir/'offsets.npy',offsets,allow_pickle=False)
        arrays = {}
        for key,meta in source['arrays'].items():
            dst = target_dir/(key+'.npy')
            if key!='offsets':
                src = addon/meta['path']
                if sha_file(src)!=meta['sha256']:
                    raise ValueError('Label sidecar hash mismatch')
                shutil.copyfile(src,dst)
            array = np.load(dst,mmap_mode='r',allow_pickle=False)
            arrays[key] = {'path':dst.relative_to(output).as_posix(),'sha256':sha_file(dst),
                           'shape':list(array.shape),'dtype':str(array.dtype)}
        stat = path.stat()
        result['sources'].append({'name':name,'path':str(path),'sha256':sha_file(path),
            'size_bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns,'source_rows':source_rows[name],
            'selected_rows':source['rows'],'fraction':0.25,'annotation_complete':True,
            'object_masks_enabled':True,'rgb_root':str(Path(getattr(args,name+'_rgb')).resolve()),
            'mask_root':str(addon),'arrays':arrays})
        print(json.dumps({'source':name,'rows':source['rows'],'trajectories':len(actions_by_video)}),flush=True)
    (output/'manifest.local.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print('Wrote local stage-3 records and manifest; render/inspect RGB before training.',flush=True)


if __name__=='__main__':
    main()

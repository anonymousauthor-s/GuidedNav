"""Freeze a seeded 25%-of-full-train selection from completed S/O annotations."""
import collections,concurrent.futures,hashlib,json,os,re,time
from pathlib import Path
import numpy as np
import argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--prior-full-data',required=True,help='Directory containing the full-source manifest and R2R indexed arrays')
parser.add_argument('--rxr-completed-exports',required=True,help='Original schema-v2 completed exports: cases/, dataset/shards/')
parser.add_argument('--completed-snapshot',required=True,help='Frozen completed-case journal snapshot from the exact selected run')
parser.add_argument('--output',required=True,help='New output directory; data/ and control/ will be created')
args=parser.parse_args()
R=Path(args.output).resolve();PRIOR=Path(args.prior_full_data).resolve();PROD=Path(args.rxr_completed_exports).resolve()
(R/'control').mkdir(parents=True,exist_ok=True);(R/'data').mkdir(exist_ok=True)

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for buf in iter(lambda:f.read(8*1024**2),b''):h.update(buf)
 return h.hexdigest()
def save(p,d):
 temp=p.with_name(p.name+'.tmp');temp.write_text(json.dumps(d,indent=2));temp.replace(p)
def progress(**d):
 d['time']=time.time();save(R/'control/data_progress.json',d);print(json.dumps(d),flush=True)
def emit_arrays(name,arrays):
 folder=R/'data'/name;folder.mkdir(exist_ok=True);result={}
 for key,a in arrays.items():
  p=folder/(key+'.npy');np.save(p,a,allow_pickle=False)
  result[key]={'path':str(p.relative_to(R/'data')),'sha256':sha(p),'shape':list(a.shape),'dtype':str(a.dtype)}
 return result
def rebalance(groups):
 valid=groups>=0;out=np.zeros(len(groups),dtype=np.float32)
 if valid.any():
  unique,counts=np.unique(groups[valid],return_counts=True)
  out[valid]=valid.sum()/(len(unique)*counts[np.searchsorted(unique,groups[valid])])
 return out
assert not (R/'data/manifest.json').exists()
previous=json.loads((PRIOR/'manifest.json').read_text());sources=[]
# R2R full V5 completed pipeline; preserve original row identities and offsets.
old=previous['sources'][0];assert old['name']=='r2r' and old['source_rows']==629862
progress(state='verifying_r2r_source')
assert sha(Path(old['path']))==old['sha256']
selected=np.sort(np.random.RandomState(42).choice(old['source_rows'],old['source_rows']//4,replace=False))
arrays={}
for key,meta in old['arrays'].items():
 p=PRIOR/meta['path'];assert sha(p)==meta['sha256'];arrays[key]=np.array(np.load(p,mmap_mode='r')[selected])
arrays['weights']=rebalance(arrays['groups']);meta=dict(old)
meta.update(fraction=.25,selected_rows=len(selected),eligible_rows=old['source_rows'],object_masks_enabled=True,
            arrays=emit_arrays('r2r',arrays),s_valid_rows=int((arrays['weights']>0).sum()),o_eligible_rows=int(arrays['o_eligible'].sum()),
            s_weight_sum=float(arrays['weights'].sum(dtype=np.float64)),sampling_unit='sample_row',sampling_seed=42,
            annotation_complete=True,annotation_provenance='Completed R2R train_full_v1 subtask/object pipeline')
meta['parent_full_source_statistics']={k:meta.pop(k) for k in ['s_groups','s_coverage','image_counts'] if k in meta}
sources.append(meta)
# Reuse the exact completed-case set, rather than sampling a later journal prefix.
snapshot=json.loads(Path(args.completed_snapshot).read_text())
done={int(k):v for k,v in snapshot['cases'].items()}
assert len(done)>=6885
save(R/'data/rxr_completed_snapshot.json',snapshot)
pool=sum(x['frames'] for x in done.values());quota=1889294//4;assert pool>=quota
chosen=np.sort(np.random.RandomState(42).choice(pool,quota,replace=False));case_selection=[];offset=0
for iid in sorted(done):
 count=done[iid]['frames'];ids=chosen[(chosen>=offset)&(chosen<offset+count)]-offset
 if len(ids):case_selection.append((iid,ids.tolist(),offset))
 offset+=count
progress(state='rxr_loading_completed_exports',completed_cases=len(done),eligible_rows=pool,selected_rows=quota)

def load_case(spec):
 iid,ids,base_offset=spec;key='%06d'%iid;s=done[iid]
 marker=json.loads((PROD/'cases'/key/'_PRODUCTION_SUCCESS.json').read_text())
 assert marker['files']==s['files'] and marker['export_schema_version']==2
 folder=PROD/'dataset/shards'/key
 raw={name:(folder/name).read_bytes() for name in ['samples_perstep.jsonl','trajectories.jsonl']}
 for name,data in raw.items():assert hashlib.sha256(data).hexdigest()==s['files'][name],(iid,name)
 traj=json.loads(raw['trajectories.jsonl']);assert traj['split']=='train' and traj['language'] in ('en-US','en-IN')
 assert int(traj['rxr_instruction_id'])==iid
 rows=[json.loads(x) for x in raw['samples_perstep.jsonl'].splitlines() if x.strip()]
 assert len(rows)==s['frames']
 parts={int(x['phase_index']):x for x in traj['subtasks']};out=[];directories={}
 for j in ids:
  row=rows[j];assert int(row['rxr_instruction_id'])==iid and row['instruction']==traj['instruction']
  phase=int(row['subtask_phase_index']);part=parts[phase];bounds=part['char_span'];a,z=bounds
  assert 0<=a<z<=len(row['instruction']) and row['instruction'][a:z].strip()==row['subtask'].strip()==part['text'].strip()
  obj=row['object_supervision'];assert isinstance(obj['mask_loss_eligible'],bool)
  assert bool(obj['active_masks'])==obj['mask_loss_eligible']
  for m in obj['active_masks']:
   p=Path(m['mask']);assert not p.is_absolute() and '..' not in p.parts
   p=PROD.parent/p
   if p.parent not in directories:
    with os.scandir(str(p.parent)) as entries:directories[p.parent]={e.name for e in entries if e.is_file()}
   assert p.name in directories[p.parent]
   assert m.get('frame_index',row['c'])==row['c'] and m.get('source_phase_index',phase)==phase
   assert len(m['mask_sha256'])==64
  assert row['action_text'].strip() and not re.sub(r'(?:STOP|[\u2190\u2192\u2191\s])','',row['action_text'])
  out.append((row,bounds,base_offset+j))
 return iid,out

arrays={'offsets':np.empty(quota,dtype=np.int64),'source_indices':np.empty(quota,dtype=np.int64),
        'spans':np.empty((quota,2),dtype=np.int32),'phases':np.empty(quota,dtype=np.int16),
        'groups':np.empty(quota,dtype=np.int32),'o_eligible':np.empty(quota,dtype=np.bool_)}
group_ids={};seen=set();masks={};h=hashlib.sha256();pos=0;i=0
output=R/'data/rxr_samples_perstep.jsonl';partial=output.with_suffix('.partial')
# Bounded batches avoid retaining a multi-gigabyte set of decoded rows in memory.
with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool_executor,partial.open('wb') as f:
 for start in range(0,len(case_selection),64):
  for iid,rows in pool_executor.map(load_case,case_selection[start:start+64]):
   for row,bounds,source_index in rows:
    identity=(iid,int(row['c']));assert identity not in seen;seen.add(identity)
    phase=int(row['subtask_phase_index']);g=group_ids.setdefault((iid,phase),len(group_ids))
    encoded=(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n').encode()
    arrays['offsets'][i]=pos;arrays['source_indices'][i]=source_index;arrays['spans'][i]=bounds
    arrays['phases'][i]=phase;arrays['groups'][i]=g;arrays['o_eligible'][i]=row['object_supervision']['mask_loss_eligible']
    for m in row['object_supervision']['active_masks']:masks[m['mask']]=m['mask_sha256']
    pos+=len(encoded);h.update(encoded);f.write(encoded);i+=1
  if start%256==0:progress(state='rxr_writing',cases=min(start+64,len(case_selection)),total_cases=len(case_selection),rows=i)
 f.flush();os.fsync(f.fileno())
assert i==quota;partial.replace(output)
arrays['weights']=rebalance(arrays['groups']);old=previous['sources'][1]
meta={'name':'rxr','path':str(output),'sha256':h.hexdigest(),'size_bytes':output.stat().st_size,'mtime_ns':output.stat().st_mtime_ns,
      'source_rows':1889294,'selected_rows':quota,'fraction':.25,'eligible_rows':pool,'eligible_trajectories':len(done),
      'rgb_root':old['rgb_root'],'mask_root':str(PROD.parent),'arrays':emit_arrays('rxr',arrays),
      'object_masks_enabled':True,'s_valid_rows':quota,'s_weight_sum':float(arrays['weights'].sum(dtype=np.float64)),
      'o_eligible_rows':int(arrays['o_eligible'].sum()),'unique_mask_files':len(masks),
      'annotation_complete':True,'sampling_unit':'sample_row','sampling_seed':42,
      'annotation_provenance':'Frozen completed schema-v2 production journal; per-export SHA256 verified; original train English expert trajectories only',
      'all_selected_mask_paths_checked':True,'missing_object_policy':'not visible/no mask is missing supervision, not a negative target'}
sources.append(meta);save(R/'data/rxr_mask_hashes.json',masks)
manifest={'format':'full_vln_expert25_so_v1','sources':sources,'selected_rows':sum(x['selected_rows'] for x in sources),
          'seed':42,'all_rows_annotation_complete':True,'navigation_extra_data':False,
          'sampling':'25% of each full train source row count without replacement, from annotation-complete candidates only',
          's_weight_normalization':'mean one among selected valid S rows within each source; equal weight per selected trajectory phase',
          'epochs':1,'parent_manifest_sha256':sha(PRIOR/'manifest.json')}
assert manifest['selected_rows']==629788
save(R/'data/manifest.json',manifest)
progress(state='complete',rows=manifest['selected_rows'],steps=(manifest['selected_rows']+255)//256,r2r_rows=sources[0]['selected_rows'],rxr_rows=quota,rxr_completed_cases=len(done),rxr_object_rows=meta['o_eligible_rows'])

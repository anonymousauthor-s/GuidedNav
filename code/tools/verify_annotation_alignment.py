"""Verify annotation pointers against authorized public text and optional local training inputs.

No downloads are performed. Original text is never written to the output reports.
With --training-data, check every row's image/history/action/prompt reconstruction
and optionally export the small conjunction-ID sidecar needed for exact prompts.
"""
import argparse
import collections
import gzip
import hashlib
import json
from pathlib import Path
import re
import sys
import numpy as np

CONJUNCTIONS = ('you can see ', 'in front of you is ', 'there is ',
                'you can spot ', 'you are toward the ', 'ahead of you is ',
                'in your sight is ')
SYSTEM_PROMPT = ('You are an autonomous navigation assistant. Your task is to '
                 '<instruction>. Devise an action sequence to follow the instruction '
                 'using the four actions: TURN LEFT (←) or TURN RIGHT (→) by 15 '
                 'degrees, MOVE FORWARD (↑) by 25 centimeters, or STOP.')
LEGACY_SUFFIX = (' First identify the immediate navigation subtask. Respond in exactly two '
                 "lines: 'SUBTASK: <short imperative navigation instruction>' followed by "
                 "'ACTION: <sequence>', where ACTION uses only TURN LEFT (←), TURN RIGHT (→), "
                 'MOVE FORWARD (↑), or STOP.')
GLYPHS = {0:'STOP', 1:'↑', 2:'←', 3:'→'}


def sha_file(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2), b''):
            value.update(block)
    return value.hexdigest()


def text_sha(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def open_text(path):
    return (gzip.open if str(path).endswith('.gz') else open)(path, 'rt', encoding='utf-8')


def public_index(path, name):
    result = collections.defaultdict(list)
    if name == 'r2r':
        with open_text(path) as stream:
            for row in json.load(stream):
                for index, text in enumerate(row['instructions']):
                    result[(int(row['path_id']), text_sha(text))].append(
                        (text, row['scan'], index, None))
    else:
        with open_text(path) as stream:
            for line in stream:
                row = json.loads(line)
                result[int(row['instruction_id'])].append(
                    (row['instruction'], row['scan'], None, row['language']))
    return result


def history(c):
    return sorted({int(round(x)) for x in np.linspace(0, c, 8) if int(round(x)) < c}) if c > 0 else []


def fail_if_overwrite(path):
    if path and Path(path).exists():
        raise FileExistsError('Refusing to overwrite existing output: '+str(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations', required=True, help='Annotation-addon directory')
    parser.add_argument('--r2r-source', required=True, help='Official FGR2R_train.json')
    parser.add_argument('--rxr-source', required=True, help='Official rxr_train_guide.jsonl.gz')
    parser.add_argument('--training-data', help='Optional locally prepared data/ with samples_perstep.jsonl and traj/')
    parser.add_argument('--report', required=True)
    parser.add_argument('--mapping-output', help='Unique public-ID/hash mappings; contains no instruction text')
    parser.add_argument('--prompt-output', help='Per-row conjunction IDs; requires --training-data')
    args = parser.parse_args()
    if args.prompt_output and not args.training_data:
        parser.error('--prompt-output requires --training-data')
    for path in (args.report, args.mapping_output, args.prompt_output):
        fail_if_overwrite(path)
    root = Path(args.annotations)
    manifest = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    report = {'scope':'Final stage-3 selected subset, not full Guided datasets',
              'text_normalization':'none; exact UTF-8 instruction SHA256', 'sources':{},
              'does_not_verify':'GPU training, simulator rendering/pixels, model metrics'}
    mappings = open(args.mapping_output, 'x', encoding='utf-8') if args.mapping_output else None
    prompts = open(args.prompt_output, 'x', encoding='utf-8') if args.prompt_output else None
    try:
        for source in manifest['sources']:
            name = source['name']
            public_path = Path(getattr(args, name+'_source'))
            public = public_index(public_path, name)
            arrays = {key:np.load(root/meta['path'], mmap_mode='r', allow_pickle=False)
                      for key,meta in source['arrays'].items() if key != 'offsets'}
            issues, counts = [], collections.Counter()
            unique, matched, videos, pose_cache = set(), set(), set(), {}
            original = (Path(args.training_data)/name/'samples_perstep.jsonl').open('rb') if args.training_data else None
            annotations = root/source['path']
            try:
                with annotations.open('rb') as stream:
                    for index, line in enumerate(stream):
                        row = json.loads(line)
                        counts['rows'] += 1
                        videos.add(row['video'])
                        source_id = int(row['fgr_path_id'] if name=='r2r' else row['rxr_instruction_id'])
                        key = (source_id, row['instruction_sha256'])
                        unique.add(key)
                        candidates = public.get(key if name=='r2r' else source_id, [])
                        candidates = [v for v in candidates if text_sha(v[0]) == row['instruction_sha256']]
                        if len(candidates) != 1:
                            counts['unmatched_or_ambiguous_rows'] += 1
                            if len(issues)<20:
                                issues.append({'source_id':source_id, 'reason':'missing/ambiguous exact upstream text', 'sha256':key[1]})
                            if original:
                                original.readline()
                            continue
                        text, scene, instruction_index, language = candidates[0]
                        if key not in matched and mappings:
                            mappings.write(json.dumps({'source':name, 'upstream_id':source_id,
                                'upstream_instruction_index':instruction_index,
                                'instruction_sha256':key[1], 'scan':scene, 'language':language})+'\n')
                        matched.add(key)
                        counts['exact_upstream_rows'] += 1
                        scene_from_video = row['video'].split('/')[-1].split('_'+name+'_')[0]
                        if scene != scene_from_video or (name=='rxr' and language not in ('en-US','en-IN')):
                            raise ValueError('Upstream scan/language mismatch')
                        span = [int(x) for x in arrays['spans'][index]]
                        if span != row['subtask_char_span'] or (span != [-1,-1] and not 0 <= span[0] < span[1] <= len(text)):
                            raise ValueError('Invalid character-span pointer')
                        if row['source_index'] != int(arrays['source_indices'][index]):
                            raise ValueError('Source index sidecar mismatch')
                        if not original:
                            continue
                        raw = json.loads(original.readline())
                        if raw['instruction'] != text or any(raw[k] != row[k] for k in ('video','ep_id','ins_id','c','n_frames')):
                            raise ValueError('Training identity/text mismatch')
                        c, n = int(row['c']), int(row['n_frames'])
                        hist = history(c)
                        images = [row['video']+'/rgb/'+format(j+1,'03d')+'.jpg' for j in hist+[c]]
                        if raw['images'] != images or raw['n_memory'] != len(hist):
                            raise ValueError('Image/history reconstruction mismatch')
                        counts['image_history_rows_verified'] += 1
                        video = row['video'].split('/')[-1]
                        if video not in pose_cache:
                            path = Path(args.training_data)/name/'traj'/(video+'.npz')
                            with np.load(path, allow_pickle=False) as pose:
                                actions = [int(x) for x in pose['actions']]
                                if (len(pose['positions']) != n or pose['rotations'].shape != (n,4)
                                        or len(actions) != n or actions[0] != -1):
                                    raise ValueError('Collected pose/action shape or reset convention mismatch')
                            pose_cache[video] = actions[1:] + [0]
                        action_ids = pose_cache[video][c:c+4]
                        action = ''.join(GLYPHS[v] for v in action_ids)
                        if action_ids != raw['action_ids'] or action != raw['action_text']:
                            raise ValueError('Action target reconstruction mismatch')
                        counts['action_rows_verified'] += 1
                        human = raw['conversations'][0]['value']
                        if human.endswith(LEGACY_SUFFIX):
                            human = human[:-len(LEGACY_SUFFIX)]
                        base = SYSTEM_PROMPT.replace('<instruction>.', text)
                        if hist:
                            base += ' These are your historical observations: <memory>.'
                        choices = [(base+' '+conj+'<image>.').rstrip() for conj in CONJUNCTIONS]
                        if human.rstrip() not in choices:
                            counts['unrecognized_prompt_rows'] += 1
                            if len(issues)<20:
                                issues.append({'source_index':row['source_index'],'reason':'unknown prompt template'})
                        else:
                            conjunction = choices.index(human.rstrip())
                            counts['normalized_prompt_rows_verified'] += 1
                            if prompts:
                                prompts.write(json.dumps({'source':name,'source_index':row['source_index'],
                                    'conjunction_id':conjunction},separators=(',',':'))+'\n')
                        target = raw['conversations'][1]['value'].strip()
                        if target != action:
                            legacy = re.fullmatch(r'SUBTASK:\s*(.*?)\s*\nACTION:\s*(.+)', target, flags=re.DOTALL)
                            if not legacy or legacy.group(1).strip() != raw.get('subtask','').strip() or legacy.group(2).strip() != action:
                                raise ValueError('Normalized assistant target mismatch')
                        old_masks = [v['mask_sha256'] for v in raw['object_supervision'].get('active_masks',[])]
                        new_masks = [v['mask_sha256'] for v in row['object_supervision'].get('active_masks',[])]
                        if old_masks != new_masks:
                            raise ValueError('Mask-label pointers differ from frozen training data')
                        counts['mask_pointer_rows_verified'] += 1
                if counts['rows'] != source['rows'] or (original and original.readline()):
                    raise ValueError('Wrong number of annotation/training rows')
            finally:
                if original:
                    original.close()
            passed = (counts['exact_upstream_rows'] == source['rows'] and not counts['unrecognized_prompt_rows'])
            report['sources'][name] = {'passed':passed, **dict(counts),
                'unique_instruction_ids':len(unique), 'exact_unique_instructions':len(matched),
                'unique_collected_trajectories':len(videos), 'pose_action_trajectories_verified':len(pose_cache),
                'public_source_sha256':sha_file(public_path),
                'annotation_jsonl_sha256':sha_file(annotations), 'issues':issues}
            print(json.dumps({'source':name, **report['sources'][name]}), flush=True)
        report['passed'] = all(v['passed'] for v in report['sources'].values())
        Path(args.report).write_text(json.dumps(report,indent=2)+'\n', encoding='utf-8')
    finally:
        if mappings:
            mappings.close()
        if prompts:
            prompts.close()
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())

"""Audit full RxR English validation with explicitly specified collision body.

Only creates task-owned reports/navmesh cache; original datasets and scenes are
never changed. This validates geometry, not policy evaluation or benchmark score.
"""
import gzip
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
import numpy as np

import argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--dataset',required=True)
parser.add_argument('--gt',required=True)
parser.add_argument('--scenes-root',required=True,help='Contains mp3d/<scan>/<scan>.glb')
parser.add_argument('--output',required=True,help='New owned cache directory for *.navmesh and audit.json')
args=parser.parse_args()
import habitat_sim
out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True)
source=Path(args.dataset).resolve();gt_file=Path(args.gt).resolve();scene_root=Path(args.scenes_root).resolve()
if hashlib.sha256(source.read_bytes()).hexdigest()!='93abddce58e8232867a8738daa8107eb82953693a9d5e70779d1eb00386b222a':
    raise ValueError('Dataset SHA256 differs from the final RxR benchmark')
if hashlib.sha256(gt_file.read_bytes()).hexdigest()!='bdfbb624132501bf297eb0ea16a99e38682668ba5e9ce1554ca5cdcde6b65a0d':
    raise ValueError('Ground-truth SHA256 differs from the final RxR benchmark')
with gzip.open(str(source), 'rt') as f:
    episodes = [e for e in json.load(f)['episodes'] if e['instruction']['language'] in ('en-US', 'en-IN')]
with gzip.open(str(gt_file), 'rt') as f:
    gt = json.load(f)
assert len(episodes) == 3669
assert Counter(e['instruction']['language'] for e in episodes) == {'en-US': 1223, 'en-IN': 2446}
report = {'count': len(episodes), 'height': 0.88, 'radius': 0.18, 'scenes': {},
          'unreachable': [], 'invalid_gt': [], 'input_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
          'gt_sha256': hashlib.sha256(gt_file.read_bytes()).hexdigest()}
for scene in sorted({Path(e['scene_id']).stem for e in episodes}):
    cfg = habitat_sim.SimulatorConfiguration()
    cfg.scene_id = str(scene_root/'mp3d'/scene/(scene+'.glb'))
    cfg.create_renderer = False
    cfg.enable_physics = False
    agent = habitat_sim.agent.AgentConfiguration()
    agent.height, agent.radius = .88, .18
    agent.sensor_specifications = []
    sim = habitat_sim.Simulator(habitat_sim.Configuration(cfg, [agent]))
    settings = habitat_sim.NavMeshSettings()
    settings.set_defaults()
    settings.agent_height, settings.agent_radius = .88, .18
    pf = habitat_sim.PathFinder()
    assert sim.recompute_navmesh(pf, settings)
    navmesh = out/(scene+'.navmesh')
    assert pf.save_nav_mesh(str(navmesh))
    scene_rows = [e for e in episodes if Path(e['scene_id']).stem == scene]
    failures = []
    for e in scene_rows:
        route = habitat_sim.MultiGoalShortestPath()
        route.requested_start = np.array(e['start_position'], dtype=np.float32)
        route.requested_ends = np.array([g['position'] for g in e['goals']], dtype=np.float32)
        if not pf.find_path(route) or not math.isfinite(route.geodesic_distance):
            failures.append(str(e['episode_id']))
        reference = gt.get(str(e['episode_id']), {}).get('locations')
        if not reference or any(len(p) != 3 or any(not math.isfinite(x) for x in p) for p in reference):
            report['invalid_gt'].append(str(e['episode_id']))
    report['scenes'][scene] = {'count':len(scene_rows), 'unreachable':failures,
                             'navmesh_sha256':hashlib.sha256(navmesh.read_bytes()).hexdigest()}
    report['unreachable'].extend(failures)
    sim.close()
    print(json.dumps({'scene': scene, 'count': len(scene_rows), 'unreachable': len(failures)}), flush=True)
report['complete'] = not report['unreachable'] and not report['invalid_gt']
(out/'audit.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report), flush=True)
assert report['complete'], 'Full RxR geometry audit failed'

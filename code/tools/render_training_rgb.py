"""Render released agent poses against a locally licensed MP3D scene installation."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trajectories', required=True, help='Directory of released per-video NPZ trajectories')
    parser.add_argument('--scenes-root', required=True, help='Contains mp3d/<scan>/<scan>.glb or <scan>/<scan>.glb')
    parser.add_argument('--output', required=True, help='RGB root; writes images/<video>/rgb/001.jpg etc.')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world-size', type=int, default=1)
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    import habitat_sim
    from habitat_sim.utils.common import quat_from_coeffs
    if not 0 <= args.rank < args.world_size:
        parser.error('rank must lie within world-size')
    files = sorted(Path(args.trajectories).glob('*.npz'))[args.rank::args.world_size]
    if args.limit:
        files = files[:args.limit]
    if not files:
        parser.error('No trajectory NPZ files found')
    root = Path(args.scenes_root)
    simulator, current_scene = None, None
    count = 0
    try:
        for path in files:
            video = path.stem
            scene = video.split('_r2r_')[0].split('_rxr_')[0]
            candidates = [root / scene / f'{scene}.glb', root / 'mp3d' / scene / f'{scene}.glb']
            scene_path = next((p for p in candidates if p.is_file()), None)
            if scene_path is None:
                raise FileNotFoundError(f'Licensed MP3D scene missing: {scene}')
            if current_scene != scene:
                if simulator is not None:
                    simulator.close()
                config = habitat_sim.SimulatorConfiguration()
                config.scene_id, config.gpu_device_id = str(scene_path), args.gpu
                config.enable_physics = False
                sensor = habitat_sim.CameraSensorSpec()
                sensor.uuid = 'rgb'
                sensor.sensor_type = habitat_sim.SensorType.COLOR
                sensor.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
                sensor.resolution = [480, 640]
                sensor.hfov = 79.0
                sensor.position = [0.0, 1.25, 0.0]
                sensor.orientation = [0.0, 0.0, 0.0]
                agent_config = habitat_sim.agent.AgentConfiguration()
                agent_config.sensor_specifications = [sensor]
                simulator = habitat_sim.Simulator(habitat_sim.Configuration(config, [agent_config]))
                current_scene = scene
            with np.load(path, allow_pickle=False) as trajectory:
                positions, rotations = trajectory['positions'], trajectory['rotations']
                if (positions.shape != (len(rotations), 3) or rotations.shape[1:] != (4,)
                        or not np.isfinite(positions).all() or not np.isfinite(rotations).all()
                        or not np.allclose(np.linalg.norm(rotations, axis=-1), 1.0, atol=2e-3)):
                    raise ValueError(f'Invalid world-space agent poses: {path.name}')
                destination = Path(args.output) / 'images' / video / 'rgb'
                destination.mkdir(parents=True, exist_ok=True)
                agent = simulator.get_agent(0)
                for index, (position, wxyz) in enumerate(zip(positions, rotations), start=1):
                    output = destination / f'{index:03d}.jpg'
                    if output.exists():
                        continue
                    state = habitat_sim.AgentState()
                    state.position = position
                    # The original collector saved [q.w,q.x,q.y,q.z]. Habitat's
                    # quat_from_coeffs takes [q.x,q.y,q.z,q.w]. These are agent
                    # poses; camera height is applied once by the sensor config.
                    state.rotation = quat_from_coeffs(wxyz[[1, 2, 3, 0]])
                    agent.set_state(state, reset_sensors=True)
                    rgb = simulator.get_sensor_observations()['rgb'][..., :3]
                    Image.fromarray(rgb).convert('RGB').save(output)
                    count += 1
            print(json.dumps({'video': video, 'frames': len(positions), 'written_total': count}), flush=True)
    finally:
        if simulator is not None:
            simulator.close()


if __name__ == '__main__':
    main()

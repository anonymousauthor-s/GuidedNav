"""Install the hash-pinned RxR meshes for the reference 0.88 m / 0.18 m collision body."""
import hashlib
import json
import math
import os
from pathlib import Path

from habitat.core.registry import registry
from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim


@registry.register_simulator(name='RxRFullGeometrySim-v1')
class RxRFullGeometrySim(HabitatSim):
    def __init__(self, config):
        self._rxr_mesh_scene = None
        audit_path = Path(os.environ['RXR_NAVMESH_AUDIT_PATH'])
        self._rxr_mesh_root = audit_path.parent
        self._rxr_mesh_audit = json.loads(audit_path.read_text(encoding='utf-8'))
        audit = self._rxr_mesh_audit
        if not audit.get('complete') or audit.get('count') != 3669 or audit.get('unreachable') or audit.get('invalid_gt'):
            raise ValueError('RxR full geometry audit is incomplete')
        super().__init__(config)
        self._install_rxr_mesh()

    def _install_rxr_mesh(self):
        scene = Path(self.habitat_config.scene).stem
        if scene == self._rxr_mesh_scene:
            return
        audit = self._rxr_mesh_audit
        for agent in self.habitat_config.agents.values():
            if not math.isclose(float(agent.height), audit['height'], abs_tol=1e-6) or not math.isclose(float(agent.radius), audit['radius'], abs_tol=1e-6):
                raise ValueError('RxR simulator collision body differs from audited navmesh')
        if scene not in audit['scenes']:
            raise ValueError(f'Scene absent from the full RxR geometry audit: {scene}')
        path = self._rxr_mesh_root / (scene + '.navmesh')
        if hashlib.sha256(path.read_bytes()).hexdigest() != audit['scenes'][scene]['navmesh_sha256']:
            raise ValueError(f'RxR navigation mesh hash mismatch: {scene}')
        if not self.pathfinder.load_nav_mesh(str(path)):
            raise RuntimeError(f'Cannot load audited RxR navmesh: {path}')
        settings = self.pathfinder.nav_mesh_settings
        if not math.isclose(settings.agent_height, audit['height'], abs_tol=1e-6) or not math.isclose(settings.agent_radius, audit['radius'], abs_tol=1e-6):
            raise ValueError('Serialized navigation mesh collision body mismatch')
        self._rxr_mesh_scene = scene

    def reconfigure(self, *args, **kwargs):
        super().reconfigure(*args, **kwargs)
        self._install_rxr_mesh()

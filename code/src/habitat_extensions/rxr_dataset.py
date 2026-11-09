"""RxR VLN-CE dataset loader for Habitat-Lab 0.2.4/OmegaConf.

The upstream VLN-CE loader targets the older yacs configuration API.  This
small adapter preserves the official episode schema while accepting the
lower-case Habitat-Lab 0.2.4 dataset config used by the joint-model evaluator.
"""

import gzip
import json
import os
from typing import Dict, List, Optional, Union

import attr

from habitat.core.dataset import Dataset
from habitat.core.registry import registry
from habitat.tasks.nav.nav import NavigationGoal
from habitat.tasks.vln.vln import VLNEpisode


DEFAULT_SCENE_PATH_PREFIX = "data/scene_datasets/"
DEFAULT_LANGUAGES = ("en-US", "en-IN")


@attr.s(auto_attribs=True)
class RxRInstructionData:
    instruction_text: str
    instruction_id: Optional[str] = None
    language: Optional[str] = None
    annotator_id: Optional[str] = None
    edit_distance: Optional[float] = None
    timed_instruction: Optional[List[Dict[str, Union[float, str]]]] = None
    instruction_tokens: Optional[List[str]] = None


@registry.register_dataset(name="RxREnglishGuideVLNCE-v1")
class RxREnglishGuideVLNCEDatasetV1(Dataset):
    """Load the official guide episodes and retain en-US/en-IN only."""

    episodes: List[VLNEpisode]

    @staticmethod
    def check_config_paths_exist(config) -> bool:
        return os.path.exists(
            config.data_path.format(split=config.split)
        ) and os.path.exists(config.scenes_dir)

    def __init__(self, config=None) -> None:
        self.episodes = []
        if config is None:
            return

        dataset_filename = config.data_path.format(split=config.split)
        with gzip.open(dataset_filename, "rt", encoding="utf-8") as handle:
            self.from_json(handle.read(), scenes_dir=config.scenes_dir)

        languages = set(config.get("languages", DEFAULT_LANGUAGES))
        self.episodes = [
            episode
            for episode in self.episodes
            if episode.instruction.language in languages
        ]
        self.episodes = list(
            filter(self.build_content_scenes_filter(config), self.episodes)
        )

    def from_json(self, json_str: str, scenes_dir: Optional[str] = None) -> None:
        for serialized_episode in json.loads(json_str)["episodes"]:
            payload = dict(serialized_episode)
            instruction = RxRInstructionData(**payload["instruction"])
            payload["instruction"] = instruction
            episode = VLNEpisode(**payload)

            if scenes_dir is not None:
                if episode.scene_id.startswith(DEFAULT_SCENE_PATH_PREFIX):
                    episode.scene_id = episode.scene_id[
                        len(DEFAULT_SCENE_PATH_PREFIX) :
                    ]
                episode.scene_id = os.path.join(scenes_dir, episode.scene_id)

            if episode.goals is not None:
                episode.goals = [NavigationGoal(**goal) for goal in episode.goals]
            self.episodes.append(episode)

    @classmethod
    def get_scenes_to_load(cls, config) -> List[str]:
        if not cls.check_config_paths_exist(config):
            return []
        dataset = cls(config)
        return sorted(
            {cls.scene_from_scene_path(episode.scene_id) for episode in dataset.episodes}
        )

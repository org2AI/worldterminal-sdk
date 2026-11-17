"""Opt-in validation of named ARX X5 joint observations and action rows."""

import json
from collections.abc import Mapping
from importlib.resources import files

import numpy as np


def arx_x5():
    """Return a fresh copy of the public joint-v1 profile."""
    return json.loads(files(__package__).joinpath("arx_x5.json").read_text(encoding="utf-8"))


class JointProfile:
    def __init__(self, config):
        canonical = arx_x5()
        if not isinstance(config, Mapping) or set(config) != set(canonical):
            raise ValueError("embodiment: expected id, state, action and video")
        for key in ("id", "state", "action"):
            if config[key] != canonical[key]:
                raise ValueError(f"embodiment.{key}: incompatible with {canonical['id']}")
        video = config["video"]
        if not isinstance(video, Mapping) or set(video) != set(canonical["video"]):
            raise ValueError("embodiment.video: expected head, left_wrist and right_wrist")
        if any(not isinstance(name, str) or not name.strip() for name in video.values()):
            raise ValueError("embodiment.video: expected nonempty camera names")
        if len(set(video.values())) != len(video):
            raise ValueError("embodiment.video: camera names must be distinct")
        self.id = canonical["id"]
        self.state = canonical["state"]
        self.action = canonical["action"]
        self.cameras = {video[role]: name for role, name in canonical["video"].items()}

    def observation(self, obs):
        check_vectors(obs.get("state"), self.state, "state")
        vision = obs.get("vision")
        if not isinstance(vision, Mapping):
            raise ValueError("vision: expected camera views")
        mapped = {key: view for key, view in vision.items() if key not in self.cameras}
        for source, target in self.cameras.items():
            view = vision.get(source)
            rgb = view.get("color") if isinstance(view, Mapping) else None
            if not (isinstance(rgb, np.ndarray) and rgb.dtype == np.uint8 and rgb.ndim == 3
                    and rgb.shape[2] == 3 and min(rgb.shape[:2]) > 0):
                raise ValueError(f"vision.{source}.color: expected uint8 RGB array (H, W, 3)")
            if target in mapped:
                raise ValueError(f"vision.{target}: conflicts with configured camera {source}")
            mapped[target] = view
        return {**obs, "vision": mapped}


def check_vectors(values, spans, path):
    if not isinstance(values, Mapping):
        raise ValueError(f"{path}: expected a mapping")
    for key, (start, stop) in spans.items():
        field = f"{path}.{key}"
        if key not in values:
            raise ValueError(f"{field}: missing")
        vector = np.asarray(values[key])
        if vector.shape != (stop - start,):
            raise ValueError(f"{field}: expected shape ({stop - start},), got {vector.shape}")
        if vector.dtype.kind not in "iuf" or not np.isfinite(vector).all():
            raise ValueError(f"{field}: expected finite numbers")

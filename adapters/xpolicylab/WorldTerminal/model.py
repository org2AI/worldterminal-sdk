"""XPolicyLab adapter for the public worldterminal-sdk client."""
from typing import Any

from XPolicyLab.model_template import ModelTemplate

from worldterminal_sdk import Client, arx_x5


class Model(ModelTemplate):
    def __init__(self, model_cfg: dict[str, Any]):
        super().__init__()
        cfg = dict(model_cfg or {})
        self.robotwin = str(cfg.get("bench_name", "")).casefold() == "robotwin"
        self._robotwin_done = False
        self._robotwin_hold = None
        profile = None
        if cfg.get("embodiment"):
            profile = arx_x5()
            if cfg["embodiment"] != profile["id"]:
                raise ValueError(f"Unknown embodiment: {cfg['embodiment']}")
        self.client = Client(cfg.get('api_url'), embodiment=profile, timeout=float(cfg.get('api_request_timeout_s') or 120),
                             connect_timeout=float(cfg.get('api_connect_seconds') or 30))

    def reset(self):
        if self.robotwin:
            self._robotwin_done = False
            self._robotwin_hold = None
        return self.client.reset()

    def update_obs(self, obs):
        if self.robotwin:
            from worldterminal_sdk.robotwin import measured_hold

            self._robotwin_hold = None
            self._robotwin_hold = measured_hold(obs)
            if self._robotwin_done:
                return None
        return self.client.update_observation(obs)

    def get_action(self):
        if self.robotwin:
            if self._robotwin_done:
                return self._hold_action()
            batch = self.client.poll()
            while batch.waiting:
                batch = self.client.poll()
            if batch.done:
                self._robotwin_done = True
                return self._hold_action()
            return list(batch.actions)
        return self.client.get_action()

    def _hold_action(self):
        if self._robotwin_hold is None:
            raise ValueError('RoboTwin terminal hold requires a fresh measured observation')
        return [{name: list(values) for name, values in self._robotwin_hold.items()}]

    def on_trial_end(self, result=None):
        return self.client.end(result)

    def close(self):
        self.client.close()

    def update_obs_batch(self, obs_list):
        raise NotImplementedError('one WorldTerminal client per environment; eval_batch=false')

    def get_action_batch(self, env_idx_list=None):
        raise NotImplementedError('one WorldTerminal client per environment; eval_batch=false')

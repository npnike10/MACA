import argparse
import numpy as np
import gymnasium as gym

# vmas currently imports argparse.BooleanOptionalAction, which is unavailable in Python 3.8.
if not hasattr(argparse, "BooleanOptionalAction"):
    class BooleanOptionalAction(argparse.Action):
        def __init__(
            self,
            option_strings,
            dest,
            default=None,
            required=False,
            help=None,
        ):
            _option_strings = []
            for option_string in option_strings:
                _option_strings.append(option_string)
                if option_string.startswith("--"):
                    _option_strings.append("--no-" + option_string[2:])
            super().__init__(
                option_strings=_option_strings,
                dest=dest,
                nargs=0,
                default=default,
                required=required,
                help=help,
            )

        def __call__(self, parser, namespace, values, option_string=None):
            setattr(
                namespace,
                self.dest,
                False if option_string and option_string.startswith("--no-") else True,
            )

    argparse.BooleanOptionalAction = BooleanOptionalAction

import vmas


class _ScalarizeDones(gym.Wrapper):
    def step(self, action):
        obs, rew, term, trunc, info = self.env.step(action)
        if isinstance(term, (list, tuple, np.ndarray)):
            term = bool(np.all(term))
        if isinstance(trunc, (list, tuple, np.ndarray)):
            trunc = bool(np.all(trunc))
        return obs, rew, term, trunc, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class VMASWrapper(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, env_name, render_mode="rgb_array", **kwargs):
        self.render_mode = render_mode
        kwargs.pop("render_mode", None)

        base = vmas.make_env(
            env_name,
            num_envs=1,
            continuous_actions=False,
            dict_spaces=False,
            terminated_truncated=True,
            wrapper="gymnasium",
            **kwargs,
        )
        base.render_mode = self.render_mode
        self._env = _ScalarizeDones(base)
        self.n_agents = self._env.unwrapped.n_agents
        self.action_space = gym.spaces.Tuple(
            [gym.spaces.Discrete(5) for _ in range(self.n_agents)]
        )
        self.observation_space = self._env.observation_space

    def seed(self, seed=None):
        self._env.reset(seed=seed)
        return [seed]

    def _compress_info(self, info):
        if isinstance(info, dict) and any(isinstance(i, dict) for i in info.values()):
            return {f"{key}/{k}": v for key, i in info.items() for k, v in i.items()}
        return info

    def reset(self, *args, **kwargs):
        obss, info = self._env.reset(*args, **kwargs)
        return obss, self._compress_info(info)

    def render(self):
        return self._env.render()

    def step(self, actions):
        obss, rews, done, truncated, info = self._env.step(actions)
        return obss, rews, done, truncated, self._compress_info(info)

    def close(self):
        return self._env.close()

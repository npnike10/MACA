import copy
import gym
import numpy as np
from collections.abc import Iterable
from harl.envs.gym.lbforaging_custom import register_custom_lbforaging_envs

try:
    import gymnasium
except ImportError:
    gymnasium = None


class GYMEnv:
    def __init__(self, args):
        self.args = copy.deepcopy(args)
        self.scenario = self.args.pop("scenario")
        self.lbforaging = self._is_lbforaging_scenario(self.scenario)
        self.vmas = self._is_vmas_scenario(self.scenario)
        if self.lbforaging:
            self.args.pop("n_agents", None)
            register_custom_lbforaging_envs()
            # DISSCv2 configs use "lbforaging:<env-id>" while custom registration uses "<env-id>".
            self.scenario = self.scenario.split(":", 1)[-1]
            if gymnasium is None:
                raise ImportError(
                    "gymnasium is required for lbforaging scenarios. Install gymnasium and lbforaging."
                )
            self.env = gymnasium.make(self.scenario, **self.args)
        elif self.vmas:
            from harl.envs.gym.vmas_wrapper import VMASWrapper

            if gymnasium is None:
                raise ImportError(
                    "gymnasium is required for VMAS scenarios. Install gymnasium and vmas[gymnasium]."
                )
            self.env = VMASWrapper(
                env_name=self._parse_vmas_env_name(self.scenario), **self.args
            )
        else:
            self.args.pop("n_agents", None)
            self.env = gym.make(self.scenario, **self.args)
        self.n_agents = 1
        self.share_observation_space = [self.env.observation_space]
        self.observation_space = [self.env.observation_space]
        self.action_space = [self.env.action_space]
        if self.env.action_space.__class__.__name__ == "Box":
            self.discrete = False
        else:
            self.discrete = True
        if self.lbforaging or self.vmas:
            self.observation_space = self._space_to_list(self.env.observation_space)
            self.share_observation_space = self.observation_space
            self.action_space = self._space_to_list(self.env.action_space)
            self.n_agents = len(self.observation_space)
            self.discrete = True

    @staticmethod
    def _is_lbforaging_scenario(scenario):
        return scenario.startswith("lbforaging") or scenario.startswith("Foraging-")

    @staticmethod
    def _is_vmas_scenario(scenario):
        return scenario.startswith("vmas-") or scenario.startswith("vmas:")

    @staticmethod
    def _parse_vmas_env_name(scenario):
        if scenario.startswith("vmas-"):
            return scenario.split("vmas-", 1)[1]
        if scenario.startswith("vmas:"):
            return scenario.split(":", 1)[1]
        return scenario

    @staticmethod
    def _space_to_list(space):
        if hasattr(space, "spaces"):
            return list(space.spaces)
        if isinstance(space, (list, tuple)):
            return list(space)
        return [space]

    @staticmethod
    def _reset_unpack(reset_output):
        if isinstance(reset_output, tuple) and len(reset_output) == 2:
            return reset_output[0]
        return reset_output

    def step(self, actions):
        """
        return local_obs, global_state, rewards, dones, infos, available_actions
        """
        if self.lbforaging or self.vmas:
            step_out = self.env.step([int(a) for a in actions.flatten()])
            if self.vmas:
                obs, rew, done, truncated, info = step_out
                if done and truncated:
                    info["bad_transition"] = True
            elif len(step_out) == 5:
                obs, rew, terminated, truncated, info = step_out
                done = np.logical_or(terminated, truncated)
                if np.all(done) and np.all(np.asarray(truncated)):
                    info["bad_transition"] = True
            else:
                obs, rew, done, info = step_out
                if np.all(done) and info.get("TimeLimit.truncated", False):
                    info["bad_transition"] = True
            if not isinstance(done, Iterable):
                done = [done] * self.n_agents
            rew = [[float(np.sum(rew))]] * self.n_agents
            return obs, obs, rew, done, [info], self.get_avail_actions()

        if self.discrete:
            step_out = self.env.step(actions.flatten()[0])
        else:
            step_out = self.env.step(actions[0])
        if len(step_out) == 5:
            obs, rew, terminated, truncated, info = step_out
            done = terminated or truncated
            if done and truncated:
                info["bad_transition"] = True
        else:
            obs, rew, done, info = step_out
            if done and info.get("TimeLimit.truncated", False):
                info["bad_transition"] = True
        return [obs], [obs], [[rew]], [done], [info], self.get_avail_actions()

    def reset(self):
        """Returns initial observations and states"""
        reset_out = self.env.reset()
        obs = [self._reset_unpack(reset_out)]
        s_obs = copy.deepcopy(obs)
        if self.lbforaging or self.vmas:
            return obs[0], s_obs[0], self.get_avail_actions()
        return obs, s_obs, self.get_avail_actions()

    def get_avail_actions(self):
        if self.lbforaging or self.vmas:
            return [
                [1] * self.action_space[agent_id].n for agent_id in range(self.n_agents)
            ]
        if self.discrete:
            avail_actions = [[1] * self.action_space[0].n]
            return avail_actions
        else:
            return None

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    def seed(self, seed):
        if hasattr(self.env, "seed"):
            self.env.seed(seed)
        else:
            self.env.reset(seed=seed)

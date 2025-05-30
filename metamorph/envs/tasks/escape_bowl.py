import numpy as np
from gym import utils

from metamorph.config import cfg
from metamorph.envs.modules.agent import Agent
from metamorph.envs.modules.bowl import Bowl
from metamorph.envs.tasks.unimal import UnimalEnv
from metamorph.envs.wrappers.hfield import HfieldObs2D
from metamorph.envs.wrappers.hfield import StandReward
from metamorph.envs.wrappers.hfield import TerminateOnFalling
from metamorph.envs.wrappers.hfield import TerminateOnEscape
from metamorph.envs.wrappers.hfield import UnimalHeightObs
from metamorph.envs.wrappers.multi_env_wrapper import MultiUnimalNodeCentricAction
from metamorph.envs.wrappers.multi_env_wrapper import MultiUnimalNodeCentricObservation


class EscapeBowlTask(UnimalEnv, utils.EzPickle):
    def __init__(self, xml_str, unimal_id, kwargs):
        UnimalEnv.__init__(self, xml_str, unimal_id, kwargs)

    ###########################################################################
    # Sim step and reset
    ###########################################################################

    def step(self, action):
        xy_pos_before = self.sim.data.get_body_xpos("torso/0")[:2].copy()
        self.do_simulation(action)
        xy_pos_after = self.sim.data.get_body_xpos("torso/0")[:2].copy()

        # Give reward if distance from initial position has increased
        reward_forward = (
            np.linalg.norm(xy_pos_after) - np.linalg.norm(xy_pos_before)
        ) * cfg.ENV.FORWARD_REWARD_WEIGHT

        ctrl_cost = self.control_cost(action)
        reward = reward_forward - ctrl_cost
        observation = self._get_obs()

        info = {
            "x_pos": xy_pos_after[0],
            "y_pos": xy_pos_after[1],
            "xy_pos_before": xy_pos_before,
            "xy_pos_after": xy_pos_after,
            "__reward__energy": self.calculate_energy(),
            "__reward__ctrl": ctrl_cost,
            "__reward__forward": reward_forward,
            "metric": np.linalg.norm(xy_pos_after),
            "name": self.unimal_id
        }

        # Update viewer with markers, if any
        if self.viewer is not None:
            self.viewer._markers[:] = []
            for marker in self.metadata["markers"]:
                self.viewer.add_marker(**marker)

        return observation, reward, False, info


def make_env_escape_bowl(xml, unimal_id, kwargs={"corruption_level" :0}):
    env = EscapeBowlTask(xml, unimal_id, kwargs=kwargs)
    # Add modules
    for module in cfg.ENV.MODULES:
        env.add_module(globals()[module])
    env.reset()
    # Add all wrappers
    env = UnimalHeightObs(env)
    env = StandReward(env)
    env = TerminateOnFalling(env)
    env = HfieldObs2D(env)
    env = TerminateOnEscape(env)

    for wrapper_func in cfg.MODEL.WRAPPERS:
        env = globals()[wrapper_func](env)
    return env

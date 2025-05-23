import numpy as np
from gym import utils

from metamorph.config import cfg
from metamorph.envs.modules.agent import Agent
from metamorph.envs.modules.terrain import Terrain
from metamorph.envs.tasks.unimal import UnimalEnv
from metamorph.envs.wrappers.hfield import StandReward
from metamorph.envs.wrappers.hfield import TerminateOnFalling
from metamorph.envs.wrappers.hfield import TerminateOnTerrainEdge
from metamorph.envs.wrappers.hfield import UnimalHeightObs
from metamorph.envs.wrappers.multi_env_wrapper import MultiUnimalNodeCentricAction
from metamorph.envs.wrappers.multi_env_wrapper import MultiUnimalNodeCentricObservation


class InclineTask(UnimalEnv, utils.EzPickle):
    def __init__(self, xml_str, unimal_id, kwargs):
        UnimalEnv.__init__(self, xml_str, unimal_id, kwargs)

    ###########################################################################
    # Sim step and reset
    ###########################################################################

    def step(self, action):
        xy_pos_before = self.sim.data.get_body_xpos("torso/0")[:2].copy()
        self.do_simulation(action)
        xy_pos_after = self.sim.data.get_body_xpos("torso/0")[:2].copy()

        xy_vel = (xy_pos_after - xy_pos_before) / self.dt
        x_vel, y_vel = xy_vel

        angle = np.deg2rad(abs(cfg.TERRAIN.INCLINE_ANGLE))
        forward_reward = cfg.ENV.FORWARD_REWARD_WEIGHT * x_vel
        forward_reward = forward_reward / np.cos(angle)

        ctrl_cost = self.control_cost(action)
        reward = forward_reward - ctrl_cost
        observation = self._get_obs()

        start_x = np.cos(angle) * (-cfg.TERRAIN.SIZE[0] + 2.0)
        metric = round((xy_pos_after[0] - start_x) / np.cos(angle), 2)
        info = {
            "__reward__ctrl": ctrl_cost,
            "__reward__energy": self.calculate_energy(),
            "x_pos": xy_pos_after[0],
            "x_vel": x_vel,
            "xy_pos_before": xy_pos_before,
            "xy_pos_after": xy_pos_after,
            "__reward__forward": forward_reward,
            "name": self.unimal_id,
            "metric": metric,
        }

        return observation, reward, False, info


def make_env_incline(xml, unimal_id, kwargs={"corruption_level" :0}):
    env = InclineTask(xml, unimal_id, kwargs=kwargs)
    # Add modules
    for module in cfg.ENV.MODULES:
        env.add_module(globals()[module])
    # Reset is needed to setup observation spaces, sim etc which might be
    # needed by wrappers
    env.reset()
    # Add all wrappers
    env = UnimalHeightObs(env)
    env = StandReward(env)
    env = TerminateOnFalling(env)
    env = TerminateOnTerrainEdge(env)

    for wrapper_func in cfg.MODEL.WRAPPERS:
        env = globals()[wrapper_func](env)

    return env

from gym import utils
from gym.envs.mujoco import mujoco_env

from modular.utils import *
from modular.wrappers import *
from metamorph.config import cfg


class ModularEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, xml):
        self.xml = xml
        mujoco_env.MujocoEnv.__init__(self, xml, 4)
        utils.EzPickle.__init__(self)

        self.metadata['num_limbs'] = 4
        self.metadata['num_joints'] = 3

        self.agent_limb_names = self.model.body_names[1:]
        self.full_limb_names = ['torso', 'thigh', 'leg', 'lower_leg', 'foot']

    def step(self, a):
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter = self.sim.data.qpos[0]
        torso_height, torso_ang = self.sim.data.qpos[1:3]
        alive_bonus = 1.0
        reward = (posafter - posbefore) / self.dt
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        s = self.state_vector()
        done = not (
            np.isfinite(s).all()
            and (np.abs(s[2:]) < 100).all()
            and (torso_height > 0.6)
            and (abs(torso_ang) < 1)
        )
        ob = self._get_obs()

        info = {
            'name': 'hopper_4', 
        }

        return ob, reward, done, info

    def _get_obs(self):
        def _get_obs_per_limb(b):
            if b == "torso":
                limb_type_vec = np.array((1, 0, 0, 0))
            elif "thigh" in b:
                limb_type_vec = np.array((0, 1, 0, 0))
            elif "leg" in b:
                limb_type_vec = np.array((0, 0, 1, 0))
            elif "foot" in b:
                limb_type_vec = np.array((0, 0, 0, 1))
            else:
                limb_type_vec = np.array((0, 0, 0, 0))
            torso_x_pos = self.data.get_body_xpos("torso")[0]
            xpos = self.data.get_body_xpos(b)
            xpos[0] -= torso_x_pos
            q = self.data.get_body_xquat(b)
            expmap = quat2expmap(q)
            obs = np.concatenate(
                [
                    xpos,
                    np.clip(self.data.get_body_xvelp(b), -10, 10),
                    self.data.get_body_xvelr(b),
                    expmap,
                    limb_type_vec,
                ]
            )
            # include current joint angle and joint range as input
            if b == "torso":
                angle = 0.0
                joint_range = [0.0, 0.0]
            else:

                body_id = self.sim.model.body_name2id(b)
                jnt_adr = self.sim.model.body_jntadr[body_id]
                qpos_adr = self.sim.model.jnt_qposadr[
                    jnt_adr
                ]  # assuming each body has only one joint
                angle = np.degrees(
                    self.data.qpos[qpos_adr]
                )  # angle of current joint, scalar
                joint_range = np.degrees(
                    self.sim.model.jnt_range[jnt_adr]
                )  # range of current joint, (2,)
                # normalize
                angle = (angle - joint_range[0]) / (joint_range[1] - joint_range[0])
                joint_range[0] = (180.0 + joint_range[0]) / 360.0
                joint_range[1] = (180.0 + joint_range[1]) / 360.0
            obs = np.concatenate([obs, [angle], joint_range])
            return obs

        full_obs = np.concatenate(
            [_get_obs_per_limb(b) for b in self.model.body_names[1:]]
        )

        return full_obs.ravel()

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(
            low=-0.005, high=0.005, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=-0.005, high=0.005, size=self.model.nv
        )
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.75
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20


def make_env(xml):
    env = ModularEnv(xml)
    if cfg.ENV.MAKE_MODULAR_MATCH_UNIMAL:
        env = ModularMatchUnimalObservationPadding(env)
        env = ModularMatchUnimalActionPadding(env)

    elif cfg.MODEL.MLP.CONSISTENT_PADDING:
        env = ConsistentModularObservationPadding(env)
        env = ConsistentModularActionPadding(env)
    else:
        env = ModularObservationPadding(env)
        env = ModularActionPadding(env)
    return env
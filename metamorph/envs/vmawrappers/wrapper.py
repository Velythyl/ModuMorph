import os

import gym


import copy
import functools

import numpy as np

from metamorph.envs.vmawrappers.vma import VMA
from metamorph.utils import spaces as spu



@functools.lru_cache(maxsize=8192)
def _get_latents(vma, xml_path):
    # i messed up the code for mjcfconvert... i grabbed the bodies per joint, when i should have been grabbing the
    # bodies per body... the reason i did that is that in some urdf (not unimal) some bodies arent actuated so there was
    # no reason to render them
    #
    # the fix: when there's more than one joint per body, only the last one will have gotten rendered. So we grab it
    # and throw away the other one(s). In unimal, this means keeping the y limb if there's both a x and a y limb
    raw_latents = vma.get_latents_for_robot(xml_path)

    tokeep = {}
    for k, v in raw_latents.items():
        if "NO_ACTUATORS" in k:
            k = "torso/0"
        elif k.replace("limbx", "limby") in raw_latents and "limby" not in k:
            continue
        
            #continue
        body_key = k.replace("limbx", "limb").replace("limby", "limb")
        tokeep[body_key] = v

    return tokeep

def get_latents(vma, xml_path):
    return copy.deepcopy(_get_latents(vma, xml_path))


class VMAObsWrapper(gym.Wrapper):
    def __init__(self, env, vma_cache_dir, _set_vma_check_path, vma_to_proprioceptive, vma_to_context):
        super(VMAObsWrapper, self).__init__(env)
        assert(os.path.exists(vma_cache_dir))
        self.vma = VMA(None, vma_cache_dir, _set_vma_check_path=_set_vma_check_path)

        self.latent_size = self.vma.get_meta()["latent_size"]
        self.unshape_first_size = self.observation_space["obs_padding_mask"].shape[0]
        self.latent_matrix_size = self.latent_size * self.unshape_first_size

        self.vma_to_proprioceptive = vma_to_proprioceptive
        self.vma_to_context = vma_to_context

        delta_obs = {}
        if self.vma_to_proprioceptive:
            delta_obs["proprioceptive"] = (self.observation_space.spaces["proprioceptive"].shape[0] + self.latent_matrix_size,)
        if self.vma_to_context:
            delta_obs["context"] = (self.observation_space.spaces["context"].shape[0] + self.latent_matrix_size,)

        if len(delta_obs) > 0:
            self.observation_space = spu.update_obs_space(env, delta_obs)

    def add_vma_obs_to_obs(self, obs):
        if (not self.vma_to_proprioceptive) and (not self.vma_to_context):
            return obs

        # right now, this impl adds the vma obs as "bodies" and reuses the obs_padding_mask. But, we could also add them as "joints", and add reuse the action mask. This would enable the policy to see where each action gets effectuated (?)
        try:
            xml_path = self.metadata["xml_path"].split("/")[-1]
            latents = get_latents(self.vma, xml_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find the latent file for {self.metadata['xml_path']}. Please run the vma latent utility.")
        x = 0
        for limb_name in self.metadata["limb_name"]:
            assert limb_name in latents, f"Expected {limb_name} to be in the latents, but it was not found. Please check the latent file for {self.metadata['xml_path']}."
        vma_obs = [latents[limb_name] for limb_name in self.metadata["limb_name"]]

        for _ in range(self.unshape_first_size - len(vma_obs)):
            vma_obs.append(np.zeros_like(vma_obs[0]))
        vma_obs = np.vstack(vma_obs)

        def add_vma_obs_to_target(vma_obs, target):
            target = target.reshape(self.unshape_first_size, -1)
            target = np.concatenate((target, vma_obs), axis=1)
            return target.reshape(self.unshape_first_size * np.prod(target[0].shape))

        obs = copy.deepcopy(obs)
        if self.vma_to_proprioceptive:
            obs["proprioceptive"] = add_vma_obs_to_target(vma_obs, obs["proprioceptive"])
        if self.vma_to_context:
            obs["context"] = add_vma_obs_to_target(vma_obs, obs["context"])
        return obs

    def step(self, action):
        ret = super(VMAObsWrapper, self).step(action)
        obs = ret[0]
        return self.add_vma_obs_to_obs(obs), *ret[1:]

    def reset(self, **kwargs):
        ret = self.env.reset(**kwargs)
        if isinstance(ret, tuple):
            obs = ret[0]
            return self.add_vma_obs_to_obs(obs), *ret[1:]
        else:
            return self.add_vma_obs_to_obs(ret)
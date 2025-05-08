import argparse
import os
import sys

import torch
import torch.nn as nn
import numpy as np
import pickle
import json
import matplotlib.pyplot as plt

from metamorph.config import cfg
from metamorph.algos.ppo.ppo import PPO
from metamorph.algos.ppo.envs import *
from metamorph.utils import file as fu
from metamorph.utils import sample as su
from metamorph.utils import sweep as swu
from metamorph.envs.vec_env.pytorch_vec_env import VecPyTorch
from metamorph.algos.ppo.model import Agent

from tools.train_ppo import set_cfg_options
from utils.box_and_whisker_stats import box_and_whisker_stats
from utils.get_checkpoint_path import get_checkpoint_path
from utils.suppress_stderr import suppress_stderr

torch.manual_seed(0)


# evaluate on a single robot
def evaluate(policy, env):
    episode_return = np.zeros(cfg.PPO.NUM_ENVS)
    not_done = np.ones(cfg.PPO.NUM_ENVS)

    with suppress_stderr():
        obs = env.reset()

    for t in trange(2000, desc='Stepping env...'):

        with suppress_stderr():
            _, act, _, _, _ = policy.act(obs, return_attention=False, compute_val=False)
            obs, reward, done, infos = env.step(act)

        idx = np.where(done)[0]
        for i in idx:
            if not_done[i] == 1:
                not_done[i] = 0
                episode_return[i] = infos[i]['episode']['r']
        if not_done.sum() == 0:
            break

    return episode_return

from tqdm import tqdm, trange



def evaluate_model(model_path, agent_path, terminate_on_fall=True, deterministic=False):
    '''
    model_path: the path of the .pt model file to evaluate
    agent_path: the path of the test agents
    policy_folder: the path to the folder which contains config.yaml (should be the same folder as model_path)
    suffix: suffix in the name of the file to save evaluation results
    terminate_on_fall: whether to early stop an episode if the agent's height is below some threshold value
    deterministic: whether take a deterministic action (mean of the Gaussian action distribution) or a random action
    '''

    test_agents = list(set([x.split('.')[0] for x in os.listdir(f'{agent_path}/xml') if x.endswith('.xml')]))[:5]

    ENV_NAME = cfg.ENV_NAME
    policy_folder = "/".join(model_path.split('/')[:-1])
    print (policy_folder)
    cfg.merge_from_file(f'{policy_folder}/yacs_config.yaml')
    cfg.ENV_NAME = ENV_NAME
    cfg.PPO.CHECKPOINT_PATH = model_path
    cfg.ENV.WALKERS = []
    cfg.ENV.WALKER_DIR = agent_path
    cfg.OUT_DIR = './eval'
    cfg.TERMINATE_ON_FALL = terminate_on_fall
    cfg.DETERMINISTIC = deterministic
    cfg.PPO.NUM_ENVS = 32
    set_cfg_options()

    ppo_trainer = PPO()
    policy = ppo_trainer.agent
    # change to eval mode as we may have dropout in the model
    policy.ac.eval()

    EVAL_OUTPUT_FOLDER = f"{policy_folder}/eval"
    os.makedirs(EVAL_OUTPUT_FOLDER, exist_ok=True)
    output_name = agent_path.replace("/", "-")

    eval_result = {}

    # avg_score stores the per-agent average evaluation return
    avg_score = []
    for agent in tqdm(test_agents, desc="Evaluating different agents..."):
        i = test_agents.index(agent)
        if agent in eval_result:
            # do not repeat evaluation
            continue

        with suppress_stderr():
            envs = make_vec_envs(xml_file=agent, training=False, norm_rew=False, render_policy=True)
            set_ob_rms(envs, get_ob_rms(ppo_trainer.envs))

        episode_return = evaluate(policy, envs)
        envs.close()
        print (agent, f'{episode_return.mean():.2f} +- {episode_return.std():.2f}')
        eval_result[agent] = episode_return
        avg_score.append(np.array(episode_return).mean())

        with open(f'{EVAL_OUTPUT_FOLDER}/eval_{test_agents[i]}.pkl', 'wb') as f:
            pickle.dump(eval_result, f)

    print ('avg score across all test agents: ', np.array(avg_score).mean())
    return box_and_whisker_stats(avg_score)
    #return {f"{output_name}/{k}": v for k,v in stats.items()}

def post_train_evaluate(checkpoint_path, agent_type, agent_path):
    #if "unimal" in agent_type.lower():
    #    cfg.ENV_NAME = "Unimal-v0"
    #elif "modular" in agent_type.lower():
    #    cfg.ENV_NAME = "Modular-v0"

    if not checkpoint_path.endswith('.pt'):
        checkpoint_path = get_checkpoint_path(checkpoint_path, -1)

    ret = evaluate_model(checkpoint_path, agent_path, cfg.TERMINATE_ON_FALL, cfg.DETERMINISTIC)
    return {f"{agent_type}/{k}":v for k,v in ret.items()}

if __name__ == '__main__':

    # example command: python tools/evaluate.py --policy_path output/example --test_folder unimals_100/test --seed 1409
    parser = argparse.ArgumentParser(description="Evaluate a RL agent")
    parser.add_argument("--policy_path", default=None, type=str)
    parser.add_argument("--policy_name", default='Unimal-v0', type=str)
    parser.add_argument("--terminate_on_fall", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--seed", default=None, type=int)
    parser.add_argument("--test_folder", default='unimals_100/test', type=str)
    args = parser.parse_args()

    path_of_latest_checkpoint = get_checkpoint_path(args.policy_path, -1)
    post_train_evaluate(path_of_latest_checkpoint, "modular", "modular/all_train")

    suffix = []
    if args.terminate_on_fall:
        suffix.append('terminate_on_fall')
    if args.deterministic:
        suffix.append('deterministic')
    if '/' in args.test_folder:
        suffix.append(args.test_folder.split('/')[1])
    else:
        suffix.append(args.test_folder)
    if 'checkpoint' in args.policy_name:
        iteration = args.policy_name.split('_')[1]
        suffix.append(f'cp_{iteration}')
    if len(suffix) == 0:
        suffix = None
    else:
        suffix = '_'.join(suffix)
    print (suffix)

    if args.seed is not None:
        seeds = [str(args.seed)]
    else:
        seeds = ['1409', '1410', '1411']

    policy_path = args.policy_path
    # `scores` saves each agent's average return in each seed
    path_of_latest_checkpoint = get_checkpoint_path(policy_path, -1)

    scores = []
    for seed in seeds:
        model_path = path_of_latest_checkpoint
        score = evaluate_model(model_path, args.test_folder, terminate_on_fall=args.terminate_on_fall, deterministic=args.deterministic)
        scores.append(score)
    scores = np.stack(scores)
    print ('avg score across seeds: ')
    test_agents = [x.split('.')[0] for x in os.listdir(f'{args.test_folder}/xml')]
    for i, agent in enumerate(test_agents):
        print (f'{agent}: {scores[:, i].mean()} +- {scores[:, i].std()}')
    scores = scores.mean(axis=1)
    print (f'overall: {scores.mean()} +- {scores.std()}')

    exit()
    # example command: python tools/evaluate.py --policy_path output/example --test_folder unimals_100/test --seed 1409
    parser = argparse.ArgumentParser(description="Evaluate a RL agent")
    parser.add_argument("--policy_path", default=None, type=str)
    parser.add_argument("--policy_name", default='Unimal-v0', type=str)
    parser.add_argument("--terminate_on_fall", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--seed", default=None, type=int)
    parser.add_argument("--test_folder", default='unimals_100/test', type=str)
    args = parser.parse_args()

    suffix = []
    if args.terminate_on_fall:
        suffix.append('terminate_on_fall')
    if args.deterministic:
        suffix.append('deterministic')
    if '/' in args.test_folder:
        suffix.append(args.test_folder.split('/')[1])
    else:
        suffix.append(args.test_folder)
    if 'checkpoint' in args.policy_name:
        iteration = args.policy_name.split('_')[1]
        suffix.append(f'cp_{iteration}')
    if len(suffix) == 0:
        suffix = None
    else:
        suffix = '_'.join(suffix)
    print (suffix)

    if args.seed is not None:
        seeds = [str(args.seed)]
    else:
        seeds = ['1409', '1410', '1411']

    policy_path = args.policy_path
    # `scores` saves each agent's average return in each seed
    scores = []
    for seed in seeds:
        model_path = os.path.join(policy_path, seed, args.policy_name + '.pt')
        score = evaluate_model(model_path, args.test_folder, os.path.join(policy_path, seed), suffix=suffix, terminate_on_fall=args.terminate_on_fall, deterministic=args.deterministic)
        scores.append(score)
    scores = np.stack(scores)
    print ('avg score across seeds: ')
    test_agents = [x.split('.')[0] for x in os.listdir(f'{args.test_folder}/xml')]
    for i, agent in enumerate(test_agents):
        print (f'{agent}: {scores[:, i].mean()} +- {scores[:, i].std()}')
    scores = scores.mean(axis=1)
    print (f'overall: {scores.mean()} +- {scores.std()}')

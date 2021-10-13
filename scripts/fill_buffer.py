import os
import sys
import time
import copy
import math
import collections
from tqdm import tqdm

import torch

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
sys.path.append('./')
sys.path.append('..')
from scripts.create_agent import createAgent
from utils.parameters import *
from storage.buffer import QLearningBufferExpert, QLearningBuffer
from helping_hands_rl_envs import env_factory
from utils.logger import Logger
from utils.schedules import LinearSchedule
from utils.torch_utils import rand_perlin_2d
from utils.env_wrapper import EnvWrapper

ExpertTransition = collections.namedtuple('ExpertTransition', 'state obs action reward next_state next_obs done step_left expert')

# hard code the num_objects and max_episode_steps for public
if env == 'block_stacking':
    buffer_size = 50
elif env == 'bottle_tray':
    buffer_size = 240
elif env == 'house_building_4':
    buffer_size = 200
elif env == 'covid_test':
    buffer_size = 2000
elif env == 'box_palletizing':
    buffer_size = 1000
elif env == 'block_bin_packing':
    buffer_size = 2000

elif env == 'bumpy_house_building_4':
    buffer_size = 2000
elif env == 'bumpy_box_palletizing':
    buffer_size = 2000

def set_seed(s):
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def addPerlinNoiseToObs(obs, c):
    for i in range(obs.size(0)):
        obs[i, 0] += (c * rand_perlin_2d((90, 90), (
            int(np.random.choice([1, 2, 3, 5, 6], 1)[0]),
            int(np.random.choice([1, 2, 3, 5, 6], 1)[0]))) + c)

def addPerlinNoiseToInHand(in_hand, c):
    if in_hand_mode != 'proj':
        for i in range(in_hand.size(0)):
            in_hand[i, 0] += (c * rand_perlin_2d((24, 24), (
                int(np.random.choice([1, 2], 1)[0]),
                int(np.random.choice([1, 2], 1)[0]))) + c)

def getCurrentObs(in_hand, obs):
    obss = []
    for i, o in enumerate(obs):
        obss.append((o.squeeze(), in_hand[i].squeeze()))
    return obss

def fillBuffer():
    plt.style.use('default')
    # env_config['render'] = False
    envs = EnvWrapper(num_processes, simulator, env, env_config, planner_config)
    agent = createAgent()
    replay_buffer = QLearningBuffer(buffer_size)
    # logging
    log_dir = os.path.join(log_pre, 'buffer')
    log_sub = env
    logger = Logger(log_dir, env, 'train', num_processes, max_episode, log_sub)

    agent.eval()
    states, in_hands, obs = envs.reset()
    prev_steps_lefts = envs.getStepLeft()
    valid = torch.ones_like(prev_steps_lefts, dtype=torch.bool)
    total = 0
    s = 0
    buffer_len = 0
    step_times = []
    pbar = tqdm(total=buffer_size)
    while len(replay_buffer) < buffer_size:
        buffer_obs = getCurrentObs(in_hands, obs)
        plan_actions = envs.getNextAction()
        actions_star_idx, actions_star = agent.getActionFromPlan(plan_actions)
        actions_star = torch.cat((actions_star, states.unsqueeze(1)), dim=1)
        t0 = time.time()
        states_, in_hands_, obs_, rewards, dones = envs.step(actions_star, auto_reset=False)
        t = time.time() - t0
        step_times.append(t)

        steps_lefts = envs.getStepLeft()

        done_idxes = torch.nonzero(dones).squeeze(1)
        if done_idxes.shape[0] != 0:
            reset_states_, reset_in_hands_, reset_obs_ = envs.reset_envs(done_idxes)
            for j, idx in enumerate(done_idxes):
                states_[idx] = reset_states_[j]
                in_hands_[idx] = reset_in_hands_[j]
                obs_[idx] = reset_obs_[j]

        buffer_obs_ = getCurrentObs(in_hands_, obs_)

        for i in range(num_processes):
            if valid[i]:
                replay_buffer.add(
                    ExpertTransition(states[i], buffer_obs[i], actions_star_idx[i], rewards[i], states_[i],
                                     buffer_obs_[i], dones[i], steps_lefts[i], torch.tensor(1))
                )

        if dones.sum():
            s += rewards.sum().int().item()
            total += dones.sum().int().item()

        pbar.set_description(
            '{}/{}, SR: {:.3f}, step time: {:.2f}; avg step time: {:.2f}'
                .format(s, total, float(s) / total if total != 0 else 0, t, np.mean(step_times))
        )
        pbar.update(len(replay_buffer) - buffer_len)
        buffer_len = len(replay_buffer)
        states = copy.copy(states_)
        obs = copy.copy(obs_)
        in_hands = copy.copy(in_hands_)
        prev_steps_lefts = envs.getStepLeft()

    logger.saveBuffer(replay_buffer)
    envs.close()


if __name__ == '__main__':
    fillBuffer()
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

def fillDeconstruct():
    def states_valid(states_list):
        if len(states_list) < 2:
            return False
        for i in range(1, len(states_list)):
            if states_list[i] != 1 - states_list[i-1]:
                return False
        return True

    def rewards_valid(reward_list):
        if reward_list[0] != 1:
            return False
        for i in range(1, len(reward_list)):
            if reward_list[i] != 0:
                return False
        return True

    plt.style.use('default')
    envs = EnvWrapper(num_processes, simulator, env, env_config, planner_config)
    agent = createAgent()
    replay_buffer = QLearningBuffer(buffer_size)
    # logging
    log_dir = os.path.join(log_pre, '{}_deconstruct_{}_{}_{}_{}'.format(alg, model, simulator, num_objects,
                                                                          max_episode_steps))
    logger = Logger(log_dir, env, 'train', num_processes, max_episode, log_sub)

    agent.eval()
    states, in_hands, obs = envs.reset()
    total = 0
    s = 0
    step_times = []
    steps = [0 for i in range(num_processes)]
    local_state = [[] for i in range(num_processes)]
    local_obs = [[] for i in range(num_processes)]
    local_action = [[] for i in range(num_processes)]
    local_reward = [[] for i in range(num_processes)]

    pbar = tqdm(total=buffer_size)
    buffer_len = 0
    while len(replay_buffer) < buffer_size:
        # buffer_obs = agent.getCurrentObs(in_hands, obs)
        plan_actions = envs.getNextAction()
        actions_star_idx, actions_star = agent.getActionFromPlan(plan_actions)
        actions_star = torch.cat((actions_star, states.unsqueeze(1)), dim=1)
        t0 = time.time()
        states_, in_hands_, obs_, rewards, dones = envs.step(actions_star, auto_reset=False)
        state_id = action_sequence.find('p')
        dones[actions_star[:, state_id] + states_ != 1] = 1
        t = time.time()-t0
        step_times.append(t)


        buffer_obs = getCurrentObs(in_hands_, obs)
        for i in range(num_processes):
            local_state[i].append(states[i])
            local_obs[i].append(buffer_obs[i])
            local_action[i].append(actions_star_idx[i])
            local_reward[i].append(rewards[i])

        steps = list(map(lambda x: x + 1, steps))

        done_idxes = torch.nonzero(dones).squeeze(1)
        if done_idxes.shape[0] != 0:
            empty_in_hands = envs.getEmptyInHand()

            buffer_obs_ = getCurrentObs(empty_in_hands, copy.deepcopy(obs_))
            reset_states_, reset_in_hands_, reset_obs_ = envs.reset_envs(done_idxes)
            for i, idx in enumerate(done_idxes):
                local_obs[idx].append(buffer_obs_[idx])
                local_state[idx].append(copy.deepcopy(states_[idx]))
                if (num_objects-2)*2 <= steps[idx] <= num_objects*2 and states_valid(local_state[idx]) and rewards_valid(local_reward[idx]):
                    s += 1
                    for j in range(len(local_reward[idx])):
                        obs = local_obs[idx][j+1]
                        next_obs = local_obs[idx][j]
                        if perlin > 0:
                            obs_w_perlin = obs[0] + (perlin * rand_perlin_2d((90, 90), (
                                int(np.random.choice([1, 2, 3, 5, 6], 1)[0]),
                                int(np.random.choice([1, 2, 3, 5, 6], 1)[0]))) + perlin)
                            next_obs_w_perlin = next_obs[0] + (perlin * rand_perlin_2d((90, 90), (
                                int(np.random.choice([1, 2, 3, 5, 6], 1)[0]),
                                int(np.random.choice([1, 2, 3, 5, 6], 1)[0]))) + perlin)
                            if in_hand_mode != 'proj':
                                in_hand_w_perlin = obs[1] + (
                                        perlin * rand_perlin_2d((24, 24), (int(np.random.choice([1, 2], 1)[0]),
                                                                             int(np.random.choice([1, 2], 1)[
                                                                                     0]))) + perlin)
                                next_in_hand_w_perlin = next_obs[1] + (
                                        perlin * rand_perlin_2d((24, 24), (int(np.random.choice([1, 2], 1)[0]),
                                                                             int(np.random.choice([1, 2], 1)[
                                                                                     0]))) + perlin)
                                obs = (obs_w_perlin, in_hand_w_perlin)
                                next_obs = (next_obs_w_perlin, next_in_hand_w_perlin)
                            else:
                                obs = (obs_w_perlin, obs[1])
                                next_obs = (next_obs_w_perlin, next_obs[1])

                        replay_buffer.add(ExpertTransition(local_state[idx][j+1],
                                                           obs,
                                                           local_action[idx][j],
                                                           local_reward[idx][j],
                                                           local_state[idx][j],
                                                           next_obs,
                                                           torch.tensor(float(j == 0)),
                                                           torch.tensor(float(j)),
                                                           torch.tensor(1)))

                states_[idx] = reset_states_[i]
                obs_[idx] = reset_obs_[i]

                total += 1
                steps[idx] = 0
                local_state[idx] = []
                local_obs[idx] = []
                local_action[idx] = []
                local_reward[idx] = []

        pbar.set_description(
            '{}/{}, SR: {:.3f}, step time: {:.2f}; avg step time: {:.2f}'
            .format(s, total, float(s)/total if total !=0 else 0, t, np.mean(step_times))
        )
        pbar.update(len(replay_buffer) - buffer_len)
        buffer_len = len(replay_buffer)

        states = copy.copy(states_)
        obs = copy.copy(obs_)

    logger.saveBuffer(replay_buffer)


if __name__ == '__main__':
    fillDeconstruct()
import numpy as np
import torch
import torch.nn.functional as F
from agents.agents_6d.dqn_6d_asr_5l import DQN6DASR5L

class Policy6DASR5L(DQN6DASR5L):
    def __init__(self, workspace, heightmap_size, device, lr=1e-4, gamma=0.9, sl=False, num_primitives=1,
                 patch_size=24, num_rz=8, rz_range=(0, 7*np.pi/8), num_ry=8, ry_range=(0, 7*np.pi/8), num_rx=8,
                 rx_range=(0, 7*np.pi/8), num_zs=16, z_range=(0.02, 0.12)):
        super().__init__(workspace, heightmap_size, device, lr, gamma, sl, num_primitives, patch_size, num_rz,
                         rz_range, num_ry, ry_range, num_rx, rx_range, num_zs, z_range)

    def update(self, batch):
        self._loadBatchToDevice(batch)
        batch_size, states, obs, action_idx, rewards, next_states, next_obs, non_final_masks, step_lefts, is_experts = self._loadLossCalcDict()
        batch_size = states.size(0)
        heightmap_size = obs[0].size(2)

        pixel = action_idx[:, 0:2]
        a3_idx = action_idx[:, 2]
        a2_idx = action_idx[:, 3]
        a4_idx = action_idx[:, 4]
        a5_idx = action_idx[:, 5]

        q1_output, obs_encoding = self.forwardFCN(states, obs[1], obs[0])
        q1_output = q1_output.reshape(batch_size, -1)
        q1_target = action_idx[:, 0] * heightmap_size + action_idx[:, 1]
        q1_loss = F.cross_entropy(q1_output, q1_target)

        q2_output = self.forwardQ2(states, obs[1], obs[0], obs_encoding, pixel)
        q2_output = q2_output.reshape(batch_size, -1)
        q2_target = a2_idx
        q2_loss = F.cross_entropy(q2_output, q2_target)

        q3_output = self.forwardQ3(states, obs[1], obs[0], obs_encoding, pixel, a2_idx)
        q3_output = q3_output.reshape(batch_size, -1)
        q3_target = a3_idx
        q3_loss = F.cross_entropy(q3_output, q3_target)

        q4_output = self.forwardQ4(states, obs[1], obs[0], obs_encoding, pixel, a2_idx, a3_idx)
        q4_output = q4_output.reshape(batch_size, -1)
        q4_target = a4_idx
        q4_loss = F.cross_entropy(q4_output, q4_target)

        q5_output = self.forwardQ5(states, obs[1], obs[0], obs_encoding, pixel, a2_idx, a3_idx, a4_idx)
        q5_output = q5_output.reshape(batch_size, -1)
        q5_target = a5_idx
        q5_loss = F.cross_entropy(q5_output, q5_target)

        loss = q1_loss + q2_loss + q3_loss + q4_loss + q5_loss

        self.fcn_optimizer.zero_grad()
        self.q2_optimizer.zero_grad()
        self.q3_optimizer.zero_grad()
        self.q4_optimizer.zero_grad()
        self.q5_optimizer.zero_grad()

        loss.backward()

        for param in self.fcn.parameters():
            param.grad.data.clamp_(-1, 1)
        self.fcn_optimizer.step()

        for param in self.q2.parameters():
            param.grad.data.clamp_(-1, 1)
        self.q2_optimizer.step()

        for param in self.q3.parameters():
            param.grad.data.clamp_(-1, 1)
        self.q3_optimizer.step()

        for param in self.q4.parameters():
            param.grad.data.clamp_(-1, 1)
        self.q4_optimizer.step()

        for param in self.q5.parameters():
            param.grad.data.clamp_(-1, 1)
        self.q5_optimizer.step()

        self.loss_calc_dict = {}

        return (q1_loss.item(), q2_loss.item(), q3_loss.item(), q4_loss.item(), q5_loss.item()), torch.tensor(0.)

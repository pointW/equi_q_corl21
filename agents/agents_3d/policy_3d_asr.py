import numpy as np
import torch
import torch.nn.functional as F
from agents.agents_3d.dqn_3d_asr import DQN3DASR

class Policy3DASR(DQN3DASR):
    def __init__(self, workspace, heightmap_size, device, lr=1e-4, gamma=0.9, sl=False, num_primitives=1,
                 patch_size=24, num_rz=8, rz_range=(0, 7 * np.pi / 8)):
        super().__init__(workspace, heightmap_size, device, lr, gamma, sl, num_primitives, patch_size, num_rz, rz_range)

    def update(self, batch):
        self._loadBatchToDevice(batch)
        batch_size, states, obs, action_idx, rewards, next_states, next_obs, non_final_masks, step_lefts, is_experts = self._loadLossCalcDict()

        pixel = action_idx[:, 0:2]
        a2_idx = action_idx[:, 2]

        q1_output, obs_encoding = self.forwardFCN(states, obs[1], obs[0])
        q1_output = q1_output.reshape(batch_size, -1)
        q1_target = action_idx[:, 0] * self.heightmap_size + action_idx[:, 1]
        q1_loss = F.cross_entropy(q1_output, q1_target)

        q2_output = self.forwardQ2(states, obs[1], obs[0], obs_encoding, pixel)
        q2_output = q2_output.reshape(batch_size, -1)
        q2_target = a2_idx
        q2_loss = F.cross_entropy(q2_output, q2_target)

        loss = q1_loss + q2_loss

        self.fcn_optimizer.zero_grad()
        self.q2_optimizer.zero_grad()
        loss.backward()

        for param in self.fcn.parameters():
            param.grad.data.clamp_(-1, 1)
        self.fcn_optimizer.step()

        for param in self.q2.parameters():
            param.grad.data.clamp_(-1, 1)
        self.q2_optimizer.step()

        self.loss_calc_dict = {}

        return (q1_loss.item(), q2_loss.item()), torch.tensor(0.)
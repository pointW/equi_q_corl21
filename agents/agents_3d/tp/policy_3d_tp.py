import numpy as np
import torch
import torch.nn.functional as F
from agents.agents_3d.tp.dqn_3d_tp import DQN3DTP

class Policy3DTP(DQN3DTP):
    def __init__(self, workspace, heightmap_size, device, lr=1e-4, gamma=0.9, sl=False, num_primitives=1,
                 patch_size=24, num_rz=8, rz_range=(0, 7*np.pi/8)):
        super().__init__(workspace, heightmap_size, device, lr, gamma, sl, num_primitives, patch_size, num_rz, rz_range)

    # def update(self, batch):
    #     batch_size = len(batch)
    #     divide_factor = batch_size
    #     small_batch_size = batch_size//divide_factor
    #     loss = 0
    #     for i in range(divide_factor):
    #         small_batch = batch[small_batch_size*i:small_batch_size*(i+1)]
    #         self._loadBatchToDevice(small_batch)
    #         _, states, obs, action_idx, rewards, next_states, next_obs, non_final_masks, step_lefts, is_experts = self._loadLossCalcDict()
    #         output = self.forwardFCN(states, obs[1], obs[0])
    #         output = output.reshape(small_batch_size, -1)
    #         target = action_idx[:, 2] * self.heightmap_size * self.heightmap_size + \
    #                  action_idx[:, 0] * self.heightmap_size + \
    #                  action_idx[:, 1]
    #         loss += F.cross_entropy(output, target)/divide_factor
    #         self.loss_calc_dict = {}
    #
    #     self.place_optimizer.zero_grad()
    #     self.pick_optimizer.zero_grad()
    #     loss.backward()
    #     self.place_optimizer.step()
    #     self.pick_optimizer.step()
    #
    #     self.loss_calc_dict = {}
    #
    #     return loss.item(), torch.tensor(0.)


    def update(self, batch):
        batch_size = len(batch)
        self._loadBatchToDevice(batch)
        _, states, obs, action_idx, rewards, next_states, next_obs, non_final_masks, step_lefts, is_experts = self._loadLossCalcDict()
        obs, in_hands = obs
        mask_0 = states == 0
        mask_1 = states == 1
        if mask_0.sum():
            obs_0 = obs[mask_0]
            action_idx_0 = action_idx[mask_0]

            pick_loss = 0
            for i in range(mask_0.sum()):
                output = self.forwardPick(obs_0[i:i+1]).reshape(1, -1)
                target = action_idx_0[i:i+1, 2] * self.heightmap_size * self.heightmap_size + \
                         action_idx_0[i:i+1, 0] * self.heightmap_size + \
                         action_idx_0[i:i+1, 1]
                pick_loss += F.cross_entropy(output, target)/mask_0.sum()

            self.pick_optimizer.zero_grad()
            pick_loss.backward()
            self.pick_optimizer.step()
        else:
            pick_loss = torch.tensor(0)
        if mask_1.sum():
            obs_1 = obs[mask_1]
            in_hands_1 = in_hands[mask_1]
            action_idx_1 = action_idx[mask_1]

            place_loss = 0
            for i in range(mask_1.sum()):
                output = self.forwardPlace(in_hands_1[i:i+1], obs_1[i:i+1]).reshape(1, -1)
                target = action_idx_1[i:i+1, 2] * self.heightmap_size * self.heightmap_size + \
                         action_idx_1[i:i+1, 0] * self.heightmap_size + \
                         action_idx_1[i:i+1, 1]
                place_loss += F.cross_entropy(output, target)/mask_1.sum()

            self.place_optimizer.zero_grad()
            place_loss.backward()
            self.place_optimizer.step()
        else:
            place_loss = torch.tensor(0)

        self.loss_calc_dict = {}

        return (pick_loss.item(), place_loss.item()), torch.tensor(0.)
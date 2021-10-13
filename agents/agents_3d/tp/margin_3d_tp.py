import numpy as np
import torch
import torch.nn.functional as F
from agents.agents_3d.tp.dqn_3d_tp import DQN3DTP
from agents.margin_base import MarginBase

class Margin3DTP(DQN3DTP, MarginBase):
    def __init__(self, workspace, heightmap_size, device, lr=1e-4, gamma=0.9, sl=False, num_primitives=1,
                 patch_size=24, num_rz=8, rz_range=(0, 7*np.pi/8), margin='l', margin_l=0.1, margin_weight=0.1,
                 softmax_beta=100):
        DQN3DTP.__init__(self, workspace, heightmap_size, device, lr, gamma, sl, num_primitives, patch_size, num_rz, rz_range)
        MarginBase.__init__(self, margin, margin_l, margin_weight, softmax_beta)

    def calcMarginLoss(self):
        batch_size, states, obs, action_idx, rewards, next_states, next_obs, non_final_masks, step_lefts, is_experts = self._loadLossCalcDict()
        q1_output = self.loss_calc_dict['q1_output']
        action_idx_dense = action_idx[:, 2] * self.heightmap_size * self.heightmap_size + \
                           action_idx[:, 0] * self.heightmap_size + \
                           action_idx[:, 1]
        return self.getMarginLossSingle(q1_output.reshape(q1_output.size(0), -1), action_idx_dense, is_experts, True)

    def update(self, batch):
        batch_size = len(batch)
        pick_loss = torch.tensor(0.).to(self.device)
        place_loss = torch.tensor(0.).to(self.device)
        td_errors = []
        for i in range(batch_size):
            small_batch = batch[i:i+1]
            self._loadBatchToDevice(small_batch)
            _, state, obs, action_idx, reward, next_state, next_obs, non_final_mask, step_left, is_expert = self._loadLossCalcDict()
            with torch.no_grad():
                q_map_prime = self.forwardFCN(next_state, next_obs[1], next_obs[0], target_net=True)
                q_prime = q_map_prime.reshape((1, -1)).max(1)[0]
                q_target = reward + self.gamma * q_prime * non_final_mask

            if state == 0:
                output = self.forwardPick(obs[0])
            else:
                output = self.forwardPlace(obs[1], obs[0])

            q_pred = output[0, action_idx[:, 2], action_idx[:, 0], action_idx[:, 1]]

            td_loss = F.smooth_l1_loss(q_pred, q_target)
            with torch.no_grad():
                td_error = torch.abs(q_pred - q_target)
                td_errors.append(td_error)

            action_idx_dense = action_idx[:, 2] * self.heightmap_size * self.heightmap_size + \
                               action_idx[:, 0] * self.heightmap_size + \
                               action_idx[:, 1]
            margin_loss = self.getMarginLossSingle(output.reshape(output.size(0), -1), action_idx_dense, is_expert, True)

            loss = td_loss + self.margin_weight*margin_loss
            if state == 0:
                pick_loss += loss/batch_size
            else:
                place_loss += loss/batch_size

        if pick_loss != 0:
            self.pick_optimizer.zero_grad()
            pick_loss.backward()
            self.pick_optimizer.step()
        if place_loss != 0:
            self.place_optimizer.zero_grad()
            place_loss.backward()
            self.place_optimizer.step()

        return (pick_loss.item(), place_loss.item()), torch.cat(td_errors)



import numpy as np
import torch
import torch.nn.functional as F
from agents.agents_3d.dqn_3d_fcn_si import DQN3DFCNSingleIn
from agents.agents_3d.base_3d_aug import Base3DAug
from utils.torch_utils import getDrQAugmentedTransition

class DQN3DFCNSingleInDrQ(DQN3DFCNSingleIn):
    def __init__(self, workspace, heightmap_size, device, lr=1e-4, gamma=0.9, sl=False, num_primitives=1,
                 patch_size=24, num_rz=8, rz_range=(0, 7 * np.pi / 8)):
        DQN3DFCNSingleIn.__init__(self, workspace, heightmap_size, device, lr, gamma, sl, num_primitives, patch_size, num_rz, rz_range)
        self.K = 2
        self.M = 2
        self.aug_type = 'cn'

    def calcTDLoss(self):
        batch_size, states, obs, action_idx, rewards, next_states, next_obs, non_final_masks, step_lefts, is_experts = self._loadLossCalcDict()
        if self.sl:
            q_targets = self.gamma ** step_lefts
        else:
            with torch.no_grad():
                q_targets = []
                for _ in range(self.K):
                    aug_next_obss = []
                    for i in range(batch_size):
                        aug_next_obs, _ = getDrQAugmentedTransition(next_obs[0][i, 0].cpu().numpy(), action_idx=None, rzs=self.rzs, aug_type=self.aug_type)
                        aug_next_obss.append(torch.tensor(aug_next_obs.reshape(1, 1, *aug_next_obs.shape)))
                    aug_next_obss = torch.cat(aug_next_obss, dim=0).to(self.device)
                    q_map_prime = self.forwardFCN(next_states, next_obs[1], aug_next_obss, target_net=True)
                    q_prime = q_map_prime.reshape((batch_size, -1)).max(1)[0]
                    q_target = rewards + self.gamma * q_prime * non_final_masks
                    q_targets.append(q_target)
                q_targets = torch.stack(q_targets).mean(dim=0)

        self.loss_calc_dict['q_target'] = q_targets
        q_outputs = []
        q_preds = []
        actions = []
        for _ in range(self.M):
            aug_obss = []
            aug_actions = []
            for i in range(batch_size):
                aug_obs, aug_action = getDrQAugmentedTransition(obs[0][i, 0].cpu().numpy(), action_idx[i].cpu().numpy(), rzs=self.rzs, aug_type=self.aug_type)
                aug_obss.append(torch.tensor(aug_obs.reshape(1, 1, *aug_obs.shape)))
                aug_actions.append(aug_action)
            aug_obss = torch.cat(aug_obss, dim=0).to(self.device)
            aug_actions = torch.tensor(aug_actions).to(self.device)
            q_output = self.forwardFCN(states, obs[1], aug_obss)
            q_pred = q_output[torch.arange(0, batch_size), aug_actions[:, 2], aug_actions[:, 0], aug_actions[:, 1]]
            q_outputs.append(q_output)
            q_preds.append(q_pred)
            actions.append(aug_actions)
        q_outputs = torch.cat(q_outputs)
        q_preds = torch.cat(q_preds)
        actions = torch.cat(actions)
        self.loss_calc_dict['q1_output'] = q_outputs
        self.loss_calc_dict['q1_action'] = actions
        q_targets = q_targets.repeat(self.M)
        td_loss = F.smooth_l1_loss(q_preds, q_targets)
        with torch.no_grad():
            td_error = torch.abs(q_preds - q_targets).reshape(batch_size, -1).mean(dim=1)

        return td_loss, td_error
import numpy as np
import torch
import torch.nn.functional as F
from copy import deepcopy
# from agents.agents_3d.base_3d import Base3D
from agents.agents_3d.dqn_3d_fcn import DQN3DFCN
from utils import torch_utils

class DQN3DTP(DQN3DFCN):
    def __init__(self, workspace, heightmap_size, device, lr=1e-4, gamma=0.9, sl=False, num_primitives=1,
                 patch_size=24, num_rz=8, rz_range=(0, 7 * np.pi / 8)):
        super().__init__(workspace, heightmap_size, device, lr, gamma, sl, num_primitives, patch_size, num_rz, rz_range)
        self.place = None
        self.pick = None
        self.target_place = None
        self.target_pick = None
        self.place_optimizer = None
        self.pick_optimizer = None

    def initNetwork(self, place, pick):
        self.place = place
        self.pick = pick
        self.target_place = deepcopy(place)
        self.target_pick = deepcopy(pick)
        self.place_optimizer = torch.optim.Adam(self.place.parameters(), lr=self.lr, weight_decay=1e-5)
        self.pick_optimizer = torch.optim.Adam(self.pick.parameters(), lr=self.lr, weight_decay=1e-5)
        self.networks.append(self.place)
        self.networks.append(self.pick)
        self.target_networks.append(self.target_place)
        self.target_networks.append(self.target_pick)
        self.optimizers.append(self.place_optimizer)
        self.optimizers.append(self.pick_optimizer)
        self.updateTarget()

    def forwardFCN(self, states, in_hand, obs, target_net=False, to_cpu=False):
        pick = self.pick if not target_net else self.target_pick
        place = self.place if not target_net else self.target_place

        batch_size = states.shape[0]

        mask_0 = states == 0
        mask_1 = states == 1

        in_hand_1 = in_hand[mask_1]
        obs_0 = obs[mask_0]
        obs_1 = obs[mask_1]
        predictions = torch.zeros(batch_size, self.num_rz, obs.shape[-2], obs.shape[-1]).to(self.device)
        if mask_0.sum():
            out_0 = pick(obs_0).reshape(-1, self.num_rz, obs_0.shape[-2], obs_0.shape[-1])
            predictions[mask_0] = out_0
        if mask_1.sum():
            out_1 = place(obs_1, in_hand_1).reshape(-1, self.num_rz, obs_1.shape[-2], obs_1.shape[-1])
            predictions[mask_1] = out_1
        if to_cpu:
            predictions = predictions.cpu()
        return predictions

    def forwardPick(self, obs, target_net=False, to_cpu=False):
        pick = self.pick if not target_net else self.target_pick
        predictions = pick(obs).reshape(-1, self.num_rz, obs.shape[-2], obs.shape[-1])
        if to_cpu:
            predictions = predictions.cpu()
        return predictions

    def forwardPlace(self, in_hand, obs, target_net=False, to_cpu=False):
        place = self.place if not target_net else self.target_place
        predictions = place(obs, in_hand).reshape(-1, self.num_rz, obs.shape[-2], obs.shape[-1])
        if to_cpu:
            predictions = predictions.cpu()
        return predictions

    def update(self, batch):
        batch_size = len(batch)
        divide_factor = batch_size
        small_batch_size = batch_size//divide_factor
        loss = 0
        td_errors = []
        for i in range(divide_factor):
            small_batch = batch[small_batch_size*i:small_batch_size*(i+1)]
            self._loadBatchToDevice(small_batch)
            td_loss, td_error = self.calcTDLoss()
            td_errors.append(td_error)
            loss += td_loss/divide_factor
            self.loss_calc_dict = {}

        self.place_optimizer.zero_grad()
        self.pick_optimizer.zero_grad()
        loss.backward()
        self.place_optimizer.step()
        self.pick_optimizer.step()

        return loss.item(), torch.cat(td_errors)

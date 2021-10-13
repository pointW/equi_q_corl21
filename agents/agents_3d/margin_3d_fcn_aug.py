import numpy as np
import torch
from agents.agents_3d.margin_3d_fcn import Margin3DFCN
from agents.agents_3d.base_3d_aug import Base3DAug

class Margin3DFCNAug(Margin3DFCN, Base3DAug):
    def __init__(self, workspace, heightmap_size, device, lr=1e-4, gamma=0.9, sl=False, num_primitives=1,
                 patch_size=24, num_rz=8, rz_range=(0, 7 * np.pi / 8), margin='l', margin_l=0.1, margin_weight=0.1,
                 softmax_beta=100):
        Base3DAug.__init__(self, workspace, heightmap_size, device, lr, gamma, sl, num_primitives, patch_size, num_rz, rz_range)
        Margin3DFCN.__init__(self, workspace, heightmap_size, device, lr, gamma, sl, num_primitives, patch_size, num_rz, rz_range, margin, margin_l, margin_weight, softmax_beta)

    def _loadBatchToDevice(self, batch):
        return Base3DAug._loadBatchToDevice(self, batch)

    def update(self, batch):
        losses = []
        td_errors = []
        divide_factor = 2
        small_batch_size = len(batch)//divide_factor
        for i in range(divide_factor):
            small_batch = batch[small_batch_size * i:small_batch_size * (i + 1)]
            loss, td_error = Margin3DFCN.update(self, small_batch)
            losses.append(loss)
            td_errors.append(td_error)
        losses = torch.tensor(losses).mean().item()
        td_errors = torch.cat(td_errors).reshape(len(batch), -1).mean(1)
        return losses, td_errors
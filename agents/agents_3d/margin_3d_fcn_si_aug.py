import numpy as np
import torch
from agents.agents_3d.margin_3d_fcn_si import Margin3DFCNSingleIn
from agents.agents_3d.base_3d_aug import Base3DAug

class Margin3DFCNSingleInAug(Margin3DFCNSingleIn, Base3DAug):
    def __init__(self, workspace, heightmap_size, device, lr=1e-4, gamma=0.9, sl=False, num_primitives=1,
                 patch_size=24, num_rz=8, rz_range=(0, 7 * np.pi / 8), margin='l', margin_l=0.1, margin_weight=0.1,
                 softmax_beta=100):
        Base3DAug.__init__(self, workspace, heightmap_size, device, lr, gamma, sl, num_primitives, patch_size, num_rz, rz_range)
        Margin3DFCNSingleIn.__init__(self, workspace, heightmap_size, device, lr, gamma, sl, num_primitives, patch_size, num_rz, rz_range, margin, margin_l, margin_weight, softmax_beta)

    def _loadBatchToDevice(self, batch):
        return Base3DAug._loadBatchToDevice(self, batch)

    def update(self, batch):
        loss, td_error = Margin3DFCNSingleIn.update(self, batch)
        td_error = td_error.reshape(self.n_aug, -1).mean(0)
        return loss, td_error
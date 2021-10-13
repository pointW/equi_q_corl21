import numpy as np
import torch
import torch.nn.functional as F
from agents.agents_3d.dqn_3d_fcn_si import DQN3DFCNSingleIn
from agents.agents_3d.policy_3d_fcn import Policy3DFCN

class Policy3DFCNSingleIn(DQN3DFCNSingleIn, Policy3DFCN):
    def __init__(self, workspace, heightmap_size, device, lr=1e-4, gamma=0.9, sl=False, num_primitives=1,
                 patch_size=24, num_rz=8, rz_range=(0, 7*np.pi/8)):
        Policy3DFCN.__init__(self, workspace, heightmap_size, device, lr, gamma, sl, num_primitives, patch_size, num_rz, rz_range)
        DQN3DFCNSingleIn.__init__(self, workspace, heightmap_size, device, lr, gamma, sl, num_primitives, patch_size, num_rz, rz_range)

    def update(self, batch):
        return Policy3DFCN.update(self, batch)


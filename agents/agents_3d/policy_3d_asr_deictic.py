import numpy as np
from agents.agents_3d.dqn_3d_asr_deictic import DQN3DASRDeictic
from agents.agents_3d.policy_3d_asr import Policy3DASR

class Policy3DASRDeictic(DQN3DASRDeictic, Policy3DASR):
    def __init__(self, workspace, heightmap_size, device, lr=1e-4, gamma=0.9, sl=False, num_primitives=1,
                 patch_size=24, num_rz=8, rz_range=(0, 7 * np.pi / 8)):
        Policy3DASR.__init__(self, workspace, heightmap_size, device, lr, gamma, sl, num_primitives, patch_size, num_rz, rz_range)
        DQN3DASRDeictic.__init__(self, workspace, heightmap_size, device, lr, gamma, sl, num_primitives, patch_size, num_rz, rz_range)

    def update(self, batch):
        return Policy3DASR.update(self, batch)

import numpy as np
import torch
import torch.nn.functional as F

from agents.agents_3d.dqn_3d_asr import DQN3DASR

class DQN3DASRSepEnc(DQN3DASR):
    def __init__(self, workspace, heightmap_size, device, lr=1e-4, gamma=0.9, sl=False, num_primitives=1,
                 patch_size=24, num_rz=8, rz_range=(0, 7 * np.pi / 8)):
        super().__init__(workspace, heightmap_size, device, lr, gamma, sl, num_primitives, patch_size, num_rz, rz_range)

    def forwardQ2(self, states, in_hand, obs, obs_encoding, pixels, target_net=False, to_cpu=False):
        patch = self.getQ2Input(obs.to(self.device), pixels.to(self.device))
        patch = self.encodeInHand(patch, in_hand.to(self.device))

        obs = obs.to(self.device)
        padding_width = int((self.padding - obs.size(2)) / 2)
        obs = F.pad(obs, (padding_width, padding_width, padding_width, padding_width), mode='constant', value=0)

        q2 = self.q2 if not target_net else self.target_q2
        q2_output = q2(obs, patch).reshape(states.size(0), self.num_primitives, -1)[
            torch.arange(0, states.size(0)),
            states.long()]
        if to_cpu:
            q2_output = q2_output.cpu()
        return q2_output
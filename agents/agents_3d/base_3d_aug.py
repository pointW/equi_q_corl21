import numpy as np
import torch
import torch.nn.functional as F

from agents.agents_3d.base_3d import Base3D

class Base3DAug(Base3D):
    def __init__(self, workspace, heightmap_size, device, lr=1e-4, gamma=0.9, sl=False, num_primitives=1,
                 patch_size=24, num_rz=8, rz_range=(0, 7 * np.pi / 8)):
        super().__init__(workspace, heightmap_size, device, lr, gamma, sl, num_primitives, patch_size, num_rz, rz_range)
        self.n_aug_rot = 4
        self.aug_flip = False
        self.n_aug = self.n_aug_rot if not self.aug_flip else 2*self.n_aug_rot

    def _loadBatchToDevice(self, batch):
        """
        load the input batch in list of transitions into tensors, and save them in self.loss_calc_dict. obs and in_hand
        are saved as tuple in obs
        :param batch: batch data, list of transitions
        :return: batch_size, states, obs, action_idx, rewards, next_states, next_obs, non_final_masks, step_lefts, is_experts
        """
        states = []
        images = []
        in_hands = []
        xys = []
        rewards = []
        next_states = []
        next_obs = []
        next_in_hands = []
        dones = []
        step_lefts = []
        is_experts = []
        for d in batch:
            states.append(d.state)
            images.append(d.obs[0])
            in_hands.append(d.obs[1])
            xys.append(d.action)
            rewards.append(d.reward.squeeze())
            next_states.append(d.next_state)
            next_obs.append(d.next_obs[0])
            next_in_hands.append(d.next_obs[1])
            dones.append(d.done)
            step_lefts.append(d.step_left)
            is_experts.append(d.expert)
        states_tensor = torch.stack(states).long().to(self.device)
        image_tensor = torch.stack(images).to(self.device)
        if len(image_tensor.shape) == 3:
            image_tensor = image_tensor.unsqueeze(1)
        in_hand_tensor = torch.stack(in_hands).to(self.device)
        if len(in_hand_tensor.shape) == 3:
            in_hand_tensor = in_hand_tensor.unsqueeze(1)
        action_tensor = torch.stack(xys).to(self.device)
        rewards_tensor = torch.stack(rewards).to(self.device)
        next_states_tensor = torch.stack(next_states).long().to(self.device)
        next_obs_tensor = torch.stack(next_obs).to(self.device)
        if len(next_obs_tensor.shape) == 3:
            next_obs_tensor = next_obs_tensor.unsqueeze(1)
        next_in_hands_tensor = torch.stack(next_in_hands).to(self.device)
        if len(next_in_hands_tensor.shape) == 3:
            next_in_hands_tensor = next_in_hands_tensor.unsqueeze(1)
        dones_tensor = torch.stack(dones).int()
        non_final_masks = (dones_tensor ^ 1).float().to(self.device)
        step_lefts_tensor = torch.stack(step_lefts).to(self.device)
        is_experts_tensor = torch.stack(is_experts).bool().to(self.device)

        augmented_obs = []
        augmented_action = []
        augmented_next_obs = []
        for i, rot in enumerate(np.linspace(0, 2*np.pi, self.n_aug_rot, endpoint=False)):
            affine_mat = np.asarray([[np.cos(rot), -np.sin(rot), 0],
                                     [np.sin(rot), np.cos(rot), 0]])
            affine_mat.shape = (2, 3, 1)
            affine_mat = torch.from_numpy(affine_mat).permute(2, 0, 1).float().to(self.device)
            affine_mat = affine_mat.repeat(len(batch), 1, 1)

            flow_grid_obs = F.affine_grid(affine_mat, image_tensor.size(), align_corners=False)
            transformed_obs = F.grid_sample(image_tensor, flow_grid_obs, mode='bilinear', padding_mode='zeros', align_corners=False)
            augmented_obs.append(transformed_obs)

            flow_grid_next_obs = F.affine_grid(affine_mat, next_obs_tensor.size(), align_corners=False)
            transformed_next_obs = F.grid_sample(next_obs_tensor, flow_grid_next_obs, mode='bilinear', padding_mode='zeros', align_corners=False)
            augmented_next_obs.append(transformed_next_obs)

            affine_mat = np.asarray([[np.cos(rot), -np.sin(rot)],
                                     [np.sin(rot), np.cos(rot)]])
            xys = action_tensor[:, :2] - self.heightmap_size/2
            transformed_xys = (xys.float() @ torch.tensor(affine_mat.T).float().to(self.device))[:, :2]
            transformed_xys += self.heightmap_size/2
            transformed_xys = transformed_xys.long().clamp(0, self.heightmap_size-1)

            thetas = action_tensor[:, 2:3]
            if self.rzs[-1] <= np.pi:
                transformed_thetas = thetas + i*(self.num_rz*2 // self.n_aug_rot)
            else:
                transformed_thetas = thetas + i*(self.num_rz // self.n_aug_rot)
            transformed_thetas = transformed_thetas % self.num_rz
            augmented_action.append(torch.cat((transformed_xys, transformed_thetas), dim=1))

            if self.aug_flip:
                flipped_obs = torch.flip(transformed_obs, (2,))
                flipped_next_obs = torch.flip(transformed_next_obs, (2,))
                flipped_xys = transformed_xys.clone()
                flipped_xys[:, 0] = self.heightmap_size-1 - flipped_xys[:, 0]
                flipped_thetas = transformed_thetas.clone()
                flipped_thetas = (-flipped_thetas) % self.num_rz

                augmented_obs.append(flipped_obs)
                augmented_next_obs.append(flipped_next_obs)
                augmented_action.append(torch.cat((flipped_xys, flipped_thetas), dim=1))

        states_tensor = states_tensor.repeat(self.n_aug)
        image_tensor = torch.cat(augmented_obs)
        in_hand_tensor = in_hand_tensor.repeat(self.n_aug, 1, 1, 1)
        action_tensor = torch.cat(augmented_action)
        rewards_tensor = rewards_tensor.repeat(self.n_aug)
        next_states_tensor = next_states_tensor.repeat(self.n_aug)
        next_obs_tensor = torch.cat(augmented_next_obs)
        next_in_hands_tensor = next_in_hands_tensor.repeat(self.n_aug, 1, 1, 1)
        non_final_masks = non_final_masks.repeat(self.n_aug)
        step_lefts_tensor = step_lefts_tensor.repeat(self.n_aug)
        is_experts_tensor = is_experts_tensor.repeat(self.n_aug)

        self.loss_calc_dict['batch_size'] = len(batch) * self.n_aug
        self.loss_calc_dict['states'] = states_tensor
        self.loss_calc_dict['obs'] = (image_tensor, in_hand_tensor)
        self.loss_calc_dict['action_idx'] = action_tensor
        self.loss_calc_dict['rewards'] = rewards_tensor
        self.loss_calc_dict['next_states'] = next_states_tensor
        self.loss_calc_dict['next_obs'] = (next_obs_tensor, next_in_hands_tensor)
        self.loss_calc_dict['non_final_masks'] = non_final_masks
        self.loss_calc_dict['step_lefts'] = step_lefts_tensor
        self.loss_calc_dict['is_experts'] = is_experts_tensor

        return states_tensor, (image_tensor, in_hand_tensor), action_tensor, rewards_tensor, next_states_tensor, \
               (next_obs_tensor, next_in_hands_tensor), non_final_masks, step_lefts_tensor, is_experts_tensor

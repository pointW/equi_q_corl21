from agents.agents_2d.dqn_2d_fcn import DQN2DFCN
from agents.agents_2d.margin_2d_fcn import Margin2DFCN
from agents.agents_2d.policy_2d_fcn import Policy2DFCN
from agents.agents_3d.dqn_3d_fcn import DQN3DFCN
from agents.agents_3d.margin_3d_fcn import Margin3DFCN
from agents.agents_3d.policy_3d_fcn import Policy3DFCN
from agents.agents_3d.dqn_3d_fcn_si import DQN3DFCNSingleIn
from agents.agents_3d.policy_3d_fcn_si import Policy3DFCNSingleIn
from agents.agents_3d.margin_3d_fcn_si import Margin3DFCNSingleIn
from agents.agents_3d.dqn_3d_asr import DQN3DASR
from agents.agents_3d.margin_3d_asr import Margin3DASR
from agents.agents_3d.policy_3d_asr import Policy3DASR
from agents.agents_3d.dqn_3d_asr_deictic import DQN3DASRDeictic
from agents.agents_3d.margin_3d_asr_deictic import Margin3DASRDeictic
from agents.agents_3d.policy_3d_asr_deictic import Policy3DASRDeictic
from agents.agents_3d.dqn_3d_asr_sepenc import DQN3DASRSepEnc
from agents.agents_3d.policy_3d_asr_sepenc import Policy3DASRSepEnc
from agents.agents_3d.dqn_3d_fcn_aug import DQN3DFCNAug
from agents.agents_3d.margin_3d_fcn_aug import Margin3DFCNAug
from agents.agents_3d.dqn_3d_fcn_si_aug import DQN3DFCNSingleInAug
from agents.agents_3d.margin_3d_fcn_si_aug import Margin3DFCNSingleInAug
from agents.agents_3d.dqn_3d_asr_aug import DQN3DASRAug
from agents.agents_3d.margin_3d_asr_aug import Margin3DASRAug
from agents.agents_3d.dqn_3d_asr_deictic_aug import DQN3DASRDeicticAug
from agents.agents_3d.margin_3d_asr_deictic_aug import Margin3DASRDeicticAug
from agents.agents_3d.dqn_3d_fcn_si_drq import DQN3DFCNSingleInDrQ
from agents.agents_3d.margin_3d_fcn_si_drq import Margin3DFCNSingleInDrQ
from agents.agents_3d.dqn_3d_asr_drq import DQN3DASRDrQ
from agents.agents_3d.margin_3d_asr_drq import Margin3DASRDrQ
from agents.agents_3d.tp.dqn_3d_tp import DQN3DTP
from agents.agents_3d.tp.policy_3d_tp import Policy3DTP
from agents.agents_3d.tp.margin_3d_tp import Margin3DTP
from agents.agents_6d.dqn_6d_asr_5l import DQN6DASR5L
from agents.agents_6d.margin_6d_asr_5l import Margin6DASR5L
from agents.agents_6d.policy_6d_asr_5l import Policy6DASR5L
from agents.agents_6d.dqn_6d_asr_5l_deictic import DQN6DASR5LDeictic
from agents.agents_6d.margin_6d_asr_5l_deictic import Margin6DASR5LDeictic
from agents.agents_6d.dqn_6d_asr_5l_deictic_35 import DQN6DASR5LDeictic35
from agents.agents_6d.margin_6d_asr_5l_deictic_35 import Margin6DASR5LDeictic35

from utils.parameters import *
from networks.models import ResUCatShared, CNNShared, UCat, CNNSepEnc, CNNPatchOnly, CNNShared5l
from networks.equivariant_models import EquResUExpand, EquResUDFReg, EquResUDFRegNOut, EquShiftQ2DF3, EquShiftQ2DF3P40, EquResUExpandRegNOut
from networks.models import ResURot, ResUTransport, ResUTransportRegress

def createAgent():
    if half_rotation:
        rz_range = (0, (num_rotations - 1) * np.pi / num_rotations)
    else:
        rz_range = (0, (num_rotations - 1) * 2 * np.pi / num_rotations)
    num_rx = 7
    min_rx = -np.pi / 8
    max_rx = np.pi / 8

    diag_length = float(heightmap_size) * np.sqrt(2)
    diag_length = int(np.ceil(diag_length / 32) * 32)
    if in_hand_mode == 'proj':
        patch_channel = 3
    else:
        patch_channel = 1
    patch_shape = (patch_channel, patch_size, patch_size)

    if alg.find('fcn_si') > -1:
        fcn_out = num_rotations * num_primitives
    else:
        fcn_out = num_primitives

    if load_sub is not None or load_model_pre is not None:
        initialize = False
    else:
        initialize = True

    # conventional cnn
    if model == 'resucat':
        fcn = ResUCatShared(1, fcn_out, domain_shape=(1, diag_length, diag_length), patch_shape=patch_shape).to(device)

    ########################### Equivariant FCN and ASR Q1 ############################

    # equivariant asr q1 with lift expansion using cyclic group
    elif model == 'equ_resu_exp':
        fcn = EquResUExpand(1, fcn_out, domain_shape=(1, diag_length, diag_length), patch_shape=patch_shape, N=equi_n, initialize=initialize).to(device)
    # equivariant asr q1 with lift expansion using dihedral group
    elif model == 'equ_resu_exp_flip':
        fcn = EquResUExpand(1, fcn_out, domain_shape=(1, diag_length, diag_length), patch_shape=patch_shape, N=equi_n, flip=True, initialize=initialize).to(device)
    # equivariant asr q1 with dynamic filter using cyclic group
    elif model == 'equ_resu_df':
        fcn = EquResUDFReg(1, fcn_out, domain_shape=(1, diag_length, diag_length), patch_shape=patch_shape, N=equi_n, initialize=initialize).to(device)
    # equivariant asr q1 with dynamic filter using dihedral group
    elif model == 'equ_resu_df_flip':
        fcn = EquResUDFReg(1, fcn_out, domain_shape=(1, diag_length, diag_length), patch_shape=patch_shape, N=equi_n, flip=True, initialize=initialize).to(device)
    # equivariant fcn with dynamic filter
    elif model == 'equ_resu_df_nout':
        fcn = EquResUDFRegNOut(1, num_primitives, domain_shape=(1, diag_length, diag_length), patch_shape=patch_shape, N=equi_n, n_middle_channels=(16, 32, 64, 128), kernel_size=3, quotient=False, last_quotient=True, initialize=initialize).to(device)
    # equivariant fcn with lift expansion
    elif model == 'equ_resu_exp_nout':
        fcn = EquResUExpandRegNOut(1, num_primitives, domain_shape=(1, diag_length, diag_length), patch_shape=patch_shape, N=equi_n, n_middle_channels=(16, 32, 64, 128), kernel_size=3, quotient=False, last_quotient=True, initialize=initialize).to(device)

    ###################################################################################

    # transporter network baselines
    elif alg.find('tp'):
        # pick = Attention(1, num_rotations, half_rotation).to(device)
        # place = Transport(1, num_rotations, half_rotation).to(device)
        pick = ResURot(1, num_rotations, half_rotation).to(device)
        if model == 'tp':
            place = ResUTransport(1, num_rotations, half_rotation).to(device)
        elif model == 'tp_regress':
            place = ResUTransportRegress(1, num_rotations, half_rotation).to(device)

    else:
        raise NotImplementedError

    if alg.find('asr') > -1:
        if alg.find('deictic') > -1:
            q2_output_size = num_primitives
        else:
            q2_output_size = num_primitives * num_rotations
        q2_input_shape = (patch_channel + 1, patch_size, patch_size)
        if q2_model == 'cnn':
            if alg.find('5l') > -1:
                q2 = CNNShared5l(q2_input_shape, q2_output_size).to(device)
            else:
                q2 = CNNShared(q2_input_shape, q2_output_size).to(device)

        ################################### Equivariant ASR Q2 ###########################
        # equivariant asr q2 with dynamic filter
        elif q2_model == 'equ_shift_df':
            if patch_size == 40:
                q2 = EquShiftQ2DF3P40(q2_input_shape, num_rotations, num_primitives, kernel_size=7, n_hidden=32, quotient=False,
                                   last_quotient=True, initialize=initialize).to(device)
            else:
                q2 = EquShiftQ2DF3(q2_input_shape, num_rotations, num_primitives, kernel_size=7, n_hidden=32, quotient=False,
                                    last_quotient=True, initialize=initialize).to(device)
        ###################################################################################

    # 2d agents (x, y)
    if action_sequence == 'xyp':
        if alg == 'dqn_fcn':
            agent = DQN2DFCN(workspace, heightmap_size, device, lr, gamma, sl, num_primitives, patch_size)
        elif alg == 'dagger_fcn':
            agent = Policy2DFCN(workspace, heightmap_size, device, lr, gamma, sl, num_primitives, patch_size)
        elif alg == 'margin_fcn':
            agent = Margin2DFCN(workspace, heightmap_size, device, lr, gamma, sl, num_primitives, patch_size, margin, margin_l, margin_weight, margin_beta)
        else:
            raise NotImplementedError
        agent.initNetwork(fcn)

    # 3d agents (x, y, theta)
    elif action_sequence == 'xyrp':
        # ASR agents
        if alg.find('asr') > -1:
            # ASR
            if alg == 'dqn_asr':
                agent = DQN3DASR(workspace, heightmap_size, device, lr, gamma, sl, num_primitives, patch_size,
                                 num_rotations, rz_range)
                agent.initNetwork(fcn, q2)
            elif alg == 'margin_asr':
                agent = Margin3DASR(workspace, heightmap_size, device, lr, gamma, sl, num_primitives, patch_size,
                                    num_rotations, rz_range, margin, margin_l, margin_weight, margin_beta)
                agent.initNetwork(fcn, q2)
            elif alg == 'dagger_asr':
                agent = Policy3DASR(workspace, heightmap_size, device, lr, gamma, sl, num_primitives, patch_size,
                                    num_rotations, rz_range)
                agent.initNetwork(fcn, q2)
            # ASR + deictic encoding
            elif alg == 'dqn_asr_deictic':
                agent = DQN3DASRDeictic(workspace, heightmap_size, device, lr, gamma, sl, num_primitives, patch_size,
                                        num_rotations, rz_range)
                agent.initNetwork(fcn, q2)
            elif alg == 'margin_asr_deictic':
                agent = Margin3DASRDeictic(workspace, heightmap_size, device, lr, gamma, sl, num_primitives, patch_size,
                                           num_rotations, rz_range, margin, margin_l, margin_weight, margin_beta)
                agent.initNetwork(fcn, q2)
            elif alg == 'dagger_asr_deictic':
                agent = Policy3DASRDeictic(workspace, heightmap_size, device, lr, gamma, sl, num_primitives, patch_size,
                                           num_rotations, rz_range)
                agent.initNetwork(fcn, q2)
            # ASR + DrQ
            elif alg == 'dqn_asr_drq':
                agent = DQN3DASRDrQ(workspace, heightmap_size, device, lr, gamma, sl, num_primitives, patch_size,
                                    num_rotations, rz_range)
                agent.initNetwork(fcn, q2)
            elif alg == 'margin_asr_drq':
                agent = Margin3DASRDrQ(workspace, heightmap_size, device, lr, gamma, sl, num_primitives, patch_size,
                                       num_rotations, rz_range, margin, margin_l, margin_weight, margin_beta)
                agent.initNetwork(fcn, q2)

            else:
                raise NotImplementedError
        # FCN agents
        elif alg.find('fcn_si') > -1:
            # FCN
            if alg == 'dqn_fcn_si':
                agent = DQN3DFCNSingleIn(workspace, heightmap_size, device, lr, gamma, sl, num_primitives, patch_size, num_rotations, rz_range)
            elif alg == 'dagger_fcn_si':
                agent = Policy3DFCNSingleIn(workspace, heightmap_size, device, lr, gamma, sl, num_primitives, patch_size, num_rotations, rz_range)
            elif alg == 'margin_fcn_si':
                agent = Margin3DFCNSingleIn(workspace, heightmap_size, device, lr, gamma, sl, num_primitives, patch_size, num_rotations, rz_range, margin, margin_l, margin_weight, margin_beta)
            # FCN + DrQ
            elif alg == 'dqn_fcn_si_drq':
                agent = DQN3DFCNSingleInDrQ(workspace, heightmap_size, device, lr, gamma, sl, num_primitives, patch_size, num_rotations, rz_range)
            elif alg == 'margin_fcn_si_drq':
                agent = Margin3DFCNSingleInDrQ(workspace, heightmap_size, device, lr, gamma, sl, num_primitives, patch_size, num_rotations, rz_range, margin, margin_l, margin_weight, margin_beta)
            else:
                raise NotImplementedError
            agent.initNetwork(fcn)

        # Rot FCN agents
        elif alg.find('fcn') > -1:
            if alg == 'dqn_fcn':
                agent = DQN3DFCN(workspace, heightmap_size, device, lr, gamma, sl, num_primitives, patch_size, num_rotations, rz_range)
            elif alg == 'dagger_fcn':
                agent = Policy3DFCN(workspace, heightmap_size, device, lr, gamma, sl, num_primitives, patch_size, num_rotations, rz_range)
            elif alg == 'margin_fcn':
                agent = Margin3DFCN(workspace, heightmap_size, device, lr, gamma, sl, num_primitives, patch_size, num_rotations, rz_range, margin, margin_l, margin_weight, margin_beta)
            else:
                raise NotImplementedError
            agent.initNetwork(fcn)

        # Transporter agents
        elif alg.find('tp') > -1:
            if alg == 'dqn_tp':
                agent = DQN3DTP(workspace, heightmap_size, device, lr, gamma, sl, num_primitives, patch_size, num_rotations, rz_range)
            elif alg == 'dagger_tp':
                agent = Policy3DTP(workspace, heightmap_size, device, lr, gamma, sl, num_primitives, patch_size, num_rotations, rz_range)
            elif alg == 'margin_tp':
                agent = Margin3DTP(workspace, heightmap_size, device, lr, gamma, sl, num_primitives, patch_size, num_rotations, rz_range, margin, margin_l, margin_weight, margin_beta)

            agent.initNetwork(place, pick)

    # 6d agent (x, y, z, theta, phi, psi)
    elif action_sequence == 'xyzrrrp':
        # ASR agents
        if alg.find('asr') > -1:
            if alg.find('5l') > -1:
                q3_input_shape = (patch_channel + 1, patch_size, patch_size)
                q4_input_shape = (patch_channel + 3, patch_size, patch_size)
                q5_input_shape = (patch_channel + 3, patch_size, patch_size)
                if alg.find('deictic') > -1:
                    q3_output_size = num_primitives
                    q4_output_size = num_primitives
                    q5_output_size = num_primitives
                    q3_input_shape = (patch_channel + 3, patch_size, patch_size)
                else:
                    q3_output_size = num_primitives * num_zs
                    q4_output_size = num_primitives * num_rx
                    q5_output_size = num_primitives * num_rx
                q3 = CNNShared5l(q3_input_shape, q3_output_size).to(device)
                q4 = CNNShared5l(q4_input_shape, q4_output_size).to(device)
                q5 = CNNShared5l(q5_input_shape, q5_output_size).to(device)
                # cnn q2-q5
                if alg == 'dqn_asr_5l':
                    agent = DQN6DASR5L(workspace, heightmap_size, device, lr, gamma, sl, num_primitives, patch_size,
                                       num_rotations, rz_range, num_rx, (min_rx, max_rx), num_rx, (min_rx, max_rx), num_zs,
                                       (min_z, max_z))
                elif alg == 'margin_asr_5l':
                    agent = Margin6DASR5L(workspace, heightmap_size, device, lr, gamma, sl, num_primitives, patch_size,
                                       num_rotations, rz_range, num_rx, (min_rx, max_rx), num_rx, (min_rx, max_rx),
                                       num_zs, (min_z, max_z), margin, margin_l, margin_weight, margin_beta)
                elif alg == 'dagger_asr_5l':
                    agent = Policy6DASR5L(workspace, heightmap_size, device, lr, gamma, sl, num_primitives, patch_size,
                                          num_rotations, rz_range, num_rx, (min_rx, max_rx), num_rx, (min_rx, max_rx),
                                          num_zs, (min_z, max_z))
                # deictic q2-q5
                elif alg == 'dqn_asr_5l_deictic':
                    agent = DQN6DASR5LDeictic(workspace, heightmap_size, device, lr, gamma, sl, num_primitives, patch_size,
                                              num_rotations, rz_range, num_rx, (min_rx, max_rx), num_rx, (min_rx, max_rx),
                                              num_zs, (min_z, max_z))
                elif alg == 'margin_asr_5l_deictic':
                    agent = Margin6DASR5LDeictic(workspace, heightmap_size, device, lr, gamma, sl, num_primitives, patch_size,
                                                 num_rotations, rz_range, num_rx, (min_rx, max_rx), num_rx, (min_rx, max_rx),
                                                 num_zs, (min_z, max_z), margin, margin_l, margin_weight, margin_beta)
                # deictic q3-q5 (specifically for using equivariant q2)
                elif alg == 'dqn_asr_5l_deictic35':
                    agent = DQN6DASR5LDeictic35(workspace, heightmap_size, device, lr, gamma, sl, num_primitives, patch_size,
                                                num_rotations, rz_range, num_rx, (min_rx, max_rx), num_rx, (min_rx, max_rx),
                                                num_zs, (min_z, max_z))
                elif alg == 'margin_asr_5l_deictic35':
                    agent = Margin6DASR5LDeictic35(workspace, heightmap_size, device, lr, gamma, sl, num_primitives, patch_size,
                                                   num_rotations, rz_range, num_rx, (min_rx, max_rx), num_rx, (min_rx, max_rx),
                                                   num_zs, (min_z, max_z), margin, margin_l, margin_weight, margin_beta)

            agent.initNetwork(fcn, q2, q3, q4, q5)

    agent.detach_es = detach_es
    agent.per_td_error = per_td_error
    agent.aug = aug
    agent.aug_type = aug_type
    return agent


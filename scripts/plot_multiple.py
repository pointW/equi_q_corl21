import os
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import os

def getRewardsSingle(rewards, window=1000):
    moving_avg = []
    i = window
    while i-window < len(rewards):
        moving_avg.append(np.average(rewards[i-window:i]))
        i += window

    moving_avg = np.array(moving_avg)
    return moving_avg

def plotLearningCurveAvg(rewards, window=1000, label='reward', color='b', shadow=True, ax=plt, legend=True, linestyle='-'):
    min_len = np.min(list(map(lambda x: len(x), rewards)))
    rewards = list(map(lambda x: x[:min_len], rewards))
    avg_rewards = np.mean(rewards, axis=0)
    avg_rewards = np.concatenate(([0], avg_rewards))
    # std_rewards = np.std(rewards, axis=0)
    std_rewards = stats.sem(rewards, axis=0)
    std_rewards = np.concatenate(([0], std_rewards))
    xs = np.arange(0, window * avg_rewards.shape[0], window)
    if shadow:
        ax.fill_between(xs, avg_rewards-std_rewards, avg_rewards+std_rewards, alpha=0.2, color=color)
    l = ax.plot(xs, avg_rewards, label=label, color=color, linestyle=linestyle, alpha=0.7)
    if legend:
        ax.legend(loc=4)
    return l

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

def plotLearningCurve(base, ep=50000, use_default_cm=False, window=1000):
    plt.style.use('ggplot')
    plt.figure(dpi=300)
    MEDIUM_SIZE = 12
    BIGGER_SIZE = 14

    plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels

    colors = "bgrycmkwbgrycmkw"
    if use_default_cm:
        color_map = {}
    else:
        color_map = {
            'equi+equi': 'b',
            'cnn+cnn': 'g',
            'tp': 'r',
            'equi_fcn_asr': 'purple',
            'cnn+cnn+aug': 'orange',

            'equi+deictic': 'purple',
            'cnn+deictic': 'r',

            'equi_fcn': 'b',
            'fcn_si': 'g',
            'fcn_si_aug': 'orange',
            'fcn': 'purple',

            'rad_cn_fcn': 'gray',
            'rad_cn_asr': 'gray',

            'drq_cn_fcn': 'orange',
            'drq_cn_asr': 'orange',

            'rad_t': 'orange',

            'q1_equi+q2_equi': 'b',
            'q1_equi+q2_cnn': 'g',
            'q1_cnn+q2_equi': 'r',
            'q1_cnn+q2_cnn': 'purple',

            'q1_equi+q2_deictic': 'g',
            'q1_cnn+q2_deictic': 'r',


            'df': 'b',
            'exp': 'g',

            'equi_fcn_': 'g',

            '5l_equi_equi': 'b',
            '5l_equi_deictic': 'g',
            '5l_equi_cnn': 'r',
            '5l_cnn_cnn': 'purple',
            '5l_cnn_deictic': 'orange',
            '5l_cnn_equi': 'y',

        }

    linestyle_map = {

    }
    name_map = {
        'equi+equi': 'Equivariant ASR',
        'cnn+cnn': 'Conventional ASR',
        'cnn+cnn+aug': 'Soft Equivariant ASR',
        'equi_fcn_asr': 'Equivariant FCN',
        'tp': 'Transporter',

        'rad_se2': 'RAD SE(2)',
        'rad_cn': 'RAD Rotation',
        'rad_t': 'RAD Translation',

        'drq_cn': 'DrQ Rotation',
        'drq_se2': 'DrQ SE(2)',
        'drq_t': 'DrQ Translation',
        'drq_shift': 'DrQ Shift',

        'rad_cn_fcn': 'RAD FCN',
        'rad_cn_asr': 'RAD ASR',
        'drq_cn_fcn': 'DrQ FCN',
        'drq_cn_asr': 'DrQ ASR',

        '5l_equi_equi': 'Equi+Equi+Deic',
        '5l_equi_deictic': 'Equi+Deic+Deic',
        '5l_equi_cnn': 'Equi+Conv+Conv',
        '5l_cnn_equi': 'Conv+Equi+Deic',
        '5l_cnn_deictic': 'Conv+Deic+Deic',
        '5l_cnn_cnn': 'Conv+Conv+Conv',

        'equi+deictic': 'Equivariant Deictic ASR',
        'cnn+deictic': 'Conventional Deictic ASR',

        'equi_fcn': 'Equivariant FCN',
        'fcn_si': 'Conventional FCN',
        'fcn_si_aug': 'Soft Equivariant FCN',
        'fcn': 'Rot FCN',

        'q1_equi+q2_equi': 'Equivariant q1 + Equivariant q2',
        'q1_equi+q2_cnn': 'Equivariant q1 + Conventional q2',
        'q1_cnn+q2_equi': 'Conventional q1 + Equivariant q2',
        'q1_cnn+q2_cnn': 'Conventional q1 + Conventional q2',

        'df': 'Dynamic Filter',
        'exp': 'Lift Expansion',

        'equi_fcn_': 'Equivariant FCN',

        'q1_equi+q2_deictic': 'Equivariant q1 + Deictic q2',
        'q1_cnn+q2_deictic': 'Conventional q1 + Deictic q2',
    }

    sequence = {
        'equi+equi': '0',
        'cnn+cnn': '1',
        'cnn+cnn+aug': '2',
        'equi_fcn_asr': '3',
        'tp': '4',

        'equi_fcn': '0',
        'fcn_si': '1',
        'fcn_si_aug': '2',
        'fcn': '3',
        'rad_cn_fcn': '3.1',
        'rad_cn_asr': '3.1',
        'drq_cn_fcn': '3.2',
        'drq_cn_asr': '3.2',

        'equi+deictic': '2',
        'cnn+deictic': '3',

        'q1_equi+q2_equi': '0',
        'q1_equi+q2_cnn': '1',
        'q1_cnn+q2_equi': '2',
        'q1_cnn+q2_cnn': '3',

        'q1_equi+q2_deictic': '0.5',
        'q1_cnn+q2_deictic': '4',

        'equi_fcn_': '1',

        '5l_equi_equi': '0',
        '5l_equi_deictic': '1',
        '5l_equi_cnn': '2',
        '5l_cnn_equi': '3',
        '5l_cnn_deictic': '4',
        '5l_cnn_cnn': '5',

    }

    # house1-4
    # plt.plot([0, 100000], [0.974, 0.974], label='expert', color='pink')
    # plt.axvline(x=10000, color='black', linestyle='--')

    # house1-5
    # plt.plot([0, 50000], [0.974, 0.974], label='expert', color='pink')
    # 0.004 pos noise
    # plt.plot([0, 50000], [0.859, 0.859], label='expert', color='pink')

    # house1-6 0.941

    # house2
    # plt.plot([0, 50000], [0.979, 0.979], label='expert', color='pink')
    # plt.axvline(x=20000, color='black', linestyle='--')

    # house3
    # plt.plot([0, 50000], [0.983, 0.983], label='expert', color='pink')
    # plt.plot([0, 50000], [0.911, 0.911], label='expert', color='pink')
    # 0.996
    # 0.911 - 0.01

    # house4
    # plt.plot([0, 50000], [0.948, 0.948], label='expert', color='pink')
    # plt.plot([0, 50000], [0.862, 0.862], label='expert', color='pink')
    # 0.875 - 0.006
    # 0.862 - 0.007 *
    # stack
    # plt.plot([0, 100000], [0.989, 0.989], label='expert', color='pink')
    # plt.axvline(x=10000, color='black', linestyle='--')

    i = 0
    methods = filter(lambda x: x[0] != '.', get_immediate_subdirectories(base))
    for method in sorted(methods, key=lambda x: sequence[x] if x in sequence.keys() else x):
        rs = []
        for j, run in enumerate(get_immediate_subdirectories(os.path.join(base, method))):
            try:
                r = np.load(os.path.join(base, method, run, 'info/rewards.npy'))
                if method.find('BC') >= 0 or method.find('tp') >= 0:
                    rs.append(r[-window:].mean())
                else:
                    rs.append(getRewardsSingle(r[:ep], window=window))
            except Exception as e:
                print(e)
                continue

        if method.find('BC') >= 0 or method.find('tp') >= 0:
            avg_rewards = np.mean(rs, axis=0)
            std_rewards = stats.sem(rs, axis=0)

            plt.plot([0, ep], [avg_rewards, avg_rewards],
                     label=name_map[method] if method in name_map else method,
                     color=color_map[method] if method in color_map else colors[i])
            plt.fill_between([0, ep], avg_rewards - std_rewards, avg_rewards + std_rewards, alpha=0.2, color=color_map[method] if method in color_map else colors[i])
        else:
            plotLearningCurveAvg(rs, window, label=name_map[method] if method in name_map else method,
                                 color=color_map[method] if method in color_map else colors[i],
                                 linestyle=linestyle_map[method] if method in linestyle_map else '-')
        i += 1


    # plt.plot([0, ep], [1.450, 1.450], label='planner')
    plt.legend(loc=0, facecolor='w', fontsize='x-large')
    plt.xlabel('number of episodes')
    if base.find('bbp') > -1:
        plt.ylabel('reward')
    else:
        plt.ylabel('task success rate')
    if base.find('bbp') == -1:
        plt.ylim((-0.01, 1.01))
    plt.xlim((-100, ep+100))
    # plt.yticks(np.arange(0., 1.05, 0.1))

    plt.tight_layout()
    plt.savefig(os.path.join(base, 'plot.png'), bbox_inches='tight',pad_inches = 0)

def showPerformance(base):
    methods = sorted(filter(lambda x: x[0] != '.', get_immediate_subdirectories(base)))
    for method in methods:
        rs = []
        for j, run in enumerate(get_immediate_subdirectories(os.path.join(base, method))):
            try:
                r = np.load(os.path.join(base, method, run, 'info/rewards.npy'))
                rs.append(r[-1000:].mean())
            except Exception as e:
                print(e)
        print('{}: {:.3f}'.format(method, np.mean(rs)))


def plotTDErrors():
    plt.style.use('ggplot')
    colors = "bgrycmkw"
    method_map = {
        'ADET': 'm',
        'ADET+Q*': 'g',
        'DAGGER': 'k',
        'DQN': 'c',
        'DQN+guided': 'y',
        'DQN+Q*': 'b',
        'DQN+Q*+guided': 'r',
        "DQfD": 'chocolate',
        "DQfD+Q*": 'grey'
    }
    i = 0

    base = '/media/dian/hdd/unet/perlin'
    for method in sorted(get_immediate_subdirectories(base)):
        rs = []
        if method[0] == '.' or method == 'DAGGER' or method == 'DQN':
            continue
        for j, run in enumerate(get_immediate_subdirectories(os.path.join(base, method))):
            try:
                r = np.load(os.path.join(base, method, run, 'info/td_errors.npy'))
                rs.append(getRewardsSingle(r[:120000], window=1000))
            except Exception as e:
                continue
        if method in method_map:
            plotLearningCurveAvg(rs, 1000, label=method, color=method_map[method])
        else:
            plotLearningCurveAvg(rs, 1000, label=method, color=colors[i])
        # plotLearningCurveAvg(rs, 1000, label=method, color=colors[i])
        i += 1

    plt.legend(loc=0)
    plt.xlabel('number of training steps')
    plt.ylabel('TD error')
    plt.yscale('log')
    # plt.ylim((0.8, 0.93))
    plt.show()

def plotLoss(base, step):
    plt.style.use('ggplot')
    colors = "bgrycmkw"
    method_map = {
        'ADET': 'm',
        'ADET+Q*': 'g',
        'DAGGER': 'k',
        'DQN': 'c',
        'DQN+guided': 'y',
        'DQN+Q*': 'b',
        'DQN+Q*+guided': 'r',
        "DQfD": 'chocolate',
        "DQfD+Q*": 'grey'
    }
    i = 0

    for method in sorted(get_immediate_subdirectories(base)):
        rs = []
        for j, run in enumerate(get_immediate_subdirectories(os.path.join(base, method))):
            try:
                r = np.load(os.path.join(base, method, run, 'info/losses.npy'))[:, 1]
                rs.append(getRewardsSingle(r[:step], window=1000))
            except Exception as e:
                continue
        if method in method_map:
            plotLearningCurveAvg(rs, 1000, label=method, color=method_map[method])
        else:
            plotLearningCurveAvg(rs, 1000, label=method, color=colors[i])
        # plotLearningCurveAvg(rs, 1000, label=method, color=colors[i])
        i += 1

    plt.legend(loc=0)
    plt.xlabel('number of training steps')
    plt.ylabel('loss')
    plt.yscale('log')
    # plt.ylim((0.8, 0.93))
    plt.tight_layout()
    plt.savefig(os.path.join(base, 'plot.png'), bbox_inches='tight', pad_inches=0)


if __name__ == '__main__':
    base = '/media/dian/hdd/mrun_results/equi_0429/corl21_aug/box18_old_aug'
    plotLearningCurve(base, 10000, window=500)
    showPerformance(base)
    # plotLoss(base, 30000)


# Equivariant Q Learning in Spatial Action Spaces

This repository contains the code of the paper [Equivariant Q Learning in Spatial Action Spaces](https://openreview.net/forum?id=IScz42A3iCI). Project website: [https://pointw.github.io/equi_q_page](https://pointw.github.io/equi_q_page).

## Installation
1. Install [anaconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/)
1. Clone this repo
    ```
    git clone https://github.com/pointW/equi_q_corl21.git
    cd equi_q_corl21
    ```
1. Create and activate conda environment
    ```
    conda create --name equi_q python=3.7
    conda activate equi_q
    ```
    Note that this project was developed under pybullet version 2.7.1. Newer version of pybullet should also work, but it is not tested. 
1. Install [PyTorch](https://pytorch.org/) (Recommended: pytorch==1.7.0, torchvision==0.8.1)
1. Install [CuPy](https://github.com/cupy/cupy)
1. Install other requirement packages
    ```
    pip install -r requirements.txt
    ```
1. Clone and install the environment repo 
    ```
    git clone https://github.com/ColinKohler/helping_hands_rl_envs.git -b dian_corl21
    cd helping_hands_rl_envs
    pip install .
    cd ..
    ```
1. Goto the scripts folder of this repo to run experiments
    ```
    cd scripts
    ```

## Environment list
Change the `[env]` accordingly to run in each environment
### 3D Environments
* Block Stacking: `block_stacking`
* Bottle Arrangement: `bottle_tray`
* House Building: `house_building_4`
* Covid Test: `covid_test`
* Box Palletizing: `box_palletizing`
* Bin Packing: `block_bin_packing`
### 6D Environments
* House Building: `bumpy_house_building_4`
* Box Palletizing: `bumpy_box_palletizing`


## Running Equivariant FCN
### Gather expert demonstrations
```
python fill_buffer.py --alg=margin_fcn_si --env=[env] --heightmap_size=90 --num_rotations=6
```
### Equivariant FCN with dynamic filter
```
python main.py --alg=margin_fcn_si --model=equ_resu_df_nout --equi_n=12 --env=[env] --heightmap_size=90 --num_rotations=6 
```
### Equivariant FCN with lift expansion
```
python main.py --alg=margin_fcn_si --model=equ_resu_exp_nout --equi_n=12 --env=[env] --heightmap_size=90 --num_rotations=6 
```

## Running Equivariant ASR
### Gather expert demonstrations
```
python fill_buffer.py --alg=margin_asr --env=[env]
```
### Equivariant ASR
```
python main.py --alg=margin_asr --model=equ_resu_df_flip --equi_n=4 --q2_model=equ_shift_df --env=[env]
```
### Equivariant ASR variations
#### Equivariant ASR with Q1 using lift expansion (instead of dynamic filter)
```
python main.py --alg=margin_asr --model=equ_resu_exp_flip --equi_n=4 --q2_model=equ_shift_df --env=[env]
```
#### Equivariant ASR with Q1 in Cyclic Group C8 (instead of Dihedral group D4)
```
python main.py --alg=margin_asr --model=equ_resu_df --equi_n=8 --q2_model=equ_shift_df --env=[env]
```
#### Equivariant ASR with Q2 using deictic encoding (instead of equivariant network)
```
python main.py --alg=margin_asr_deictic --model=equ_resu_df_flip --equi_n=4 --q2_model=cnn --env=[env]
```

## Running Equivariant ASR in SE(3)
### Gather expert demonstrations
```
python fill_buffer.py --alg=margin_asr_5l_deictic35 --env=[env] --action_sequence=xyzrrrp --in_hand_mode=proj --patch_size=40
```
### Equivariant ASR with Q2 using equivariant network and Q3-Q5 using deictic encoding
```
python main.py --alg=margin_asr_5l_deictic35 --model=equ_resu_df_flip --equi_n=4 --q2_model=equ_shift_df --env=[env] --load_aug_n=0 --action_sequence=xyzrrrp --in_hand_mode=proj --patch_size=40
```
### Equivariant ASR with Q2-Q5 using deictic encoding
```
python main.py --alg=margin_asr_5l_deictic --model=equ_resu_df_flip --equi_n=4 --q2_model=cnn --env=[env] --load_aug_n=0 --action_sequence=xyzrrrp --in_hand_mode=proj --patch_size=40
```

## Results
The training results will be saved under `scripts/outputs`

## Citation
```
@inproceedings{
wang2021equivariant,
title={Equivariant \$Q\$ Learning in Spatial Action Spaces},
author={Dian Wang and Robin Walters and Xupeng Zhu and Robert Platt},
booktitle={5th Annual Conference on Robot Learning },
year={2021},
url={https://openreview.net/forum?id=IScz42A3iCI}
}
```
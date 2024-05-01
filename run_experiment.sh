#!/usr/bin/env bash

GPU_IDX=0
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES=$GPU_IDX

IDENTIFIER='tpatches_debug'
CONFIG='configs\ikeaasm\config_ikeaasm.yaml'

#CONFIG='datasets\data\MSRAction3D_fps'

LOGDIR='./log/'

echo 'hello'

python train.py --identifier $IDENTIFIER --config $CONFIG --logdir $LOGDIR --fix_random_seed
python test.py --identifier $IDENTIFIER --model_ckpt '000200.pt' --logdir $LOGDIR --fix_random_seed
python ./evaluate.py --identifier $IDENTIFIER --logdir $LOGDIR
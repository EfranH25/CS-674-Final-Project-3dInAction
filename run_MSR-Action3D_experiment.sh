
#!/usr/bin/env bash

GPU_IDX=0
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES=$GPU_IDX

IDENTIFIER='debug_msr_action'
CONFIG='configs/msr-action3d/config_msr_action3d.yaml'
LOGDIR='./log/'

python train.py --identifier $IDENTIFIER --config $CONFIG --logdir $LOGDIR --fix_random_seed
#python test.py --identifier $IDENTIFIER --model_ckpt '000200.pt' --logdir $LOGDIR --fix_random_seed
# python ./evaluate.py --identifier $IDENTIFIER --logdir $LOGDIR

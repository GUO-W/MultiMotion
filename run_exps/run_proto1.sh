#!/bin/bash
path_to_your_repo='/local_scratch/wguo/repos/3Dmotion/MultiMotion_private' #TODO :change to your own path
export PYTHONPATH=$PYTHONPATH:$path_to_your_repo
cd $path_to_your_repo 

####################################
## train/test Common-Action-Split ##
####################################

## test our pretrained model
python main/main_pi_3d.py --is_eval --protocol 'pro1'  --test_epo 25\
    --ckpt ./checkpoint/pretrain_ckpt/ckpt_pro1.pth.tar

## train
#python main/main_pi_3d.py --protocol 'pro1' --epoch 40

## test avg
#python main/main_pi_3d.py --is_eval --save_results --protocol 'pro1'  --test_epo 25

## test act-wise
#for t in $(seq 0 6)
#do
#    echo $t
#    python main/main_pi_3d.py --is_eval --save_results --protocol 'pro1'  --test_epo 25 --test_split $t 
#done


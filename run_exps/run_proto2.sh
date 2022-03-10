#!/bin/bash
path_to_your_repo='/local_scratch/wguo/repos/3Dmotion/MultiMotion_private' #TODO :change to your own path
export PYTHONPATH=$PYTHONPATH:$path_to_your_repo
cd $path_to_your_repo

####################################
## train/test Single-Action-Aplit ##
####################################

## test on our pretrained model
for p in $(seq 0 6)
do
    echo $p
    python main/main_pi_3d.py --is_eval --protocol $p  --test_epo 25\
        --ckpt './checkpoint/pretrain_ckpt/ckpt_pro2_act'$p'.pth.tar'
done


## train
#for p in $(seq 0 6)
#do
#    echo $p
#    python main/main_pi_3d.py --protocol $p --epoch 30
#done


## test
#for p in $(seq 0 6)
#do
#    echo $p
#    python main/main_pi_3d.py --is_eval --save_results --protocol $p --test_epo 25
#done

##write test
#cd ../outputs
#python write_results.py 
#cd ../run_exp



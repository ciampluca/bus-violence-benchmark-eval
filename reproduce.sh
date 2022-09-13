#!/bin/bash

set -e

REPS=5
GPU=0

EXPS=(
    #################################
    # Real Life Violence Situations
    #################################
    real-life-violence/resnet18-2plus1d-rgb
    real-life-violence/resnet18-3d-rgb
    real-life-violence/resnet18-2plus1d-fd
    real-life-violence/resnet18-3d-fd
    real-life-violence/slowfast-rgb
    real-life-violence/slowfast-fd
    real-life-violence/biconvlstm-eccv2018-rgb
    real-life-violence/biconvlstm-eccv2018-fd
    real-life-violence/convlstm-avss2017-rgb
    real-life-violence/convlstm-avss2017-fd
    real-life-violence/videoswintransformer-k400-rgb
    real-life-violence/videoswintransformer-k400-fd
    ############################
    # Surveillance Camera Fight
    ############################
    surveillance-camera-fight/resnet18-2plus1d-rgb
    surveillance-camera-fight/resnet18-3d-rgb
    surveillance-camera-fight/resnet18-2plus1d-fd
    surveillance-camera-fight/resnet18-3d-fd
    surveillance-camera-fight/slowfast-rgb
    surveillance-camera-fight/slowfast-fd
    surveillance-camera-fight/devtr-rgb
    surveillance-camera-fight/devtr-fd
    surveillance-camera-fight/biconvlstm-eccv2018-rgb
    surveillance-camera-fight/biconvlstm-eccv2018-fd
    surveillance-camera-fight/convlstm-avss2017-rgb
    surveillance-camera-fight/convlstm-avss2017-fd
    surveillance-camera-fight/videoswintransformer-k400-rgb
    surveillance-camera-fight/videoswintransformer-k400-fd
    ############################
    # RWF-2000
    ############################
    RWF-2000/resnet18-2plus1d-rgb
    RWF-2000/resnet18-3d-rgb
    RWF-2000/resnet18-2plus1d-fd
    RWF-2000/resnet18-3d-fd
    RWF-2000/slowfast-rgb
    RWF-2000/slowfast-fd
    RWF-2000/biconvlstm-eccv2018-rgb
    RWF-2000/biconvlstm-eccv2018-fd
    RWF-2000/convlstm-avss2017-rgb
    RWF-2000/convlstm-avss2017-fd
    RWF-2000/videoswintransformer-k400-rgb
    RWF-2000/videoswintransformer-k400-fd
)

# Train & Evaluate on the same dataset
for REP in $(seq 0 $(( $REPS - 1 ))); do    # Multiple repetitions of the same experiment, varying the split seed (for cross-validation)
    for EXP in ${EXPS[@]}; do
        CUDA_VISIBLE_DEVICES=$GPU HYDRA_FULL_ERROR=1 python train.py experiment=$EXP data.train.split_seed=$REP data.validation.split_seed=$REP
        if [[ $EXP == RWF* ]]
        then
            CUDA_VISIBLE_DEVICES=$GPU HYDRA_FULL_ERROR=1 python evaluate.py runs/experiment=$EXP/run-$REP --debug --test-split all --data-root ./data/RWF-2000/test        
        else
            CUDA_VISIBLE_DEVICES=$GPU HYDRA_FULL_ERROR=1 python evaluate.py runs/experiment=$EXP/run-$REP --debug 
        fi 
    done
done

# Evaluate on the bus dataset (cross dataset validation)
for REP in $(seq 0 $(( $REPS - 1 ))); do
    for EXP in ${EXPS[@]}; do
        CUDA_VISIBLE_DEVICES=$GPU HYDRA_FULL_ERROR=1 python evaluate.py runs/experiment=$EXP/run-$REP --debug --cross-dataset --dataset-name bus-violence --test-split all --data-root ./data/bus-violence
    done
done

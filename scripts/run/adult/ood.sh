#!/bin/bash

export TOKENIZERS_PARALLELISM=False

ROOT_FOLDER=~/Documents/Projects/
PROJECT_FOLDER=$ROOT_FOLDER/DP-2Stage
# cd $PROJECT_FOLDER 

LLM=GPT2
DATASET_NAME=adult

#train stage 1 
TAG=${PROJECT_FOLDER}/scripts/experiments/${LLM}/2stage/${DATASET_NAME}/ood/stage1
CONFIG=${TAG}/full_nodp.sh
bash $PROJECT_FOLDER/scripts/2Stage_train.sh $CONFIG


#train stage 2
TAG=${PROJECT_FOLDER}/scripts/experiments/${LLM}/2stage/${DATASET_NAME}/ood/stage2
CONFIG=${TAG}/full_dp-eps1.sh
bash $PROJECT_FOLDER/scripts/2Stage_train.sh $CONFIG

# #generate
bash $PROJECT_FOLDER/scripts/generate.sh $CONFIG synth_0 1000
bash $PROJECT_FOLDER/scripts/generate.sh $CONFIG synth_1 2000
bash $PROJECT_FOLDER/scripts/generate.sh $CONFIG synth_2 3000
bash $PROJECT_FOLDER/scripts/generate.sh $CONFIG synth_3 4000
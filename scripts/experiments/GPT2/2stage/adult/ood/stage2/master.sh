LEARNING_RATE=0.0005
LR_SCHEDULER_TYPE=linear


STAGE1_DATASET=airline
STAGE1_CHKPT_PATH=epoch5
STAGE1_EPOCH=5

STAGE1_WEIGHTED_LOSS=-1
STAGE1_BS=32
# STAGE1_DATASET_SIZE=30932
STAGE1_DATASET_SIZE=1000 #testing
STAGE1_LEARNING_RATE=0.0005
STAGE1_ID=ts${STAGE1_DATASET_SIZE}-bs${STAGE1_BS}-epoch${STAGE1_EPOCH}
STAGE1_FINETUNE_ADAPTER=entire
MODEL_NAME_OR_PATH_2STAGE="${PROJECT_FOLDER}/runs/2Stage_LR${STAGE1_LEARNING_RATE}-k${SEED}-${LR_SCHEDULER_TYPE}/stage1_shuffle-${SHUFFLE_DATASET}-${DATASET_NAME}_wl${STAGE1_WEIGHTED_LOSS}/${STAGE1_DATASET}/NonDP/${LLM}/${STAGE1_FINETUNE_ADAPTER}/${STAGE1_ID}/${STAGE1_CHKPT_PATH}"
CHECKPOINT_PATH_2STAGE=${MODEL_NAME_OR_PATH_2STAGE}/model.safetensors


# stage2 specifics
# SHUFFLE_DATASET=True
ADJUSTED_NAME=ood-using-${STAGE1_DATASET}
IDENTIFIER=${STAGE1_FINETUNE_ADAPTER}_${ADJUSTED_NAME}_${STAGE1_ID}-pretrain-stage1-for-${STAGE1_CHKPT_PATH}
BASEFOLDER="${PROJECT_FOLDER}/runs/2Stage_LR${LEARNING_RATE}-k${SEED}-${LR_SCHEDULER_TYPE}/stage2_shuffle-${SHUFFLE_DATASET}-${DATASET_NAME}_${IDENTIFIER}_wl${WEIGHTED_LOSS}"

FINETUNE_EPOCH=10
SAVE_EVERY_EPOCH=5

CHECKPOINT_EPOCH=$FINETUNE_EPOCH

if [[ ${STAGE} == '2' ]]
    then
    ENABLE_PRIVACY=True
    SETTING="DP/${LLM}"
    OUTPUT_DIR=${BASEFOLDER}/${DATASET_NAME}/${SETTING}/${FINETUNE_ADAPTER}/ts${MAX_FINETUNE_TRAIN_SIZE}-bs${TRAIN_BATCH_SIZE}-epoch${FINETUNE_EPOCH}-mbs${MICRO_BATCH_SIZE}-eps${EPSILON}-clip${CLIP}/epoch${CHECKPOINT_EPOCH}

else
    ENABLE_PRIVACY=False
    SETTING="NonDP/${LLM}"
    OUTPUT_DIR=${BASEFOLDER}/${DATASET_NAME}/${SETTING}/${FINETUNE_ADAPTER}/ts${MAX_FINETUNE_TRAIN_SIZE}-bs${TRAIN_BATCH_SIZE}-epoch${FINETUNE_EPOCH}/epoch${CHECKPOINT_EPOCH}

fi

#  Generation
CHECKPOINT_PATH=$OUTPUT_DIR/model.safetensors
SYNTH_FOLDER=$OUTPUT_DIR/synth_data_cat_temp${TEMPERATURE}_top${TOP_P}
ROOT_FOLDER=~/Documents/Projects/
PROJECT_FOLDER=$ROOT_FOLDER/DP-2Stage
cd $PROJECT_FOLDER 

CHECKPOINT_PATH=None
LR_SCHEDULER_TYPE=linear
# RESUME_CHECKPOINT_PATH=none
RESUME_FROM_CHECKPOINT=True
SHUFFLE_DATASET=True
START_COL=income

CONFIG=$1

echo "CONFIG....."
echo ${CONFIG}
source ${CONFIG}

if [[ ${MODEL_NAME_OR_PATH_2STAGE} ]]
then 
    MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH_2STAGE}
fi

if [[ ${CHECKPOINT_PATH_2STAGE} ]]
then 
    CHECKPOINT_PATH=${CHECKPOINT_PATH_2STAGE}
fi

echo $ENABLE_PRIVACY, $TRAIN_FILE

if [[ ${STAGE} == '2' ]]
    then
    if [[ $FINETUNE_STEP != '0' ]]
    then
        OUTPUT_DIR=${BASEFOLDER}/${DATASET_NAME}/${SETTING}/${FINETUNE_ADAPTER}/ts${MAX_FINETUNE_TRAIN_SIZE}-bs${TRAIN_BATCH_SIZE}-step${FINETUNE_STEP}-mbs${MICRO_BATCH_SIZE}-eps${EPSILON}-clip${CLIP}
    else
        OUTPUT_DIR=${BASEFOLDER}/${DATASET_NAME}/${SETTING}/${FINETUNE_ADAPTER}/ts${MAX_FINETUNE_TRAIN_SIZE}-bs${TRAIN_BATCH_SIZE}-epoch${FINETUNE_EPOCH}-mbs${MICRO_BATCH_SIZE}-eps${EPSILON}-clip${CLIP}
    fi

    python ${PROJECT_FOLDER}/ft_opacus.py \
    --train_file ${TRAIN_FILE} \
    --validation_file ${VALIDATION_FILE}\
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --model_type ${MODEL_TYPE}\
    --train_batch_size ${TRAIN_BATCH_SIZE} \
    --validation_batch_size ${EVAL_BATCH_SIZE} \
    --output_dir ${OUTPUT_DIR} \
    --finetune_type ${FINETUNE_ADAPTER}\
    --num_train_epochs ${FINETUNE_EPOCH}\
    --device ${DEVICE}\
    --seed ${SEED}\
    --config_name ${MODEL_NAME_OR_PATH}\
    --tokenizer_name ${TOKENIZER_NAME}\
    --max_train_samples ${MAX_FINETUNE_TRAIN_SIZE}\
    --loading_4_bit ${LOADING_4_BIT} \
    --enable_privacy ${ENABLE_PRIVACY} \
    --micro_batch_size ${MICRO_BATCH_SIZE} \
    --target_epsilon ${EPSILON} \
    --max_grad_norm ${CLIP} \
    --save_every_epoch ${SAVE_EVERY_EPOCH}\
    --evaluation_mode True\
    --generation_mode False\
    --sampling_max_allowed_time 900\
    --n_synth_samples 16281\
    --synth_folder ${OUTPUT_DIR}/synth_data\
    --do_impute True\
    --sample_batch $TRAIN_BATCH_SIZE\
    --learning_rate ${LEARNING_RATE}\
    --max_train_steps ${FINETUNE_STEP}\
    --save_every_step ${SAVE_EVERY_STEP}\
    --weighted_loss ${WEIGHTED_LOSS} \
    --cache_dir ${PROJECT_FOLDER}/cache \
    --checkpoint_path ${CHECKPOINT_PATH} \
    --lr_scheduler_type ${LR_SCHEDULER_TYPE} \
    --resume_from_checkpoint ${RESUME_FROM_CHECKPOINT} \
    --shuffle_dataset ${SHUFFLE_DATASET} \
    --start_col ${START_COL}
else
    if [[ $FINETUNE_STEP != '0' ]]
    then
        OUTPUT_DIR=${BASEFOLDER}/${DATASET_NAME}/${SETTING}/${FINETUNE_ADAPTER}/ts${MAX_FINETUNE_TRAIN_SIZE}-bs${TRAIN_BATCH_SIZE}-step${FINETUNE_STEP}
    else
        OUTPUT_DIR=${BASEFOLDER}/${DATASET_NAME}/${SETTING}/${FINETUNE_ADAPTER}/ts${MAX_FINETUNE_TRAIN_SIZE}-bs${TRAIN_BATCH_SIZE}-epoch${FINETUNE_EPOCH}
    fi

    echo $OUTPUT_DIR
    python ${PROJECT_FOLDER}/ft_opacus.py \
    --train_file ${TRAIN_FILE} \
    --validation_file ${VALIDATION_FILE}\
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --model_type ${MODEL_TYPE}\
    --train_batch_size ${TRAIN_BATCH_SIZE} \
    --validation_batch_size ${EVAL_BATCH_SIZE} \
    --output_dir ${OUTPUT_DIR} \
    --finetune_type ${FINETUNE_ADAPTER}\
    --num_train_epochs ${FINETUNE_EPOCH}\
    --device ${DEVICE}\
    --seed ${SEED}\
    --config_name ${MODEL_NAME_OR_PATH}\
    --tokenizer_name ${TOKENIZER_NAME}\
    --max_train_samples ${MAX_FINETUNE_TRAIN_SIZE}\
    --loading_4_bit ${LOADING_4_BIT} \
    --enable_privacy ${ENABLE_PRIVACY} \
    --save_every_epoch ${SAVE_EVERY_EPOCH}\
    --evaluation_mode True\
    --generation_mode False\
    --sampling_max_allowed_time 900\
    --n_synth_samples 16281\
    --synth_folder ${OUTPUT_DIR}/synth_data\
    --do_impute True\
    --sample_batch $TRAIN_BATCH_SIZE\
    --learning_rate ${LEARNING_RATE}\
    --max_train_steps ${FINETUNE_STEP}\
    --save_every_step ${SAVE_EVERY_STEP}\
    --weighted_loss ${WEIGHTED_LOSS} \
    --cache_dir ${PROJECT_FOLDER}/cache \
    --checkpoint_path ${CHECKPOINT_PATH} \
    --lr_scheduler_type ${LR_SCHEDULER_TYPE} \
    --resume_from_checkpoint ${RESUME_FROM_CHECKPOINT} \
    --shuffle_dataset ${SHUFFLE_DATASET} \
    --start_col ${START_COL}
fi
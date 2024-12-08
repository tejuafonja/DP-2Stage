ROOT_FOLDER=~/Documents/Projects/
PROJECT_FOLDER=$ROOT_FOLDER/DP-2Stage

CONFIG=$1
SYNTH_SAVE_AS=$2
GENERATION_SEED=$3
START_PROMPT="default"
START_COL="income"
SHUFFLE_DATASET=True
REJECTION_SAMPLE=False


source ${CONFIG}
export HF_HOME=${PROJECT_FOLDER}
SYNTH_FOLDER=${SYNTH_FOLDER}_impute-${DO_IMPUTE}
echo $SYNTH_FOLDER
# CUDA_LAUNCH_BLOCKING=1 
python $PROJECT_FOLDER/ft_opacus.py \
--train_file ${TRAIN_FILE} \
--model_name_or_path ${MODEL_NAME_OR_PATH} \
--model_type ${MODEL_TYPE} \
--output_dir ${OUTPUT_DIR} \
--finetune_type ${FINETUNE_ADAPTER} \
--device ${DEVICE} \
--seed ${SEED} \
--config_name ${MODEL_NAME_OR_PATH} \
--tokenizer_name ${MODEL_NAME_OR_PATH} \
--max_train_samples ${MAX_FINETUNE_TRAIN_SIZE} \
--loading_4_bit ${LOADING_4_BIT} \
--train_mode False \
--evaluation_mode False \
--generation_mode True \
--n_synth_samples ${N_SYNTH_SAMPLES} \
--synth_folder ${SYNTH_FOLDER} \
--do_impute ${DO_IMPUTE} \
--sample_batch $SAMPLE_BATCH_SIZE \
--checkpoint_path ${CHECKPOINT_PATH} \
--n_synth_set ${N_SYNTH_SET} \
--generation_seed $GENERATION_SEED \
--top_p ${TOP_P} \
--temperature ${TEMPERATURE} \
--synth_save_as ${SYNTH_SAVE_AS} \
--start_prompt ${START_PROMPT} \
--cache_dir ${PROJECT_FOLDER}/cache \
--start_col ${START_COL} \
--shuffle_dataset ${SHUFFLE_DATASET} \
--rejection_sample ${REJECTION_SAMPLE}
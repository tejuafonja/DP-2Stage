TRAIN_BATCH_SIZE=32
MICRO_BATCH_SIZE=16
EVAL_BATCH_SIZE=32

EPSILON=1
CLIP=1

FINETUNE_EPOCH=10
FINETUNE_STEP=0

SAVE_EVERY_EPOCH=5
SAVE_EVERY_STEP=0


DATASET_NAME="adult"
# TRAIN_FILE="${PROJECT_FOLDER}/data/${DATASET_NAME}/k1000/train.csv"
# VALIDATION_FILE="${PROJECT_FOLDER}/data/${DATASET_NAME}/k1000/test.csv"

# for testing
TRAIN_FILE="${PROJECT_FOLDER}/data/${DATASET_NAME}/k1000/valid.csv"
VALIDATION_FILE="${PROJECT_FOLDER}/data/${DATASET_NAME}/k1000/valid.csv"
TRAIN_SIZE=`cat ${TRAIN_FILE} | wc -l`
MAX_FINETUNE_TRAIN_SIZE=$(($TRAIN_SIZE-1))

START_COL=income

SAMPLE_BATCH_SIZE=100
N_SYNTH_SAMPLES=${MAX_FINETUNE_TRAIN_SIZE}


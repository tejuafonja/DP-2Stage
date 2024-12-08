LLM=GPT2
source $PROJECT_FOLDER/scripts/experiments/${LLM}.sh
source $PROJECT_FOLDER/scripts/experiments/${LLM}/2stage/airline_general.sh

STAGE1_DATASET=airline
FINETUNE_ADAPTER=entire

SHUFFLE_DATASET=False

FINETUNE_EPOCH=5
SAVE_EVERY_EPOCH=5

DATASET_NAME=${STAGE1_DATASET}

# TRAIN_FILE="${PROJECT_FOLDER}/data/${STAGE1_DATASET}/k1000/train.csv"
# VALIDATION_FILE="${PROJECT_FOLDER}/data/airline/k1000/test.csv"

# testing
TRAIN_FILE="${PROJECT_FOLDER}/data/${STAGE1_DATASET}/k1000/valid.csv"
VALIDATION_FILE="${PROJECT_FOLDER}/data/airline/k1000/valid.csv"


WEIGHTED_LOSS=-1
source $PROJECT_FOLDER/scripts/experiments/${LLM}/2stage/adult/stage1_master.sh
LLM=GPT2
source $PROJECT_FOLDER/scripts/experiments/${LLM}.sh
source $PROJECT_FOLDER/scripts/experiments/${LLM}/2stage/adult_general.sh

WEIGHTED_LOSS=0.65
STAGE=2

LEARNING_RATE=0.0005
LR_SCHEDULER_TYPE=linear

SHUFFLE_DATASET=False

FINETUNE_ADAPTER="entire"

DO_IMPUTE=False
REJECTION_SAMPLE=True

EPSILON=1
CLIP=1
source $PROJECT_FOLDER/scripts/experiments/${LLM}/2stage/${DATASET_NAME}/ood/stage2/master.sh
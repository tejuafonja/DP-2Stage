ROOT_FOLDER=~/Documents/Projects/
PROJECT_FOLDER=$ROOT_FOLDER/DP-2Stage

DATASET_NAME=adult

SEED=1000
RUN_FOLDER=runs/baseline

SAMPLE_SIZE=1000

# Non-DP
# python ${PROJECT_FOLDER}/baselines/sdv_ctgan.py --n_synth_set 4 --dataset_name ${DATASET_NAME} --dataset_path ${PROJECT_FOLDER}/data/${DATASET_NAME}/k1000/train.csv  --n_synth_samples ${SAMPLE_SIZE} --do_train --do_generate --device cuda --output_dir ${RUN_FOLDER} --epochs 300 --seed ${SEED}
# python ${PROJECT_FOLDER}/baselines/sdv_tvae.py --n_synth_set 4 --dataset_name ${DATASET_NAME} --dataset_path ${PROJECT_FOLDER}/data/${DATASET_NAME}/k1000/train.csv  --n_synth_samples ${SAMPLE_SIZE} --do_train --do_generate --device cuda --output_dir ${RUN_FOLDER} --epochs 300 --seed ${SEED}

# pip install smartnoise-synth #needs this installed but will overwright opacus.
# pip install -r requirements.txt #to restore versions we're working with
# python ${PROJECT_FOLDER}/baselines/dpvae.py --output_dir ${RUN_FOLDER} --epochs 300 --n_synth_set 4 --dataset_name ${DATASET_NAME}  --dataset_path ${PROJECT_FOLDER}/data/${DATASET_NAME}/k1000/train.csv  --n_synth_samples ${SAMPLE_SIZE} --do_train --do_generate --device_type cuda --seed ${SEED}

# DP
# pip install smartnoise-synth
python ${PROJECT_FOLDER}/baselines/smartnoise.py --n_synth_set 4 --dataset_name ${DATASET_NAME} --dataset_path ${PROJECT_FOLDER}/data/${DATASET_NAME}/k1000/train.csv  --target_epsilon 1 --n_synth_samples ${SAMPLE_SIZE} --synthesizer dpctgan --output_dir ${RUN_FOLDER}
# python ${PROJECT_FOLDER}/baselines/smartnoise.py --n_synth_set 4 --dataset_name ${DATASET_NAME} --dataset_path ${PROJECT_FOLDER}/data/${DATASET_NAME}/k1000/train.csv  --target_epsilon 1 --n_synth_samples ${SAMPLE_SIZE} --synthesizer dpgan --output_dir ${RUN_FOLDER}

# pip install -r requirements.txt #to restore versions we're working with
# python ${PROJECT_FOLDER}/baselines/dpvae.py --output_dir ${RUN_FOLDER} --n_synth_set 4 --dataset_name ${DATASET_NAME}  --dataset_path ${PROJECT_FOLDER}/data/${DATASET_NAME}/k1000/train.csv  --n_synth_samples ${SAMPLE_SIZE} --do_train --do_generate --device_type cuda --target_epsilon 1 --enable_privacy --epochs 300

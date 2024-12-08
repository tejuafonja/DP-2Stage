ROOT_FOLDER=~/Documents/Projects/
PROJECT_FOLDER=$ROOT_FOLDER/DP-2Stage

BINS="50 20"
SCORERS="auc f1 accuracy"
MODELS="xgb logistic"
SAVE_NAME=tabular_metrics

RESULT_PATH=${PROJECT_FOLDER}/results/${SAVE_NAME}.csv

PATHS="${PROJECT_FOLDER}/runs/2Stage_LR0.0005-k1000-linear/stage2_shuffle-False-adult_entire_ood-using-airline_ts1000-bs32-epoch5-pretrain-stage1-for-epoch5_wl0.65/adult/DP/GPT2/entire/ts1000-bs32-epoch10-mbs16-eps1-clip1/epoch10/synth_data_cat_temp0.7_top1.0_impute-False/processed_tables/"
FAKE_PATHS=`ls ${PATHS}/*.csv`

for fake_path in ${FAKE_PATHS}
do 
    value='/airline'

    if [[ $fake_path == *$value* ]]; then
    DATASET_NAME=airline
    TARGET_NAME=satisfaction
    else
    DATASET_NAME=adult
    TARGET_NAME=income
    fi

    TRAIN_PATH=${PROJECT_FOLDER}/data/${DATASET_NAME}/k1000/train.csv
    TEST_PATH=${PROJECT_FOLDER}/data/${DATASET_NAME}/k1000/test.csv

    echo $fake_path
    python ${PROJECT_FOLDER}/metrics/run.py \
        --fake_path ${fake_path} \
        --metric_name exact_duplicates \
        --train_path $TRAIN_PATH\
        --test_path $TEST_PATH \
        --result_path $RESULT_PATH

    for bins in ${BINS}
    do
        echo $fake_path, ${bins}
        python ${PROJECT_FOLDER}/metrics/run.py \
            --fake_path ${fake_path} \
            --metric_name histogram_intersection \
            --bins ${bins} \
            --train_path $TRAIN_PATH\
            --test_path $TEST_PATH \
            --result_path $RESULT_PATH
    done

    for scorer in ${SCORERS}
    do
        for model in ${MODELS}
        do
            echo $fake_path, ${scorer}, ${model}
            python ${PROJECT_FOLDER}/metrics/run.py \
                --fake_path ${fake_path} \
                --metric_name efficacy_test \
                --model_name $model \
                --scorer $scorer \
                --train_path $TRAIN_PATH\
                --test_path $TEST_PATH \
                --result_path $RESULT_PATH\
                --target_name ${TARGET_NAME}
        done
    done
done
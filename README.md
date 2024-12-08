# **DP-2Stage: Adapting Language Models as Differentially Private Tabular Data Generators**

This repository provides tools and scripts DP-2Stage, a two-stage fine-tuning framework for differentially private tabular data generation.

---

## **Environment Setup**

1. Set the `PYTHONPATH`:
   ```bash
   export PYTHONPATH=$PWD
   ```

2. Create and activate the project environment:
   ```bash
   conda create -n dp2stage python=3.9
   conda activate dp2stage
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## **Data Preparation**

### Download the Dataset
To download the **Adult** dataset:
```bash
python download_dataset.py -name adult --train_subset 30932 --valid_subset 1000 --split_by_seed --seed 1000 --use_full_dataset
```

For the **Airline** dataset:
1. Create a [Kaggle account](https://www.kaggle.com/).
2. Generate an API key and save it to `~/.kaggle/kaggle.json`.
```bash
python download_dataset.py -name airline --train_subset 103904 --valid_subset 1000 --split_by_seed --seed 1000 --use_full_dataset
```

### Create Pseudo Data (Stage 1 Training)
To generate independent uniform pseudo data for the Adult dataset:
```bash
python utils/create_independent_uniform_pseudo_data.py \
    --dataset_path ./data/adult/k1000/train.csv \
    --seed 1000 \
    --output_dir ./data/adult-uniform/k1000 \
    --output_name train \
    --n_synth_samples 30932
```

---

## **Training and Sampling**

### Run Training and Sampling
Edit the necessary configuration in the scripts, then execute:
```bash
bash scripts/run/adult/ood.sh
```

---

## **Evaluate Synthetic Data**

To evaluate the synthetic data, modify the evaluation scripts as needed, and then run:
```bash
bash scripts/tabular_metrics.sh
```

---

## **Baselines**

To run the **SmartChoice** baseline models:
1. It is recommended to create a separate environment due to dependency conflicts:
   ```bash
   conda create -n smartchoice python=3.9
   conda activate smartchoice
   ```

2. Install the baseline dependencies:
   ```bash
   pip install -r requirements.txt
   pip install smartnoise-synth
   ```

Alternatively, you can revert the `opacus` version to match this project's requirements by reinstalling dependencies from `requirements.txt`.

Modify the `scripts/baseline.sh` as needed, and the run:
```bash
bash scripts/baselines.sh
```
---

## **Notes**

- Ensure scripts are updated with your specific paths and configurations before running.
- Refer to the documentation or script comments for additional options and explanations.
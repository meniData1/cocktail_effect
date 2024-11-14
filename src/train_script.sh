#!/bin/bash

# change model to desired model
MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"

DATASETS=(
  headline
  fpb
  fin_ner_cls
  finqa
  convfinqa
  twitter_topics
  twitter_sa
  orca_math
  open_orca
)

export DS_SKIP_CUDA_CHECK=1

# each one separately
for DATASET in "${DATASETS[@]}"; do
  echo "Starting process for $DATASET with PID $$"
  python src/cocktail.py --datasets $DATASET --model $MODEL
  echo "Starting training process for $DATASET with PID $$"
  nohup llamafactory-cli train training_configs/current_config.yaml  >> nohup.log 2>&1 &
  wait $!
done

# on every pair of datasets
for i in "${!DATASETS[@]}"; do
  for (( j=i+1; j<${#DATASETS[@]}; j++ )); do
    DATASET1="${DATASETS[$i]}"
    DATASET2="${DATASETS[$j]}"
    echo "Starting process for $DATASET1 and $DATASET2 with PID $$"
    python src/cocktail.py --datasets $DATASET1 $DATASET2 --model $MODEL
    echo "Starting training process for $DATASET1 and $DATASET2 with PID $$"
    nohup llamafactory-cli train training_configs/current_config.yaml >> nohup.log 2>&1 &
    wait $!
  done
done

# every leave-one-out
for DATASET in "${DATASETS[@]}"; do
  echo "Starting process for all datasets except $DATASET with PID $$"
  DATASETS_EXCEPT=("${DATASETS[@]/$DATASET}")
  python src/cocktail.py --datasets ${DATASETS_EXCEPT[@]} --model $MODEL
  echo "Starting training process for all datasets except $DATASET with PID $$"
  nohup llamafactory-cli train training_configs/current_config.yaml >> nohup.log 2>&1 &
  wait $!
done

# on all datasets
echo "Starting process for all datasets with PID $$"
python src/cocktail.py --datasets ${DATASETS[@]} --model $MODEL
echo "Starting training process for all datasets with PID $$"
nohup llamafactory-cli train training_configs/current_config.yaml >> nohup.log 2>&1 &
wait $!

echo "Finished training and evaluation for all datasets with model $MODEL"

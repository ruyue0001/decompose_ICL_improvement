#!/usr/bin/bash

for dataset_name in "roc_story" "roc_story_ending" "samsum" "reddit"
do
    accelerate launch --config_file "./acc_config_dist.yaml" "main.py" -num_demo "0" -dataset "$dataset_name" -model_name "mistral" -seed "100" -batch_size 4 -gen_style "train_short" -no_format

    accelerate launch --config_file "./acc_config_dist.yaml" "main.py" -num_demo "0" -dataset "$dataset_name" -model_name "llama2" -seed "100" -batch_size 4 -gen_style "train_short" -no_format

    for style in "train_short" "active" "passive" "formal"
    do
        accelerate launch --config_file "./acc_config_dist.yaml" "main.py" -num_demo "5" -dataset "$dataset_name" -model_name "mistral" -seed "100" -batch_size 4 -gen_style "$style" -no_format
    done

    for style in "train_short" "active" "passive" "formal"
    do
        accelerate launch --config_file "./acc_config_dist.yaml" "main.py" -num_demo "5" -dataset "$dataset_name" -model_name "llama2" -seed "100" -batch_size 4 -gen_style "$style" -hint -no_format
    done
done
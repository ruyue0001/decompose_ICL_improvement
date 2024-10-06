#!/usr/bin/bash

for dataset_name in "glue_sst2" "glue_rte" "tweet_hate" "trec_" "ag_new" "glue_mrpc" "wnli" "medical_questions_pairs" "hs18"
do
    accelerate launch --config_file "./acc_config_dist.yaml" "main.py" -num_demo "0" -dataset "$dataset_name" -model_name "mistral" -seed "100" -batch_size 8 -no_format

    accelerate launch --config_file "./acc_config_dist.yaml" "main.py" -num_demo "0" -dataset "$dataset_name" -model_name "llama2" -seed "100" -batch_size 4 -no_format

    accelerate launch --config_file "./acc_config_dist.yaml" "main.py" -num_demo "0" -dataset "$dataset_name" -model_name "mistral" -seed "100" -batch_size 8

    accelerate launch --config_file "./acc_config_dist.yaml" "main.py" -num_demo "0" -dataset "$dataset_name" -model_name "llama2" -seed "100" -batch_size 4

    for seed in "13" "21" "42" "87" "100"
    do
        accelerate launch --config_file "./acc_config_dist.yaml" "main.py" -num_demo "5" -dataset "$dataset_name" -model_name "mistral" -seed "$seed" -batch_size 8
    done

    for seed in "13" "21" "42" "87" "100"
    do
        accelerate launch --config_file "./acc_config_dist.yaml" "main.py" -num_demo "5" -dataset "$dataset_name" -model_name "llama2" -seed "$seed" -batch_size 4 -hint
    done
done
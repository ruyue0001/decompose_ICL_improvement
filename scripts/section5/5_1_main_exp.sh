#!/usr/bin/bash

for dataset_name in "glue_sst2" "glue_rte" "tweet_hate" "trec_" "ag_new" "glue_mrpc" "wnli" "medical_questions_pairs" "hs18"
do
    for seed in "13" "21" "42" "87" "100"
    do
        accelerate launch --config_file "./acc_config_dist.yaml" "main.py" -num_demo "5" -dataset "$dataset_name" -model_name "mistral" -seed "$seed" -batch_size 8 -no_format
        
    done

    python txt2jsonl.py -num_demo "5" -dataset "$dataset_name" -model_name "mistral" -eval_setting "ICL"

    python handle_classification.py --task_name "$dataset_name" --model_name "mistral" --exp_name "ICL"

    for seed in "13" "21" "42" "87" "100"
    do
        accelerate launch --config_file "./acc_config_dist.yaml" "main.py" -num_demo "5" -dataset "$dataset_name" -model_name "llama2" -seed "$seed" -batch_size 4 -hint -no_format
    done

    python txt2jsonl.py -num_demo "5" -dataset "$dataset_name" -model_name "llama2" -eval_setting "ICL"

    python handle_classification.py --task_name "$dataset_name" --model_name "llama2" --exp_name "ICL"

done
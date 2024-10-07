#!/usr/bin/bash

for dataset_name in "glue_sst2" "glue_rte" "tweet_hate" "trec_" "ag_new" "glue_mrpc" "wnli" "medical_questions_pairs" "hs18"
do
    accelerate launch --config_file "./acc_config_dist.yaml" "main.py" -num_demo "0" -dataset "$dataset_name" -model_name "mistral" -seed "100" -batch_size 8 -no_format
    python txt2jsonl.py -num_demo "0" -dataset "$dataset_name" -model_name "mistral" -eval_setting "zeroshot"
    python handle_classification.py --task_name "$dataset_name" --model_name "mistral" --exp_name "zeroshot"

    accelerate launch --config_file "./acc_config_dist.yaml" "main.py" -num_demo "0" -dataset "$dataset_name" -model_name "llama2" -seed "100" -batch_size 4 -no_format
    python txt2jsonl.py -num_demo "0" -dataset "$dataset_name" -model_name "llama2" -eval_setting "zeroshot"
    python handle_classification.py --task_name "$dataset_name" --model_name "llama2" --exp_name "zeroshot"

    accelerate launch --config_file "./acc_config_dist.yaml" "main.py" -num_demo "0" -dataset "$dataset_name" -model_name "mistral" -seed "100" -batch_size 8
    python txt2jsonl.py -num_demo "0" -dataset "$dataset_name" -model_name "mistral" -eval_setting "DI"
    python handle_classification.py --task_name "$dataset_name" --model_name "mistral" --exp_name "DI"

    accelerate launch --config_file "./acc_config_dist.yaml" "main.py" -num_demo "0" -dataset "$dataset_name" -model_name "llama2" -seed "100" -batch_size 4
    python txt2jsonl.py -num_demo "0" -dataset "$dataset_name" -model_name "llama2" -eval_setting "DI"
    python handle_classification.py --task_name "$dataset_name" --model_name "llama2" --exp_name "DI"

    for seed in "13" "21" "42" "87" "100"
    do
        accelerate launch --config_file "./acc_config_dist.yaml" "main.py" -num_demo "5" -dataset "$dataset_name" -model_name "mistral" -seed "$seed" -batch_size 8
    done
    python txt2jsonl.py -num_demo "5" -dataset "$dataset_name" -model_name "mistral" -eval_setting "DI_ICL"
    python handle_classification.py --task_name "$dataset_name" --model_name "mistral" --exp_name "DI_ICL"

    for seed in "13" "21" "42" "87" "100"
    do
        accelerate launch --config_file "./acc_config_dist.yaml" "main.py" -num_demo "5" -dataset "$dataset_name" -model_name "llama2" -seed "$seed" -batch_size 4 -hint
    done
    python txt2jsonl.py -num_demo "5" -dataset "$dataset_name" -model_name "llama2" -eval_setting "DI_ICL"
    python handle_classification.py --task_name "$dataset_name" --model_name "llama2" --exp_name "DI_ICL"
done
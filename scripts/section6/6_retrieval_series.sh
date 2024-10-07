#!/usr/bin/bash

for dataset_name in "glue_sst2" "glue_rte" "tweet_hate" "trec_" "ag_new" "glue_mrpc" "wnli" "medical_questions_pairs" "hs18"
do
    for ret_setting in "ret" "homo-ret" "homo-rand" "hetero-ret" "hetero-rand"
    do
        accelerate launch --config_file "./acc_config_dist.yaml" "main.py" -num_demo "0" -dataset "$dataset_name" -model_name "mistral" -seed "100" -batch_size 8 -no_format -ret -ret_setting "$ret_setting"

        accelerate launch --config_file "./acc_config_dist.yaml" "main.py" -num_demo "0" -dataset "$dataset_name" -model_name "llama2" -seed "100" -batch_size 4 -no_format -ret -ret_setting "$ret_setting"

        python txt2jsonl.py -num_demo "5" -dataset "$dataset_name" -model_name "mistral" -eval_setting "retrieval_series/{$ret_setting}"
        python handle_classification.py --task_name "$dataset_name" --model_name "mistral" --exp_name "retrieval_series/{$ret_setting}"
    done


done
seeds="13 21 42 87 100"
dataset_path="data"

for n_samples in 1 3 5 7 10 15
do
    task_name="sst2"
    benchmark_name="glue"
    for seed in $seeds; do
    python generate_icl.py \
        --task_name $task_name \
        --benchmark_name $benchmark_name \
        --output_dir  $dataset_path/$n_samples/glue_sst2/k-$n_samples-seed-$seed \
        --seed $seed \
        --n_samples $n_samples
    done

    task_name="rte"
    benchmark_name="glue"

    for seed in $seeds; do
    python generate_icl.py \
        --task_name $task_name \
        --benchmark_name $benchmark_name \
        --output_dir $dataset_path/$n_samples/glue_rte/k-$n_samples-seed-$seed \
        --seed $seed \
        --n_samples $n_samples
    done

    task_name="mrpc"
    benchmark_name="glue"

    for seed in $seeds; do
    python generate_icl.py \
        --task_name $task_name \
        --benchmark_name $benchmark_name \
        --output_dir $dataset_path/$n_samples/glue_mrpc/k-$n_samples-seed-$seed \
        --seed $seed \
        --n_samples $n_samples
    done


    task_name="hate"
    benchmark_name="tweet_eval"

    for seed in $seeds; do
    python generate_icl.py \
        --task_name $task_name \
        --benchmark_name $benchmark_name \
        --output_dir $dataset_path/$n_samples/tweet_hate/k-$n_samples-seed-$seed \
        --seed $seed \
        --n_samples $n_samples
    done

    task_name="ag_news"
    benchmark_name="huggingface"

    for seed in $seeds; do
    python generate_icl.py \
        --task_name $task_name \
        --benchmark_name $benchmark_name \
        --output_dir $dataset_path/$n_samples/ag_new/k-$n_samples-seed-$seed \
        --seed $seed \
        --n_samples $n_samples
    done


    task_name="trec"
    benchmark_name="huggingface"

    for seed in $seeds; do
    python generate_icl.py \
        --task_name $task_name \
        --benchmark_name $benchmark_name \
        --output_dir $dataset_path/$n_samples/trec_/k-$n_samples-seed-$seed \
        --seed $seed \
        --n_samples $n_samples
    done

    task_name="wnli"
    benchmark_name="glue"
    for seed in $seeds; do
    python generate_icl.py \
        --task_name $task_name \
        --benchmark_name $benchmark_name \
        --output_dir  $dataset_path/$n_samples/wnli/k-$n_samples-seed-$seed \
        --seed $seed \
        --n_samples $n_samples
    done

    task_name="hate_speech18"
    benchmark_name="huggingface"
    for seed in $seeds; do
    python generate_icl.py \
        --task_name $task_name \
        --benchmark_name $benchmark_name \
        --output_dir  $dataset_path/$n_samples/hs18/k-$n_samples-seed-$seed \
        --seed $seed \
        --n_samples $n_samples
    done

    task_name="medical_questions_pairs"
    benchmark_name="huggingface"
    for seed in $seeds; do
    python generate_icl.py \
        --task_name $task_name \
        --benchmark_name $benchmark_name \
        --output_dir  $dataset_path/$n_samples/medical_questions_pairs/k-$n_samples-seed-$seed \
        --seed $seed \
        --n_samples $n_samples
    done
done


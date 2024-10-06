# NER
    task_name="conll2003"
    benchmark_name="huggingface"

    for seed in $seeds; do
    python generate_icl.py \
        --task_name $task_name \
        --benchmark_name $benchmark_name \
        --output_dir $dataset_path/$n_samples/conll03/k-$n_samples-seed-$seed \
        --seed $seed \
        --n_samples $n_samples
    done


    task_name="wnut_17"
    benchmark_name="huggingface"

    for seed in $seeds; do
    python generate_icl.py \
        --task_name $task_name \
        --benchmark_name $benchmark_name \
        --output_dir $dataset_path/$n_samples/wnut17/k-$n_samples-seed-$seed \
        --seed $seed \
        --n_samples $n_samples
    done
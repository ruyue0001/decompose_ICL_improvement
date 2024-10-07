## Does In-Context Learning Really Learn? Rethinking How Large Language Models Respond and Solve Tasks via In-Context Learning

This repository contains codes for COLM 2024 paper: [Does In-Context Learning Really Learn? Rethinking How Large Language Models Respond and Solve Tasks via In-Context Learning](https://openreview.net/pdf?id=i2oJjC0ESQ)

## Getting Started

### Enviorments

The codes were tested on:

    Python >= 3.10

    transformers >= 4.36

    datasets >= 2.20.0

    pytorch >= 2.1.0

    faiss-cpu >= 1.7.4

    pandas

    scikit-learn

    numpy

Optional dependencies include:

    accelerate >= 0.30.1 (For distributed inference, used together with deepspeed)

    deepspeed >= 0.12.6

### Datasets Preprocessing

We do not own the datasets evaluated in our experiments. Please download them via huggingface. 

`scripts/prepare_datasets/generate_dataset.sh` is for this purpose. It downloads the datasets from huggingface and do some pre-processing, including sampling the samples used for in-context demonstrations.

## Inference

`section5`, `section6` and `section7` folders contain the bash files used for experiments in correspoding sections of our paper. Replace `accelerate launch --config_file "./acc_config_dist.yaml"` with `python` to perform single-gpu inference. You need to set the batch size accordingly.

## Citation
```
@inproceedings{
    long2024does,
    title={Does In-Context Learning Really Learn? Rethinking How Large Language Models Respond and Solve Tasks via In-Context Learning},
    author={Quanyu Long and Yin Wu and Wenya Wang and Sinno Jialin Pan},
    booktitle={First Conference on Language Modeling},
    year={2024},
    url={https://openreview.net/forum?id=i2oJjC0ESQ}
}
```

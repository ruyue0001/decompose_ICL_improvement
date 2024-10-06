import argparse
import os
from glob import glob
import jsonlines

import sari
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from nltk.tokenize import word_tokenize

import evaluate
rouge = evaluate.load('rouge')
from rouge_score import rouge_scorer

def corpus_sari_score(src, pred, ref):
    scores = []
    for a, b, c in zip(src, pred, ref):
        scores += [sari.SARIsent(a, b, c)]
        # for EASSE metrics
        # c = [[k] for k in c]
        # scores += [corpus_sari([a], [b], c)]
    sari_avg = round(sum(scores) / len(scores), 4)
    print("SARI: ", sari_avg)
    return scores

def corpus_bleu1(pred_list, ref_list):
    return round(corpus_bleu(ref_list, pred_list, [1, 0, 0, 0]), 4)

def corpus_bleu2(pred_list, ref_list):
    return round(corpus_bleu(ref_list, pred_list, [0.5, 0.5, 0, 0]), 4)

def corpus_bleu3(pred_list, ref_list):
    return round(corpus_bleu(ref_list, pred_list, [1/3, 1/3, 1/3, 0]), 4)

def corpus_bleu4(pred_list, ref_list):
    return round(corpus_bleu(ref_list, pred_list, [0.25, 0.25, 0.25, 0.25]), 4)


def main(task, model, exp):
    # if model in ["gpt3", "chatgpt"]:
    #     file_path = os.path.join("exp2", exp, model, task, "s*.jsonl")
    # else:
    #     file_path = os.path.join("exp2", exp, model, task, "*_clean.jsonl")
    file_path = os.path.join("generation", exp, model, task, "*.jsonl")
    files = sorted(glob(file_path))
    # print (files)

    for file_name in files:
        print ("=============================")
        print ("Seed:", file_name)

        if task in ["common_gen", "roc_story_ending", "roc_story"]:
            pred_list = []
            ref_list = []
            count = 0
            with jsonlines.open(file_name) as f:
                for line in f.iter():
                    pred = line['output_text'].strip().lower()
                    ref = line['gold_label'].strip().lower()

                    if model == "mistral":
                        pred = pred.split('\n')[0]
                        pred = pred.split('(')[0].strip()

                    elif model == "llama2":
                        if '\n\n' in pred:
                            pred0 = pred.split('\n\n')[0]
                            pred1 = pred.split('\n\n')[1]
                            if corpus_bleu2([pred0],[[ref]]) > corpus_bleu2([pred1],[[ref]]):
                                pred = pred0
                            else:
                                pred = pred1

                    pred_list.append(word_tokenize(pred))
                    ref_list.append([word_tokenize(ref)])
                    count += 1


            print("Instances number:", count)
            # print("Corpus bleu1:", corpus_bleu1(pred_list, ref_list))
            # print("Corpus bleu2:", corpus_bleu2(pred_list, ref_list))
            # print("Corpus bleu3:", corpus_bleu3(pred_list, ref_list))
            # print("Corpus bleu4:", corpus_bleu4(pred_list, ref_list))
            print (corpus_bleu1(pred_list, ref_list), corpus_bleu2(pred_list, ref_list),corpus_bleu3(pred_list, ref_list),corpus_bleu4(pred_list, ref_list))

        elif task in ['reddit', 'samsum']:
            pred_list = []
            ref_list = []
            count = 0
            with jsonlines.open(file_name) as f:
                for line in f.iter():
                    pred = line['output_text'].strip().lower()
                    ref = line['gold_label'].strip().lower()

                    if model == "mistral":
                        pred = pred.split('\n')[0]
                        pred = pred.split('(')[0].strip()

                    elif model == "llama2":
                        if '\n\n' in pred:
                            pred0 = pred.split('\n\n')[0]
                            pred1 = pred.split('\n\n')[1]
                            if rouge.compute(predictions=[pred0], references=[[ref]])['rougeL'] > rouge.compute(predictions=[pred1], references=[[ref]])['rougeL']:
                                pred = pred0
                            else:
                                pred = pred1

                    pred_list.append(pred)
                    ref_list.append([ref])
                    count += 1

            print("Instances number:", count)
            results = rouge.compute(predictions=pred_list, references=ref_list)
            print(results)



        elif task in ['wikiauto']:
            pred_list = []
            ref_list = []
            source_list= []
            count = 0
            with jsonlines.open(file_name) as f:
                for line in f.iter():
                    pred = line['output_text'].strip().lower()
                    ref = line['gold_label'].strip().lower()
                    source = line['input_query'].strip().lower()

                    pred_list.append(pred)
                    ref_list.append(ref)
                    source_list.append(source)
                    count += 1

            print("Instances number:", count)
            sari = corpus_sari_score(source_list, pred_list, ref_list)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generation tasks evaluation.")
    parser.add_argument("--task_name", type=str, default="common_gen", choices=['common_gen','roc_story_ending', 'roc_story', 'wikiauto', 'reddit', 'samsum'])
    parser.add_argument("--model_name", type=str, default="chatgpt", choices=['mistral', 'gpt3', 'chatgpt', 'llama2'])
    parser.add_argument("--exp_name", type=str, default="acc0", choices=['acc0', 'acc1', 'acc2', 'acc3', 'flip', 'retrieval','diverse'])

    args = parser.parse_args()

    if args.exp_name == "acc0":
        print ("Prompt: only gives the {INS_TASK}")
    elif args.exp_name == "acc1":
        print("Prompt: only gives the {INS_TASK}{INS_FORMAT}")
    elif args.exp_name in ["acc2", "flip"]:
        print("Prompt: only gives the {INS_TASK}{DEMO}")
    else:
        print("Prompt: gives the {INS_TASK}{INS_FORMAT}{DEMO}")


    main(args.task_name, args.model_name, args.exp_name)
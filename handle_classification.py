import argparse
import os
from glob import glob
import jsonlines

from post_process import sst2, financial_pb, poem, wnli, super_glue_cb, glue_rte, tweet_hate, tweet_feminist, tweet_atheism, ag_news, trec, mrpc, medical_question, hs18


def main(task, model, exp, num_k):
    # if model in ["gpt3", "chatgpt"]:
    #     file_path = os.path.join("results", exp, model, task, "s*.jsonl")
    # else:
    #     file_path = os.path.join("results", exp, model, task, "*_clean.jsonl")
    if exp == 'num_k':
        file_path = os.path.join('results', exp, model, num_k, task, "s*.jsonl")
    else:
        file_path = os.path.join("results", exp, model, task, "s*.jsonl")
    files = sorted(glob(file_path))
    # print (files)

    for file_name in files:
        print ("=============================")
        print ("Seed:", file_name)
        if model in ["mistral", "llama2"] and exp not in ["DI", "DI_ICL"]:
            if "clean" not in file_name:
                continue
        if task == "glue_sst2":
            sst2.eval(file_name, model, exp)

            if exp in ["ICL", "DI_ICL", "Incorrect_Label", "retrieval", "reverse_rand", "reverse_retrieval"]:
                with_out_ICL_file_name = os.path.join("results", "zeroshot", model, task)
                if model in ["mistral","llama2"]:
                    if model == "mistral":
                        with_out_ICL_file_name += "/s13_clean.jsonl"
                    else:
                        with_out_ICL_file_name += "/s100_clean.jsonl"
                else:
                    with_out_ICL_file_name += "/s100.jsonl"
                sst2.eval_component(with_out_ICL_file_name, file_name, model)

        # elif task == "financial_pb":
        #     for pp_level in ["naive", "normal", "human-eval"]:
        #         cnt_total_instance, cnt_in_format, cnt_in_format_right_pred = financial_pb.eval(file_name, model, pp_level)
        #         print("Total instances:", cnt_total_instance)
        #         print("Number for instances in [", pp_level, "] post-processing:", cnt_in_format)
        #         print("Number for correct predictions in [", pp_level, "] post-processing:", cnt_in_format_right_pred)
        #
        # elif task == "poem":
        #     for pp_level in ["naive", "normal", "human-eval"]:
        #         cnt_total_instance, cnt_in_format, cnt_in_format_right_pred = poem.eval(file_name, model, pp_level)
        #         print("Total instances:", cnt_total_instance)
        #         print("Number for instances in [", pp_level, "] post-processing:", cnt_in_format)
        #         print("Number for correct predictions in [", pp_level, "] post-processing:", cnt_in_format_right_pred)

        elif task == "wnli":
            wnli.eval(file_name, model, exp)

            if exp in ["ICL", "DI_ICL", "Incorrect_Label", "retrieval", "reverse_rand", "reverse_retrieval"]:
                with_out_ICL_file_name = os.path.join("results", "zeroshot", model, task)
                if model in ["mistral", "llama2"]:
                    if model == "mistral":
                        with_out_ICL_file_name += "/s100_clean.jsonl"
                    else:
                        with_out_ICL_file_name += "/s100_clean.jsonl"
                else:
                    with_out_ICL_file_name += "/s100.jsonl"
                wnli.eval_component(with_out_ICL_file_name, file_name, model)

        elif task == "super_glue_cb":
            for pp_level in ["naive", "normal", "human-eval"]:
                cnt_total_instance, cnt_in_format, cnt_in_format_right_pred = super_glue_cb.eval(file_name, model, pp_level)
                print("Total instances:", cnt_total_instance)
                print("Number for instances in [", pp_level, "] post-processing:", cnt_in_format)
                print("Number for correct predictions in [", pp_level, "] post-processing:", cnt_in_format_right_pred)

        elif task == "glue_rte":
            glue_rte.eval(file_name, model, exp)

            if exp in ["ICL", "DI_ICL", "Incorrect_Label", "retrieval", "reverse_rand", "reverse_retrieval"]:
                with_out_ICL_file_name = os.path.join("results", "zeroshot", model, task)
                if model in ["mistral", "llama2"]:
                    if model == "mistral":
                        with_out_ICL_file_name += "/s13_clean.jsonl"
                    else:
                        with_out_ICL_file_name += "/s100_clean.jsonl"
                else:
                    with_out_ICL_file_name += "/s100.jsonl"
                glue_rte.eval_component(with_out_ICL_file_name, file_name, model)

        elif task == "medical_questions_pairs":
            medical_question.eval(file_name, model, exp)

            if exp in ["ICL", "DI_ICL", "Incorrect_Label", "retrieval", "reverse_rand", "reverse_retrieval"]:
                with_out_ICL_file_name = os.path.join("results", "zeroshot", model, task)
                if model in ["mistral", "llama2"]:
                    if model == "mistral":
                        with_out_ICL_file_name += "/s100_clean.jsonl"
                    else:
                        with_out_ICL_file_name += "/s100_clean.jsonl"
                else:
                    with_out_ICL_file_name += "/s100.jsonl"
                medical_question.eval_component(with_out_ICL_file_name, file_name, model)

        elif task == "glue_mrpc":
            mrpc.eval(file_name, model, exp)

            if exp in ["ICL", "DI_ICL", "Incorrect_Label", "retrieval", "reverse_rand", "reverse_retrieval"]:
                with_out_ICL_file_name = os.path.join("results", "zeroshot", model, task)
                if model in ["mistral", "llama2"]:
                    if model == "mistral":
                        with_out_ICL_file_name += "/s100_clean.jsonl"
                    else:
                        with_out_ICL_file_name += "/s100_clean.jsonl"
                else:
                    with_out_ICL_file_name += "/s100.jsonl"
                mrpc.eval_component(with_out_ICL_file_name, file_name, model)

        elif task == "tweet_hate":
            tweet_hate.eval(file_name, model, exp)

            if exp in ["ICL", "DI_ICL", "Incorrect_Label", "retrieval", "reverse_rand", "reverse_retrieval"]:
                with_out_ICL_file_name = os.path.join("results", "zeroshot", model, task)
                if model in ["mistral", "llama2"]:
                    if model == "mistral":
                        with_out_ICL_file_name += "/s13_clean.jsonl"
                    else:
                        with_out_ICL_file_name += "/s100_clean.jsonl"
                else:
                    with_out_ICL_file_name += "/s100.jsonl"
                tweet_hate.eval_component(with_out_ICL_file_name, file_name, model)

        elif task == "hs18":
            hs18.eval(file_name, model, exp)

            if exp in ["ICL", "DI_ICL", "Incorrect_Label", "retrieval", "reverse_rand", "reverse_retrieval"]:
                with_out_ICL_file_name = os.path.join("results", "zeroshot", model, task)
                if model in ["mistral", "llama2"]:
                    if model == "mistral":
                        with_out_ICL_file_name += "/s100_clean.jsonl"
                    else:
                        with_out_ICL_file_name += "/s100_clean.jsonl"
                else:
                    with_out_ICL_file_name += "/s100.jsonl"
                hs18.eval_component(with_out_ICL_file_name, file_name, model)

        elif task == "tweet_stance_feminist":
            tweet_feminist.eval(file_name, model, exp)

            if exp in ["ICL", "Incorrect_Label"]:
                with_out_ICL_file_name = os.path.join("results", "zeroshot", model, task)
                if model in ["mistral", "llama2"]:
                    if model == "mistral":
                        with_out_ICL_file_name += "/s13_clean.jsonl"
                    else:
                        with_out_ICL_file_name += "/s100_clean.jsonl"
                else:
                    with_out_ICL_file_name += "/s100.jsonl"
                tweet_feminist.eval_component(with_out_ICL_file_name, file_name, model)

        elif task == "tweet_stance_atheism":
            for pp_level in ["naive", "normal", "human-eval"]:
                cnt_total_instance, cnt_in_format, cnt_in_format_right_pred = tweet_atheism.eval(file_name, model, pp_level)
                print("Total instances:", cnt_total_instance)
                print("Number for instances in [", pp_level, "] post-processing:", cnt_in_format)
                print("Number for correct predictions in [", pp_level, "] post-processing:", cnt_in_format_right_pred)

        elif task == "ag_new":
            ag_news.eval(file_name, model, exp)

            if exp in ["ICL", "DI_ICL", "Incorrect_Label", "retrieval", "reverse_rand", "reverse_retrieval"]:
                with_out_ICL_file_name = os.path.join("results", "zeroshot", model, task)
                if model in ["mistral", "llama2"]:
                    if model == "mistral":
                        with_out_ICL_file_name += "/s100_clean.jsonl"
                    else:
                        with_out_ICL_file_name += "/s100_clean.jsonl"
                else:
                    with_out_ICL_file_name += "/s100.jsonl"
                ag_news.eval_component(with_out_ICL_file_name, file_name, model)

        elif task == "trec_":
            trec.eval(file_name, model, exp)

            if exp in ["ICL", "DI_ICL", "Incorrect_Label", "retrieval", "reverse_rand", "reverse_retrieval"]:
                with_out_ICL_file_name = os.path.join("results", "zeroshot", model, task)
                if model in ["mistral", "llama2"]:
                    if model == "mistral":
                        with_out_ICL_file_name += "/s13_clean.jsonl"
                    else:
                        with_out_ICL_file_name += "/s100_clean.jsonl"
                else:
                    with_out_ICL_file_name += "/s100.jsonl"
                trec.eval_component(with_out_ICL_file_name, file_name, model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply perturbations to a sentence.")
    parser.add_argument("--task_name", type=str, default="glue_sst2", choices=['ag_new', 'bc2gm', 'conll03', 'financial_pb', 'glue_rte', 'glue_sst2', 'poem', 'super_glue_cb', 'trec_', 'tweet_hate', 'tweet_stance_atheism', 'tweet_stance_feminist', 'wnut17', 'wnli', 'medical_questions_pairs', 'glue_mrpc', 'hs18'])
    parser.add_argument("--model_name", type=str, default="chatgpt", choices=['mistral', 'flant5', 'gpt3', 'chatgpt', 'llama2'])
    parser.add_argument("--exp_name", type=str, default="zeroshot", choices=['zeroshot', 'DI', 'ICL', 'DI_ICL', 'Incorrect_Label', 'retrieval','diverse', 'num_k', 'reverse_rand', 'reverse_retrieval'])
    parser.add_argument('--num_k', type=str, default='5')

    args = parser.parse_args()

    if args.exp_name == "zeroshot":
        print ("Prompt: only gives the {INS_TASK}")
    elif args.exp_name == "DI":
        print("Prompt: only gives the {INS_TASK}{INS_FORMAT}")
    elif args.exp_name in ["ICL", "Incorrect_Label"]:
        print("Prompt: only gives the {INS_TASK}{DEMO}")
    else:
        print("Prompt: gives the {INS_TASK}{INS_FORMAT}{DEMO}")


    main(args.task_name, args.model_name, args.exp_name, args.num_k)
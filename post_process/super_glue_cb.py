import jsonlines

def eval(file_name, model, pp_level = 'naive'):
    cnt_total_instance = 0
    cnt_in_format = 0
    cnt_in_format_right_pred = 0
    assert pp_level in ['naive', 'normal', 'human-eval']
    with jsonlines.open(file_name) as f:
        for line in f.iter():
            cnt_total_instance += 1
            output = line['output_text'].lower()
            label = line["gold_label"].lower()
            if model == "chatgpt":
                if pp_level == "naive":
                    if output in ['entailment', 'neutral', 'contradiction']:
                        cnt_in_format += 1
                    # else:
                    #     print ("[",output,"] is not in [", pp_level, "] evaluation.")
                    if output == label:
                        cnt_in_format_right_pred += 1
                    # else:
                    #     print (output, label)

                elif pp_level == "normal":
                    # if 'positive' in output and "negative" in output:
                    #     print (output)
                    if 'entailment' in output or 'contradiction' in output or 'neutral' in output:
                        cnt_in_format += 1
                    # else:
                    #     print ("[",output,"] is not in [", pp_level, "] evaluation.")
                    if label in output:
                        cnt_in_format_right_pred += 1

                elif pp_level == "human-eval":
                    if 'entailment' in output and 'contradiction' not in output and 'neutral' not in output:
                        cnt_in_format += 1
                        if label == 'entailment':
                            cnt_in_format_right_pred += 1
                    elif 'contradiction' in output and 'entailment' not in output and 'neutral' not in output:
                        cnt_in_format += 1
                        if label == 'contradiction':
                            cnt_in_format_right_pred += 1
                    elif 'refute' in output and 'entailment' not in output and 'neutral' not in output:
                        cnt_in_format += 1
                        if label == 'contradiction':
                            cnt_in_format_right_pred += 1
                    elif 'false' in output and 'entailment' not in output and 'neutral' not in output:
                        cnt_in_format += 1
                        if label == 'contradiction':
                            cnt_in_format_right_pred += 1
                    elif 'neutral' in output and 'entailment' not in output and 'contradiction' not in output:
                        cnt_in_format += 1
                        if label == 'neutral':
                            cnt_in_format_right_pred += 1
                    else:
                        print (output)
                        continue

            elif model == "flant5":
                if pp_level == "naive":
                    if output in ['entailment', 'neutral', 'contradiction']:
                        cnt_in_format += 1
                    # else:
                        # print ("[",output,"] is not in [", pp_level, "] evaluation.")
                    if output == label:
                        cnt_in_format_right_pred += 1

                elif pp_level == "normal":
                    # if 'positive' in output and "negative" in output:
                    #     print (output)
                    if 'entailment' in output or 'contradiction' in output or 'neutral' in output:
                        cnt_in_format += 1
                    # else:
                    #     print ("[",output,"] is not in [", pp_level, "] evaluation.")
                    if label in output:
                        cnt_in_format_right_pred += 1

                elif pp_level == "human-eval":
                    if 'entailment' in output and 'contradiction' not in output and 'neutral' not in output:
                        cnt_in_format += 1
                        if label == 'entailment':
                            cnt_in_format_right_pred += 1
                    elif 'contradiction' in output and 'entailment' not in output and 'neutral' not in output:
                        cnt_in_format += 1
                        if label == 'contradiction':
                            cnt_in_format_right_pred += 1
                    elif 'neutral' in output and 'entailment' not in output and 'contradiction' not in output:
                        cnt_in_format += 1
                        if label == 'neutral':
                            cnt_in_format_right_pred += 1
                    elif 'yes' in output and 'no' not in output and 'it is not possible to tell' not in output:
                        cnt_in_format += 1
                        if label == 'entailment':
                            cnt_in_format_right_pred += 1
                    elif 'no' in output and 'yes' not in output and 'it is not possible to tell' not in output:
                        cnt_in_format += 1
                        if label == 'contradiction':
                            cnt_in_format_right_pred += 1
                    elif 'it is not possible to tell' in output:
                        cnt_in_format += 1
                        if label == 'neutral':
                            cnt_in_format_right_pred += 1
                    else:
                        print (output)
                        continue

    return cnt_total_instance, cnt_in_format, cnt_in_format_right_pred
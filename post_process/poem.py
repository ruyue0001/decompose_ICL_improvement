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
                    if output in ['positive', 'negative', 'no_impact']:
                        cnt_in_format += 1
                    # else:
                    #     print ("[",output,"] is not in [", pp_level, "] evaluation.")
                    if output == label:
                        cnt_in_format_right_pred += 1

                elif pp_level == "normal":
                    # if 'positive' in output and "negative" in output:
                    #     print (output)
                    if 'positive' in output or 'negative' in output or 'no_impact' in output:
                        cnt_in_format += 1
                    # else:
                    #     print ("[",output,"] is not in [", pp_level, "] evaluation.")
                    if label in output:
                        cnt_in_format_right_pred += 1

                elif pp_level == "human-eval":
                    if "positive" in output and "negative" not in output and "no_impact" not in output:
                        cnt_in_format += 1
                        if label == "positive":
                            cnt_in_format_right_pred += 1
                    elif "negative" in output and "positive" not in output and "no_impact" not in output:
                        cnt_in_format += 1
                        if label == "negative":
                            cnt_in_format_right_pred += 1
                    elif "no_impact" in output and "positive" not in output and "negative" not in output:
                        cnt_in_format += 1
                        if label == "no_impact":
                            cnt_in_format_right_pred += 1
                    elif "neutral" in output and "positive" not in output and "negative" not in output:
                        cnt_in_format += 1
                        if label == "no_impact":
                            cnt_in_format_right_pred += 1
                    else:
                        print (output)
                        continue

            elif model == "flant5":
                if pp_level == "naive":
                    if output in ['positive', 'negative', 'no_impact']:
                        cnt_in_format += 1
                    # else:
                    #     print ("[",output,"] is not in [", pp_level, "] evaluation.")
                    if output == label:
                        cnt_in_format_right_pred += 1

                elif pp_level == "normal":
                    # if 'positive' in output and "negative" in output:
                    #     print (output)
                    if 'positive' in output or 'negative' in output or 'no_impact' in output:
                        cnt_in_format += 1
                    # else:
                    #     print ("[",output,"] is not in [", pp_level, "] evaluation.")
                    if label in output:
                        cnt_in_format_right_pred += 1

                elif pp_level == "human-eval":
                    if "positive" in output and "negative" not in output and "no_imapct" not in output:
                        cnt_in_format += 1
                        if label == "positive":
                            cnt_in_format_right_pred += 1
                    elif "negative" in output and "positive" not in output and "no_impact" not in output:
                        cnt_in_format += 1
                        if label == "negative":
                            cnt_in_format_right_pred += 1
                    elif "no_impact" in output and "positive" not in output and "negative" not in output:
                        cnt_in_format += 1
                        if label == "no_impact":
                            # print (output)
                            cnt_in_format_right_pred += 1
                            # print ("======")
                    else:
                        print (output)
                        continue

    return cnt_total_instance, cnt_in_format, cnt_in_format_right_pred
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
                    if "favor" in output:
                        cnt_in_format += 1
                        # print (output)
                        if label == "favor":
                            cnt_in_format_right_pred += 1
                    elif "against" in output:
                        cnt_in_format += 1
                        # print(output)
                        if label == "against":
                            cnt_in_format_right_pred += 1
                    elif "none" in output:
                        cnt_in_format += 1
                        # print(output)
                        if label == "none":
                            cnt_in_format_right_pred += 1
                    else:
                        # print(output)
                        continue

                elif pp_level == "normal":
                    if "favor" in output:
                        cnt_in_format += 1
                        # print (output)
                        if label == "favor":
                            cnt_in_format_right_pred += 1
                    elif "against" in output:
                        cnt_in_format += 1
                        # print(output)
                        if label == "against":
                            cnt_in_format_right_pred += 1
                    elif "none" in output:
                        cnt_in_format += 1
                        # print(output)
                        if label == "none":
                            cnt_in_format_right_pred += 1
                    elif "positive" in output:
                        cnt_in_format += 1
                        # print (output)
                        if label == "favor":
                            cnt_in_format_right_pred += 1
                    elif "negative" in output:
                        cnt_in_format += 1
                        # print(output)
                        if label == "against":
                            cnt_in_format_right_pred += 1
                    elif "neutral" in output:
                        cnt_in_format += 1
                        # print(output)
                        if label == "none":
                            cnt_in_format_right_pred += 1
                    else:
                        # print(output)
                        continue

                elif pp_level == "human-eval":
                    if "favor" in output:
                        cnt_in_format += 1
                        # print (output)
                        if label == "favor":
                            cnt_in_format_right_pred += 1
                    elif "against" in output:
                        cnt_in_format += 1
                        # print(output)
                        if label == "against":
                            cnt_in_format_right_pred += 1
                    elif "none" in output:
                        cnt_in_format += 1
                        # print(output)
                        if label == "none":
                            cnt_in_format_right_pred += 1
                    elif "positive" in output:
                        cnt_in_format += 1
                        # print (output)
                        if label == "favor":
                            cnt_in_format_right_pred += 1
                    elif "negative" in output:
                        cnt_in_format += 1
                        # print(output)
                        if label == "against":
                            cnt_in_format_right_pred += 1
                    elif "neutral" in output or "does not express" in output or 'irrelevant' in output:
                        cnt_in_format += 1
                        # print(output)
                        if label == "none":
                            cnt_in_format_right_pred += 1
                    elif "support" in output:
                        cnt_in_format += 1
                        # print (output)
                        if label == "favor":
                            cnt_in_format_right_pred += 1
                    elif "for" in output:
                        cnt_in_format += 1
                        # print (output)
                        if label == "favor":
                            cnt_in_format_right_pred += 1
                    else:
                        print(output)
                        continue

            elif model == "flant5":
                if pp_level == "naive":
                    if "favor" in output:
                        cnt_in_format += 1
                        # print (output)
                        if label == "favor":
                            cnt_in_format_right_pred += 1
                    elif "against" in output:
                        cnt_in_format += 1
                        # print(output)
                        if label == "against":
                            cnt_in_format_right_pred += 1
                    elif "none" in output:
                        cnt_in_format += 1
                        # print(output)
                        if label == "none":
                            cnt_in_format_right_pred += 1
                    else:
                        # print(output)
                        continue

                elif pp_level == "normal":
                    if "favor" in output:
                        cnt_in_format += 1
                        # print (output)
                        if label == "favor":
                            cnt_in_format_right_pred += 1
                    elif "against" in output:
                        cnt_in_format += 1
                        # print(output)
                        if label == "against":
                            cnt_in_format_right_pred += 1
                    elif "none" in output:
                        cnt_in_format += 1
                        # print(output)
                        if label == "none":
                            cnt_in_format_right_pred += 1
                    elif "positive" in output:
                        cnt_in_format += 1
                        # print (output)
                        if label == "favor":
                            cnt_in_format_right_pred += 1
                    elif "negative" in output:
                        cnt_in_format += 1
                        # print(output)
                        if label == "against":
                            cnt_in_format_right_pred += 1
                    elif "neutral" in output:
                        cnt_in_format += 1
                        # print(output)
                        if label == "none":
                            cnt_in_format_right_pred += 1
                    else:
                        # print(output)
                        continue

                elif pp_level == "human-eval":
                    if "favor" in output:
                        cnt_in_format += 1
                        # print (output)
                        if label == "favor":
                            cnt_in_format_right_pred += 1
                    elif "against" in output:
                        cnt_in_format += 1
                        # print(output)
                        if label == "against":
                            cnt_in_format_right_pred += 1
                    elif "none" in output:
                        cnt_in_format += 1
                        # print(output)
                        if label == "none":
                            cnt_in_format_right_pred += 1
                    elif "positive" in output:
                        cnt_in_format += 1
                        # print (output)
                        if label == "favor":
                            cnt_in_format_right_pred += 1
                    elif "negative" in output:
                        cnt_in_format += 1
                        # print(output)
                        if label == "against":
                            cnt_in_format_right_pred += 1
                    elif "neutral" in output:
                        cnt_in_format += 1
                        # print(output)
                        if label == "none":
                            cnt_in_format_right_pred += 1
                    elif "support" in output:
                        cnt_in_format += 1
                        # print (output)
                        if label == "favor":
                            cnt_in_format_right_pred += 1
                    elif "for" in output:
                        cnt_in_format += 1
                        # print (output)
                        if label == "favor":
                            cnt_in_format_right_pred += 1
                    else:
                        print(output)
                        continue

    return cnt_total_instance, cnt_in_format, cnt_in_format_right_pred
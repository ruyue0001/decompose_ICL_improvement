import jsonlines
import re

def is_mistral_positive(sent):
    if sent.startswith("positive"):
        return True
    elif "label: positive" in sent:
        return True
    elif "as positive" in sent or "as \"positive\"" in sent:
        return True
    elif "would be positive" in sent or "would be \"positive\"" in sent:
        return True
    else:
        return False
def is_mistral_negative(sent):
    if sent.startswith("negative"):
        return True
    elif "as negative" in sent or "as \"negative\"" in sent:
        return True
    elif "would be negative" in sent or "would be \"negative\"" in sent:
        return True
    else:
        return False

def is_llama2_positive(sent):
    if sent.startswith("positive"):
        return True
    elif "label: positive" in sent or "\npositive" in sent:
        return True
    elif "as positive" in sent or "as \"positive\"" in sent:
        return True
    elif "would be positive" in sent or "would be \"positive\"" in sent:
        return True
    else:
        return False

def is_llama2_negative(sent):
    if sent.startswith("negative"):
        return True
    elif "label: negative" in sent or "\nnegative" in sent:
        return True
    elif "as negative" in sent or "as \"negative\"" in sent:
        return True
    elif "would be negative" in sent or "would be \"negative\"" in sent:
        return True
    else:
        return False

def eval_component(with_out_ICL_path, with_ICL_path, model):
    w_icl_in_space_in_format = set()
    w_icl_in_space_in_format_right_pred = set()
    w_icl_in_space_out_format = set()
    w_icl_in_space = set()
    w_icl_out_space = set()
    wo_icl_in_space_in_format = set()
    wo_icl_in_space_in_format_right_pred = set()
    wo_icl_in_space_out_format = set()
    wo_icl_in_space = set()
    wo_icl_out_space = set()
    with jsonlines.open(with_out_ICL_path) as f_wo_icl:
        with jsonlines.open(with_ICL_path) as f_w_icl:
            cnt = 0
            for wo_icl_line, w_icl_line in zip(f_wo_icl, f_w_icl):
                cnt += 1
                wo_icl_output = wo_icl_line['output_text'].strip().lower()
                w_icl_output = w_icl_line['output_text'].strip().lower()
                if model == "mistral":
                    if "clean" in with_ICL_path:
                        assert wo_icl_line["input_query"] == w_icl_line["input_query"]
                    w_icl_output = w_icl_output.split('\n')[0]
                    w_icl_output = w_icl_output.split('(')[0].strip()
                    assert  w_icl_output != ''
                    # print (w_icl_output)
                # print (wo_icl_line, '\n-----\n', w_icl_line, '\n------\n')
                assert  wo_icl_line["gold_label"].strip().lower() == w_icl_line["gold_label"].strip().lower()
                label = wo_icl_line["gold_label"].strip().lower()

                if model in ["chatgpt", "gpt3"]:
                    if ("positive" in wo_icl_output or "negative" in wo_icl_output) and "neutral" not in wo_icl_output:
                        wo_icl_in_space.add(cnt)
                        if wo_icl_output == "positive" or wo_icl_output == "negative":
                            wo_icl_in_space_in_format.add(cnt)
                            if wo_icl_output == label:
                                wo_icl_in_space_in_format_right_pred.add(cnt)
                        else:
                            # print (wo_icl_output)
                            wo_icl_in_space_out_format.add(cnt)
                    else:
                        wo_icl_out_space.add(cnt)

                    if ("positive" in w_icl_output or "negative" in w_icl_output) and "neutral" not in w_icl_output:
                        w_icl_in_space.add(cnt)
                        if w_icl_output == "positive" or w_icl_output == "negative":
                            w_icl_in_space_in_format.add(cnt)
                            if w_icl_output == label:
                                w_icl_in_space_in_format_right_pred.add(cnt)
                        else:
                            w_icl_in_space_out_format.add(cnt)
                    else:
                        w_icl_out_space.add(cnt)

                elif model == "mistral":
                    if ("positive" in wo_icl_output or "negative" in wo_icl_output) and "neutral" not in wo_icl_output and "mix" not in wo_icl_output:
                        wo_icl_in_space.add(cnt)
                        if is_mistral_positive(wo_icl_output) or is_mistral_negative(wo_icl_output):
                            wo_icl_in_space_in_format.add(cnt)
                            if (is_mistral_positive(wo_icl_output) and label == "positive") or (is_mistral_negative(wo_icl_output) and label == "negative"):
                                wo_icl_in_space_in_format_right_pred.add(cnt)
                        else:
                            # print ("==========")
                            # print (wo_icl_output)
                            wo_icl_in_space_out_format.add(cnt)
                    else:
                        wo_icl_out_space.add(cnt)

                    if ("positive" in w_icl_output or "negative" in w_icl_output) and "neutral" not in w_icl_output and "mix" not in w_icl_output:
                        w_icl_in_space.add(cnt)
                        if is_mistral_positive(w_icl_output) or is_mistral_negative(w_icl_output):
                            w_icl_in_space_in_format.add(cnt)
                            if (is_mistral_positive(w_icl_output) and label == "positive") or (
                                is_mistral_negative(w_icl_output) and label == "negative"):
                                w_icl_in_space_in_format_right_pred.add(cnt)
                        else:
                            w_icl_in_space_out_format.add(cnt)
                    else:
                        w_icl_out_space.add(cnt)

                elif model == "llama2":
                    if ("positive" in wo_icl_output or "negative" in wo_icl_output) and "neutral" not in wo_icl_output and "mix" not in wo_icl_output:
                        wo_icl_in_space.add(cnt)
                        if is_llama2_positive(wo_icl_output) or is_llama2_negative(wo_icl_output):
                            wo_icl_in_space_in_format.add(cnt)
                            if (is_llama2_positive(wo_icl_output) and label == "positive") or (
                                    is_llama2_negative(wo_icl_output) and label == "negative"):
                                wo_icl_in_space_in_format_right_pred.add(cnt)
                        else:
                            # print ("==========")
                            # print (wo_icl_output)
                            wo_icl_in_space_out_format.add(cnt)
                    else:
                        wo_icl_out_space.add(cnt)

                    if ("positive" in w_icl_output or "negative" in w_icl_output) and "neutral" not in w_icl_output and "mix" not in w_icl_output:
                        w_icl_in_space.add(cnt)
                        if is_llama2_positive(w_icl_output) or is_llama2_negative(w_icl_output):
                            w_icl_in_space_in_format.add(cnt)
                            if (is_llama2_positive(w_icl_output) and label == "positive") or (
                                    is_llama2_negative(w_icl_output) and label == "negative"):
                                w_icl_in_space_in_format_right_pred.add(cnt)
                        else:
                            w_icl_in_space_out_format.add(cnt)
                    else:
                        w_icl_out_space.add(cnt)



    print("Total instance:", cnt)
    print ("Without ICL, file path:", with_out_ICL_path)
    print("Without ICL, in space number:", len(wo_icl_in_space))
    print("Without ICL, out space number:", len(wo_icl_out_space))
    print("Without ICL, in space and in format number:", len(wo_icl_in_space_in_format))
    print("Without ICL, in space but out of format number:", len(wo_icl_in_space_out_format))

    print("With ICL, in space number:", len(w_icl_in_space))
    print("With ICL, out space number:", len(w_icl_out_space))
    print("With ICL, in space and in format number:", len(w_icl_in_space_in_format))
    print("With ICL, in space but out of format number:", len(w_icl_in_space_out_format))


    # with icl, in space and in format now becomes bigger, including original part of in_space_out_format, and part of out_space
    # wo_icl_in_space_in_format - w_icl_in_space_in_format, is very small, we can overlook it


    # out of space
    # print (len(wo_icl_out_space))
    # print (len(w_icl_out_space))
    # print (len(wo_icl_out_space - w_icl_out_space))
    # print (len(w_icl_out_space - wo_icl_out_space))
    # print (len(wo_icl_out_space & w_icl_out_space))

    new_ISIF_OOS = wo_icl_out_space - w_icl_out_space
    print("new ISIF from OOS:", len(new_ISIF_OOS))
    new_OOS_ISIF = w_icl_out_space - wo_icl_out_space
    print("new OOS from ISIF: -", len(new_OOS_ISIF))
    with jsonlines.open(with_out_ICL_path) as f_wo_icl:
        with jsonlines.open(with_ICL_path) as f_w_icl:
            cnt = 0
            cnt_org_oos_icl_right = 0
            cnt_org_right_icl_oos = 0
            for wo_icl_line, w_icl_line in zip(f_wo_icl, f_w_icl):
                cnt += 1
                wo_icl_output = wo_icl_line['output_text'].strip().lower()
                w_icl_output = w_icl_line['output_text'].strip().lower()
                if model == "mistral":
                    if "clean" in with_ICL_path:
                        assert wo_icl_line["input_query"] == w_icl_line["input_query"]
                    w_icl_output = w_icl_output.split('\n')[0]
                    w_icl_output = w_icl_output.split('(')[0].strip()
                    assert w_icl_output != ''
                    # print (w_icl_output)

                assert wo_icl_line["gold_label"].strip().lower() == w_icl_line["gold_label"].strip().lower()
                label = wo_icl_line["gold_label"].strip().lower()

                if cnt in new_ISIF_OOS: # this part contributes to the performance
                    # print (wo_icl_output, w_icl_output, label)
                    if model in ["chatgpt", "gpt3"]:
                        if w_icl_output == label:
                            cnt_org_oos_icl_right += 1
                    elif model == "mistral":
                        if (is_mistral_positive(w_icl_output) and label == "positive") or (is_mistral_negative(w_icl_output) and label == "negative"):
                            cnt_org_oos_icl_right += 1
                    elif model == "llama2":
                        if (is_llama2_positive(w_icl_output) and label == "positive") or (
                                is_llama2_negative(w_icl_output) and label == "negative"):
                            cnt_org_oos_icl_right += 1
                elif cnt in new_OOS_ISIF: #this part decrease the performance
                    if model in ["chatgpt", "gpt3"]:
                        if wo_icl_output == label:
                            cnt_org_right_icl_oos += 1
                    elif model == "mistral":
                        if (is_mistral_positive(wo_icl_output) and label == "positive") or (is_mistral_negative(wo_icl_output) and label == "negative"):
                            cnt_org_right_icl_oos += 1
                    elif model == "llama2":
                        if (is_llama2_positive(wo_icl_output) and label == "positive") or (
                                is_llama2_negative(wo_icl_output) and label == "negative"):
                            cnt_org_right_icl_oos += 1


    print("cnt_org_oos_icl_right:", cnt_org_oos_icl_right)
    print("cnt_org_right_icl_oos: -", cnt_org_right_icl_oos)


    # in space out of format
    # print (len(wo_icl_in_space_out_format))
    # print (len(w_icl_in_space_out_format))
    # print (len(wo_icl_in_space_out_format - w_icl_in_space_out_format))
    # print (len(w_icl_in_space_out_format - wo_icl_in_space_out_format))
    # print (len(wo_icl_in_space_out_format & w_icl_in_space_out_format))
    new_ISIF_ISOOF = wo_icl_in_space_out_format - w_icl_in_space_out_format
    print("new ISIF from ISOOF:", len(new_ISIF_ISOOF))
    new_ISOOF_ISIF = w_icl_in_space_out_format - wo_icl_in_space_out_format
    print("new ISOOF from ISIF: -", len(new_ISOOF_ISIF))

    with jsonlines.open(with_out_ICL_path) as f_wo_icl:
        with jsonlines.open(with_ICL_path) as f_w_icl:
            cnt = 0
            cnt_org_isoof_icl_right = 0
            cnt_org_right_icl_isoof = 0
            for wo_icl_line, w_icl_line in zip(f_wo_icl, f_w_icl):
                cnt += 1
                wo_icl_output = wo_icl_line['output_text'].strip().lower()
                w_icl_output = w_icl_line['output_text'].strip().lower()
                if model == "mistral":
                    if "clean" in with_ICL_path:
                        assert wo_icl_line["input_query"] == w_icl_line["input_query"]
                    w_icl_output = w_icl_output.split('\n')[0]
                    w_icl_output = w_icl_output.split('(')[0].strip()
                    assert w_icl_output != ''
                    # print (w_icl_output)

                assert wo_icl_line["gold_label"].strip().lower() == w_icl_line["gold_label"].strip().lower()
                label = wo_icl_line["gold_label"].strip().lower()

                if cnt in new_ISIF_ISOOF:
                    # print (wo_icl_output, w_icl_output, label)
                    if model in ["chatgpt", "gpt3"]:
                        if w_icl_output == label:
                            cnt_org_isoof_icl_right += 1
                    elif model == "mistral":
                        if (is_mistral_positive(w_icl_output) and label == "positive") or (is_mistral_negative(w_icl_output) and label == "negative"):
                            cnt_org_isoof_icl_right += 1
                    elif model == "llama2":
                        if (is_llama2_positive(w_icl_output) and label == "positive") or (
                                is_llama2_negative(w_icl_output) and label == "negative"):
                            cnt_org_isoof_icl_right += 1

                if cnt in new_ISOOF_ISIF:
                    if model in ["chatgpt", "gpt3"]:
                        if wo_icl_output == label:
                            cnt_org_right_icl_isoof += 1
                    elif model == "mistral":
                        if (is_mistral_positive(wo_icl_output) and label == "positive") or (is_mistral_negative(wo_icl_output) and label == "negative"):
                            cnt_org_right_icl_isoof += 1
                    elif model == "llama2":
                        if (is_llama2_positive(wo_icl_output) and label == "positive") or (
                                is_llama2_negative(wo_icl_output) and label == "negative"):
                            cnt_org_right_icl_isoof += 1

    print("cnt_org_isoof_icl_right:", cnt_org_isoof_icl_right)
    print("cnt_org_right_icl_isoof: -", cnt_org_right_icl_isoof)


    # In space in format
    # print (len(wo_icl_in_space_in_format))
    # print (len(w_icl_in_space_in_format))
    # print (len(wo_icl_in_space_in_format & w_icl_in_space_in_format))
    # print (len(w_icl_in_space_in_format - wo_icl_in_space_in_format))
    # print (len(wo_icl_in_space_in_format - w_icl_in_space_in_format))

    # ISIF intersect part
    intersect_ISIF = wo_icl_in_space_in_format & w_icl_in_space_in_format
    for i in (wo_icl_in_space_in_format - w_icl_in_space_in_format):
        assert (i in w_icl_out_space or i in w_icl_in_space_out_format)

    with jsonlines.open(with_out_ICL_path) as f_wo_icl:
        with jsonlines.open(with_ICL_path) as f_w_icl:
            cnt = 0
            cnt_org_pred_right = 0
            cnt_icl_pred_right = 0
            cnt_org_right_icl_right = 0
            cnt_org_wrong_icl_wrong = 0
            cnt_org_wrong_icl_right = 0
            cnt_org_right_icl_wrong = 0
            for wo_icl_line, w_icl_line in zip(f_wo_icl, f_w_icl):
                cnt += 1
                wo_icl_output = wo_icl_line['output_text'].strip().lower()
                w_icl_output = w_icl_line['output_text'].strip().lower()
                if model == "mistral":
                    if "clean" in with_ICL_path:
                        assert wo_icl_line["input_query"] == w_icl_line["input_query"]
                    w_icl_output = w_icl_output.split('\n')[0]
                    w_icl_output = w_icl_output.split('(')[0].strip()
                    assert w_icl_output != ''
                    # print (w_icl_output)

                assert wo_icl_line["gold_label"].strip().lower() == w_icl_line["gold_label"].strip().lower()
                label = wo_icl_line["gold_label"].strip().lower()

                if cnt in intersect_ISIF:
                    if model in ["chatgpt", "gpt3"]:
                        if wo_icl_output == label:
                            cnt_org_pred_right += 1
                            if w_icl_output == label:
                                cnt_icl_pred_right += 1
                                cnt_org_right_icl_right += 1
                            else:
                                cnt_org_right_icl_wrong += 1
                        else:
                            if w_icl_output == label:
                                cnt_icl_pred_right += 1
                                cnt_org_wrong_icl_right += 1
                            else:
                                cnt_org_wrong_icl_wrong += 1
                    elif model == "mistral":
                        if (is_mistral_positive(wo_icl_output) and label == "positive") or (is_mistral_negative(wo_icl_output) and label == "negative"):
                            cnt_org_pred_right += 1
                            if (is_mistral_positive(w_icl_output) and label == "positive") or (is_mistral_negative(w_icl_output) and label == "negative"):
                                cnt_icl_pred_right += 1
                                cnt_org_right_icl_right += 1
                            else:
                                cnt_org_right_icl_wrong += 1
                        else:
                            if (is_mistral_positive(w_icl_output) and label == "positive") or (is_mistral_negative(w_icl_output) and label == "negative"):
                                cnt_icl_pred_right += 1
                                cnt_org_wrong_icl_right += 1
                            else:
                                cnt_org_wrong_icl_wrong += 1
                    elif model == "llama2":
                        if (is_llama2_positive(wo_icl_output) and label == "positive") or (
                                is_llama2_negative(wo_icl_output) and label == "negative"):
                            cnt_org_pred_right += 1
                            if (is_llama2_positive(w_icl_output) and label == "positive") or (
                                    is_llama2_negative(w_icl_output) and label == "negative"):
                                cnt_icl_pred_right += 1
                                cnt_org_right_icl_right += 1
                            else:
                                cnt_org_right_icl_wrong += 1
                        else:
                            if (is_llama2_positive(w_icl_output) and label == "positive") or (
                                    is_llama2_negative(w_icl_output) and label == "negative"):
                                cnt_icl_pred_right += 1
                                cnt_org_wrong_icl_right += 1
                            else:
                                cnt_org_wrong_icl_wrong += 1

    print (cnt, len(intersect_ISIF), cnt_org_pred_right, cnt_icl_pred_right)
    print ("cnt_org_right_icl_right:", cnt_org_right_icl_right)
    print ("cnt_org_wrong_icl_wrong:", cnt_org_wrong_icl_wrong)
    print ("cnt_org_wrong_icl_right:", cnt_org_wrong_icl_right)
    print ("cnt_org_right_icl_wrong:", cnt_org_right_icl_wrong)


def eval(file_name, model, exp = "acc2"):
    cnt_total_instance = 0
    cnt_is = 0
    cnt_isif = 0
    cnt_isoof = 0
    cnt_is_right_pred = 0
    cnt_isif_right_pred = 0
    with jsonlines.open(file_name) as f:
        for line in f.iter():
            cnt_total_instance += 1
            output = line['output_text'].strip().lower()
            label = line["gold_label"].strip().lower()
            if model in ["chatgpt", "gpt3"]:
                if ("positive" in output or "negative" in output) and "neutral" not in output:
                    # in space
                    cnt_is += 1
                    if ("positive" in output and label == "positive") or ("negative" in output and label == "negative"):
                        cnt_is_right_pred += 1
                    # in space and in format
                    if output in ['positive', 'negative']:
                        cnt_isif += 1
                        if output == label:
                            cnt_isif_right_pred += 1
                    else:
                        # in space but out of format
                        print (output)
                        cnt_isoof += 1

                else:
                    # print (output)
                    continue


            elif model == "mistral":
                if exp in ["acc2", "acc3","flip", "retrieval","num_k", "reverse_rand", "reverse_retrieval"]:
                    output = output.split('\n')[0]
                    output = output.split('(')[0].strip()
                    # print (output)
                if ("positive" in output or "negative" in output) and "neutral" not in output and "mix" not in output:
                    cnt_is += 1

                    if is_mistral_positive(output) or is_mistral_negative(output):
                        cnt_isif += 1
                        if (is_mistral_positive(output) and label == "positive") or (is_mistral_negative(output) and label == "negative"):
                            cnt_isif_right_pred += 1
                            cnt_is_right_pred += 1
                    else:
                        cnt_isoof += 1
                        for token in output.split():
                            if token in ["negative", "positive"]:
                                if token == label:
                                    cnt_is_right_pred += 1
                                    break
                        # print ("=============")
                        # print (output)

                else:
                    # print ("==========")
                    # print (output)
                    continue

            elif model == "llama2":
                if ("positive" in output or "negative" in output) and "neutral" not in output and "mix" not in output:
                    cnt_is += 1
                    if is_llama2_positive(output) or is_llama2_negative(output):
                        cnt_isif += 1
                        if (is_llama2_positive(output) and label == "positive") or (is_llama2_negative(output) and label == "negative"):
                            cnt_isif_right_pred += 1
                            cnt_is_right_pred += 1
                    else:
                        cnt_isoof += 1
                        for token in output.split():
                            if token in ["negative", "positive"]:
                                if token == label:
                                    cnt_is_right_pred += 1
                                    break
                        # print ("=============")
                        # print (output)
                else:
                    # print ("==========")
                    # print (output)
                    continue


            elif model == "flant5":
                if pp_level == "naive":
                    # if 'positive' in output and "negative" in output:
                    #     print (output)
                    if output in ['positive', 'negative']:
                        cnt_in_format += 1
                    # else:
                    #     print("[", output, "] is not in [", pp_level, "] evaluation.")
                    if output == label:
                        cnt_in_format_right_pred += 1

                elif pp_level == "normal":
                    # if 'positive' in output and "negative" in output:
                    #     print (output)
                    if 'positive' in output or 'negative' in output:
                        cnt_in_format += 1
                    # else:
                    #     print ("[",output,"] is not in [", pp_level, "] evaluation.")
                    if label in output:
                        cnt_in_format_right_pred += 1

                elif pp_level == "human-eval":
                    if "positive" in output and "negative" not in output and "neutral" not in output:
                        cnt_in_format += 1
                        if label == "positive":
                            cnt_in_format_right_pred += 1
                    elif "negative" in output and "positive" not in output and "neutral" not in output:
                        cnt_in_format += 1
                        if label == "negative":
                            cnt_in_format_right_pred += 1
                    else:
                        continue

    print("Total instances:", cnt_total_instance)
    print("Number for ISIF (post-processing) instances:", cnt_isif)
    print("Number for correct predictions in ISIF:", cnt_isif_right_pred)
    print("Number for ISOOF (format issue) instances:", cnt_isoof)
    print("Number for IS (human-eval) instances:", cnt_is)
    print("Number for correct predictions in IS:", cnt_is_right_pred)

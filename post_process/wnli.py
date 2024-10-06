import jsonlines
import re

def is_entailment(sent):
    if sent == "entail" or sent == "entailment" or sent == "label: entailment":
        return True
    elif sent == "yes":
        return True
    else:
        return False

def is_non_entailment(sent):
    if "non-entail" in sent or "not entail" in sent or sent == "no":
        return True
    else:
        return False

def is_mistral_entailment(sent):
    if "\"entailment\"" in sent:
        return True
    elif sent == "entailment" or sent == "label: entailment" or sent.startswith("entailment.") or sent.startswith("label: entailment."):
        return True
    elif "premise entails the hypothesis" in sent or "hypothesis is entailed by the premise" in sent:
        return True
    elif re.findall(r"premise \"[\w,.' ]+\" entails[\w ]hypothesis", sent):
        return True
    elif re.findall(r"hypothesis \"[\w,.' ]+\" is entailed", sent):
        return True
    else:
        return False

def is_mistral_non_entailment(sent):
    if "\"non-entailment\"" in sent:
        return True
    elif sent == "non-entailment" or sent == "label: non-entailment" or sent.startswith("non-entailment.") or sent.startswith("label: non-entailment."):
        return True
    elif "premise does not entail the hypothesis" in sent or "hypothesis is not entailed by the premise" in sent:
        return True
    elif re.findall(r"premise \"[\w,.' ]+\" does not entail[\w ]hypothesis", sent):
        return True
    elif re.findall(r"hypothesis \"[\w,.' ]+\" is not entailed", sent):
        return True
    else:
        return False


def is_llama2_entailment(sent):
    if "\"entailment\"" in sent or "\"entailment.\"" in sent:
        return True
    elif sent == "entailment" or sent == "label: entailment" or sent.startswith("entailment.") or "label: entailment." in sent or "label: entailment\n" in sent or "\nlabel: entailment" in sent:
        return True
    elif "premise entails the hypothesis" in sent or "hypothesis is entailed by the premise" in sent:
        return True
    elif re.findall(r"premise \"[\w,.' ]+\" entails[\w ]hypothesis", sent):
        return True
    elif re.findall(r"hypothesis \"[\w,.' ]+\" is entailed", sent):
        return True
    else:
        return False

def is_llama2_non_entailment(sent):
    if "\"non-entailment\"" in sent or "\"non-entailment.\"" in sent:
        return True
    elif sent == "non-entailment" or sent == "label: non-entailment" or sent.startswith("non-entailment.") or "label: non-entailment." in sent or "label: non-entailment\n" in sent or "\nlabel: non-entailment" in sent:
        return True
    elif "premise does not entail the hypothesis" in sent or "hypothesis is not entailed by the premise" in sent:
        return True
    elif re.findall(r"premise \"[\w,.' ]+\" does not entail[\w]hypothesis", sent):
        return True
    elif re.findall(r"hypothesis \"[\w,.' ]+\" is not entailed", sent):
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
                    assert w_icl_output != ''
                    # print (w_icl_output)

                assert wo_icl_line["gold_label"].strip().lower() == w_icl_line["gold_label"].strip().lower()
                label = wo_icl_line["gold_label"].strip().lower()

                if model in ["chatgpt", "gpt3"]:
                    if ("entail" in wo_icl_output or "yes" in wo_icl_output or wo_icl_output == "no") and "neither" not in wo_icl_output:
                        wo_icl_in_space.add(cnt)
                        if is_entailment(wo_icl_output) or is_non_entailment(wo_icl_output):
                            wo_icl_in_space_in_format.add(cnt)
                            if (is_entailment(wo_icl_output) and label == "entailment") or (is_non_entailment(wo_icl_output) and label == "non-entailment"):
                                wo_icl_in_space_in_format_right_pred.add(cnt)
                        else:
                            wo_icl_in_space_out_format.add(cnt)
                    else:
                        wo_icl_out_space.add(cnt)

                    if ("entail" in w_icl_output or "yes" in w_icl_output or w_icl_output == "no") and "neither" not in w_icl_output:
                        w_icl_in_space.add(cnt)
                        if is_entailment(w_icl_output) or is_non_entailment(w_icl_output):
                            w_icl_in_space_in_format.add(cnt)
                            if (is_entailment(w_icl_output) and label == "entailment") or (is_non_entailment(w_icl_output) and label == "non-entailment"):
                                w_icl_in_space_in_format_right_pred.add(cnt)
                        else:
                            w_icl_in_space_out_format.add(cnt)
                    else:
                        w_icl_out_space.add(cnt)

                elif model == "mistral":
                    if 'entail' in wo_icl_output and "true" not in wo_icl_output and "flase" not in wo_icl_output and "neutral" not in wo_icl_output:
                        wo_icl_in_space.add(cnt)
                        if is_mistral_entailment(wo_icl_output) or is_mistral_non_entailment(wo_icl_output):
                            wo_icl_in_space_in_format.add(cnt)
                            if (is_mistral_entailment(wo_icl_output) and label == "entailment") or (is_mistral_non_entailment(wo_icl_output) and label == "non-entailment"):
                                wo_icl_in_space_in_format_right_pred.add(cnt)
                        else:
                            wo_icl_in_space_out_format.add(cnt)
                    else:
                        wo_icl_out_space.add(cnt)


                    if 'entail' in w_icl_output and "true" not in w_icl_output and "flase" not in w_icl_output and "neutral" not in w_icl_output:
                        w_icl_in_space.add(cnt)
                        if is_mistral_entailment(w_icl_output) or is_mistral_non_entailment(w_icl_output):
                            w_icl_in_space_in_format.add(cnt)
                            if (is_mistral_entailment(w_icl_output) and label == "entailment") or (is_mistral_non_entailment(w_icl_output) and label == "non-entailment"):
                                w_icl_in_space_in_format_right_pred.add(cnt)
                        else:
                            w_icl_in_space_out_format.add(cnt)
                    else:
                        w_icl_out_space.add(cnt)

                elif model == "llama2":
                    if 'entail' in wo_icl_output and "true" not in wo_icl_output and "flase" not in wo_icl_output and "neutral" not in wo_icl_output:
                        wo_icl_in_space.add(cnt)
                        if is_llama2_entailment(wo_icl_output) or is_llama2_non_entailment(wo_icl_output):
                            wo_icl_in_space_in_format.add(cnt)
                            if (is_llama2_entailment(wo_icl_output) and label == "entailment") or (is_llama2_non_entailment(wo_icl_output) and label == "non-entailment"):
                                wo_icl_in_space_in_format_right_pred.add(cnt)
                        else:
                            wo_icl_in_space_out_format.add(cnt)
                    else:
                        wo_icl_out_space.add(cnt)


                    if 'entail' in w_icl_output and "true" not in w_icl_output and "flase" not in w_icl_output and "neutral" not in w_icl_output:
                        w_icl_in_space.add(cnt)
                        if is_llama2_entailment(w_icl_output) or is_llama2_non_entailment(w_icl_output):
                            w_icl_in_space_in_format.add(cnt)
                            if (is_llama2_entailment(w_icl_output) and label == "entailment") or (is_llama2_non_entailment(w_icl_output) and label == "non-entailment"):
                                w_icl_in_space_in_format_right_pred.add(cnt)
                        else:
                            w_icl_in_space_out_format.add(cnt)
                    else:
                        w_icl_out_space.add(cnt)




    print("Total instance:", cnt)
    print("Without ICL, file path:", with_out_ICL_path)
    print("Without ICL, in space number:", len(wo_icl_in_space))
    print("Without ICL, out space number:", len(wo_icl_out_space))
    print("Without ICL, in space and in format number:", len(wo_icl_in_space_in_format))
    print("Without ICL, in space but out of format number:", len(wo_icl_in_space_out_format))

    print("With ICL, in space number:", len(w_icl_in_space))
    print("With ICL, out space number:", len(w_icl_out_space))
    print("With ICL, in space and in format number:", len(w_icl_in_space_in_format))
    print("With ICL, in space but out of format number:", len(w_icl_in_space_out_format))

    # compute label space improvement
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

                if cnt in new_ISIF_OOS:  # this part contributes to the performance
                    # print (wo_icl_output, w_icl_output, label)
                    if model in ["chatgpt", "gpt3"]:
                        if (is_entailment(w_icl_output) and label == "entailment") or (is_non_entailment(w_icl_output) and label == "non-entailment"):
                            cnt_org_oos_icl_right += 1
                    elif model == "mistral":
                        if (is_mistral_entailment(w_icl_output) and label == "entailment") or (is_mistral_non_entailment(w_icl_output) and label == "non-entailment"):
                            cnt_org_oos_icl_right += 1
                    elif model == "llama2":
                        if (is_llama2_entailment(w_icl_output) and label == "entailment") or (
                                is_llama2_non_entailment(w_icl_output) and label == "non-entailment"):
                            cnt_org_oos_icl_right += 1
                elif cnt in new_OOS_ISIF:  # this part decrease the performance
                    if model in ["chatgpt", "gpt3"]:
                        if (is_entailment(wo_icl_output) and label == "entailment") or (is_non_entailment(wo_icl_output) and label == "non-entailment"):
                            cnt_org_right_icl_oos += 1
                    elif model == "mistral":
                        if (is_mistral_entailment(wo_icl_output) and label == "entailment") or (is_mistral_non_entailment(wo_icl_output) and label == "non-entailment"):
                            cnt_org_right_icl_oos += 1
                    elif model == "llama2":
                        if (is_llama2_entailment(wo_icl_output) and label == "entailment") or (
                                is_llama2_non_entailment(wo_icl_output) and label == "non-entailment"):
                            cnt_org_right_icl_oos += 1

    print("cnt_org_oos_icl_right:", cnt_org_oos_icl_right)
    print("cnt_org_right_icl_oos: -", cnt_org_right_icl_oos)


    #compute format improvement
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
                        if (is_entailment(w_icl_output) and label == "entailment") or (is_non_entailment(w_icl_output) and label == "non-entailment"):
                            cnt_org_isoof_icl_right += 1
                    elif model == "mistral":
                        if (is_mistral_entailment(w_icl_output) and label == "entailment") or (is_mistral_non_entailment(w_icl_output) and label == "non-entailment"):
                            cnt_org_isoof_icl_right += 1
                    elif model == "llama2":
                        if (is_llama2_entailment(w_icl_output) and label == "entailment") or (
                                is_llama2_non_entailment(w_icl_output) and label == "non-entailment"):
                            cnt_org_isoof_icl_right += 1

                if cnt in new_ISOOF_ISIF:
                    if model in ["chatgpt", "gpt3"]:
                        if (is_entailment(wo_icl_output) and label == "entailment") or (is_non_entailment(wo_icl_output) and label == "non-entailment"):
                            cnt_org_right_icl_isoof += 1
                    elif model == "mistral":
                        if (is_mistral_entailment(wo_icl_output) and label == "entailment") or (is_mistral_non_entailment(wo_icl_output) and label == "non-entailment"):
                            cnt_org_right_icl_isoof += 1
                    elif model == "llama2":
                        if (is_llama2_entailment(wo_icl_output) and label == "entailment") or (
                                is_llama2_non_entailment(wo_icl_output) and label == "non-entailment"):
                            cnt_org_right_icl_isoof += 1

    print("cnt_org_isoof_icl_right:", cnt_org_isoof_icl_right)
    print("cnt_org_right_icl_isoof: -", cnt_org_right_icl_isoof)


    # compute context semantic improvement
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
                        if (is_entailment(wo_icl_output) and label == "entailment") or (is_non_entailment(wo_icl_output) and label == "non-entailment"):
                            cnt_org_pred_right += 1
                            if (is_entailment(w_icl_output) and label == "entailment") or (is_non_entailment(w_icl_output) and label == "non-entailment"):
                                cnt_icl_pred_right += 1
                                cnt_org_right_icl_right += 1
                            else:
                                cnt_org_right_icl_wrong += 1
                        else:
                            if (is_entailment(w_icl_output) and label == "entailment") or (is_non_entailment(w_icl_output) and label == "non-entailment"):
                                cnt_icl_pred_right += 1
                                cnt_org_wrong_icl_right += 1
                            else:
                                cnt_org_wrong_icl_wrong += 1
                    elif model == "mistral":
                        if (is_mistral_entailment(wo_icl_output) and label == "entailment") or (is_mistral_non_entailment(wo_icl_output) and label == "non-entailment"):
                            cnt_org_pred_right += 1
                            if (is_mistral_entailment(w_icl_output) and label == "entailment") or (is_mistral_non_entailment(w_icl_output) and label == "non-entailment"):
                                cnt_icl_pred_right += 1
                                cnt_org_right_icl_right += 1
                            else:
                                cnt_org_right_icl_wrong += 1
                        else:
                            if (is_mistral_entailment(w_icl_output) and label == "entailment") or (is_mistral_non_entailment(w_icl_output) and label == "non-entailment"):
                                cnt_icl_pred_right += 1
                                cnt_org_wrong_icl_right += 1
                            else:
                                cnt_org_wrong_icl_wrong += 1
                    elif model == "llama2":
                        if (is_llama2_entailment(wo_icl_output) and label == "entailment") or (is_llama2_non_entailment(wo_icl_output) and label == "non-entailment"):
                            cnt_org_pred_right += 1
                            if (is_llama2_entailment(w_icl_output) and label == "entailment") or (is_llama2_non_entailment(w_icl_output) and label == "non-entailment"):
                                cnt_icl_pred_right += 1
                                cnt_org_right_icl_right += 1
                            else:
                                cnt_org_right_icl_wrong += 1
                        else:
                            if (is_llama2_entailment(w_icl_output) and label == "entailment") or (is_llama2_non_entailment(w_icl_output) and label == "non-entailment"):
                                cnt_icl_pred_right += 1
                                cnt_org_wrong_icl_right += 1
                            else:
                                cnt_org_wrong_icl_wrong += 1

    print(cnt, len(intersect_ISIF), cnt_org_pred_right, cnt_icl_pred_right)
    print("cnt_org_right_icl_right:", cnt_org_right_icl_right)
    print("cnt_org_wrong_icl_wrong:", cnt_org_wrong_icl_wrong)
    print("cnt_org_wrong_icl_right:", cnt_org_wrong_icl_right)
    print("cnt_org_right_icl_wrong:", cnt_org_right_icl_wrong)


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
                if 'entail' in output or "yes" in output or output == "no":
                    cnt_is += 1
                    if is_entailment(output) or is_non_entailment(output):
                        cnt_isif += 1
                        if (is_entailment(output) and label == "entailment") or (is_non_entailment(output) and label == "non-entailment"):
                            cnt_is_right_pred += 1
                            cnt_isif_right_pred += 1
                    else:
                        cnt_isoof += 1
                        print(output)
                        if "entails" in output or "yes" in output or "entailed" in output:
                            if label == "entailment":
                                cnt_is_right_pred += 1
                        else:
                            print (output)
                else:
                    #out of space
                    # print (output)
                    continue

            elif model == "mistral":
                if exp in ["acc2", "acc3", "flip", "retrieval", "num_k", "reverse_rand", "reverse_retrieval"]:
                    output = output.split('\n')[0]
                    output = output.split('(')[0].strip()
                if 'entail' in output and "true" not in output and "flase" not in output and "neutral" not in output:
                    cnt_is += 1
                    # print (output)
                    if is_mistral_entailment(output) or is_mistral_non_entailment(output):
                        cnt_isif += 1
                        if (is_mistral_entailment(output) and label == "entailment") or (
                                is_mistral_non_entailment(output) and label == "non-entailment"):
                            cnt_is_right_pred += 1
                            cnt_isif_right_pred += 1
                    else:
                        cnt_isoof += 1
                        print("=========")
                        print(output, label)
                else:
                    # print("=========")
                    # print(output)
                    continue


            elif model == "llama2":
                if 'entail' in output and "true" not in output and "false" not in output and "neutral" not in output:
                    cnt_is += 1
                    # print (output)
                    if is_llama2_entailment(output) or is_llama2_non_entailment(output):
                        cnt_isif += 1
                        if (is_llama2_entailment(output) and label == "entailment") or (
                                is_llama2_non_entailment(output) and label == "non-entailment"):
                            cnt_is_right_pred += 1
                            cnt_isif_right_pred += 1
                    else:
                        cnt_isoof += 1
                        print("=========")
                        print(output, label)
                else:
                    # print("=========")
                    # print(output)
                    continue

    print("Total instances:", cnt_total_instance)
    print("Number for ISIF (post-processing) instances:", cnt_isif)
    print("Number for correct predictions in ISIF:", cnt_isif_right_pred)
    print("Number for ISOOF (format issue) instances:", cnt_isoof)
    print("Number for IS (human-eval) instances:", cnt_is)
    print("Number for correct predictions in IS:", cnt_is_right_pred)

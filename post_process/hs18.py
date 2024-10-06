import jsonlines
import re

def is_hate(sent):
    if sent == "hate" or sent == "hate speech":
        return True
    else:
        return False

def is_non_hate(sent):
    if "non-hate" in sent or "not hate" in sent or sent == "neutral":
        return True
    else:
        return False


def is_mistral_in_space(sent):
    if sent.startswith("neutral"):
        return True
    tmp = re.findall("label:[\w\-\(\)\[\], ]*", sent)
    if tmp:
        if "hate" in tmp[0]:
            return True
        else:
            # print (sent, tmp)
            return False
    else:
        if "hate" in sent:
            return True
        else:
            return False

def is_llama2_in_space(sent):
    tmp = re.findall("label:[\w\-\(\)\[\], ]*", sent)
    if tmp:
        if "hate" in tmp[0]:
            return True
        else:
            # print (sent, tmp)
            return False
    else:
        if "hate" in sent:
            return True
        else:
            return False

def is_mistral_hate(sent):
    tmp = re.findall("label:[\w\-\(\)\[\], ]*", sent)
    if tmp:
        pred = tmp[0][6:]
        if ("not" not in pred) and ("non" not in pred):
            # print ("************",sent, pred)
            return True
        else:
            # print("************", sent, pred)
            return False
    else:
        if "\"hate\"" in sent:
            return True
        elif sent.startswith("hate"):
            return True
        elif re.findall(r'contains[\w ]*hate', sent):
            return True
        else:
            return False

def is_mistral_non_hate(sent):
    if "neutral" in sent:
        return True
    tmp = re.findall("label:[\w\-\(\)\[\], ]*", sent)
    if tmp:
        pred = tmp[0][6:]
        if "not" in pred or "non" in pred:
            return True
        else:
            # print("************", sent, pred)
            return False
    else:
        if "non-hate" in sent:
            return True
        else:
            if re.findall(r'not[\w ]*hate', sent):
                # print("************", sent, re.findall(r'not[\w ]*hate', sent))
                return True
            elif re.findall(r'there is no[\w ]*hate', sent):
                # print("************", sent, re.findall(r'not[\w ]*hate', sent))
                return True
            else:
                return False


def is_llama2_hate(sent):
    tmp = re.findall("label:[\w\-\(\)\[\], ]*", sent)
    if tmp:
        pred = tmp[0][6:]
        if ("not" not in pred) and ("non" not in pred):
            # print ("************",sent, pred)
            return True
        else:
            # print("************", sent, pred)
            return False
    else:
        if "\"hate\"" in sent:
            return True
        elif sent.startswith("hate"):
            return True
        elif re.findall(r'contains[\w ]*hate', sent):
            return True
        else:
            return False

def is_llama2_non_hate(sent):
    tmp = re.findall("label:[\w\-\(\)\[\], ]*", sent)
    if tmp:
        pred = tmp[0][6:]
        if "not" in pred or "non" in pred:
            return True
        else:
            # print("************", sent, pred)
            return False
    else:
        if "\"non-hate\"" in sent:
            return True
        elif sent.startswith("non-hate"):
            return True
        else:
            if re.findall(r'not[\w ]*hate', sent):
                # print("************", sent, re.findall(r'not[\w ]*hate', sent))
                return True
            elif re.findall(r'there is no[\w ]*hate', sent):
                # print("************", sent, re.findall(r'not[\w ]*hate', sent))
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
                    if "hate" in wo_icl_output or "neutral" in wo_icl_output or "non" in wo_icl_output:
                        wo_icl_in_space.add(cnt)
                        if is_hate(wo_icl_output) or is_non_hate(wo_icl_output):
                            wo_icl_in_space_in_format.add(cnt)
                            if (is_hate(wo_icl_output) and label == "hate") or (is_non_hate(wo_icl_output) and label == "non-hate"):
                                wo_icl_in_space_in_format_right_pred.add(cnt)
                        else:
                            wo_icl_in_space_out_format.add(cnt)
                    else:
                        wo_icl_out_space.add(cnt)

                    if "hate" in w_icl_output or "neutral" in w_icl_output or "non" in w_icl_output:
                        w_icl_in_space.add(cnt)
                        if is_hate(w_icl_output) or is_non_hate(w_icl_output):
                            w_icl_in_space_in_format.add(cnt)
                            if (is_hate(w_icl_output) and label == "hate") or (is_non_hate(w_icl_output) and label == "non-hate"):
                                w_icl_in_space_in_format_right_pred.add(cnt)
                        else:
                            w_icl_in_space_out_format.add(cnt)
                    else:
                        w_icl_out_space.add(cnt)

                elif model == "mistral":
                    if is_mistral_in_space(wo_icl_output):
                        wo_icl_in_space.add(cnt)
                        if is_mistral_hate(wo_icl_output) or is_mistral_non_hate(wo_icl_output):
                            wo_icl_in_space_in_format.add(cnt)
                            if (is_mistral_hate(wo_icl_output) and label == "hate") or (is_mistral_non_hate(wo_icl_output) and label == "non-hate"):
                                wo_icl_in_space_in_format_right_pred.add(cnt)
                        else:
                            wo_icl_in_space_out_format.add(cnt)
                    else:
                        wo_icl_out_space.add(cnt)


                    if is_mistral_in_space(w_icl_output):
                        w_icl_in_space.add(cnt)
                        if is_mistral_hate(w_icl_output) or is_mistral_non_hate(w_icl_output):
                            w_icl_in_space_in_format.add(cnt)
                            if (is_mistral_hate(w_icl_output) and label == "hate") or (is_mistral_non_hate(w_icl_output) and label == "non-hate"):
                                w_icl_in_space_in_format_right_pred.add(cnt)
                        else:
                            w_icl_in_space_out_format.add(cnt)
                    else:
                        w_icl_out_space.add(cnt)

                elif model == "llama2":
                    if is_llama2_in_space(wo_icl_output):
                        wo_icl_in_space.add(cnt)
                        if is_llama2_hate(wo_icl_output) or is_llama2_non_hate(wo_icl_output):
                            wo_icl_in_space_in_format.add(cnt)
                            if (is_llama2_hate(wo_icl_output) and label == "hate") or (is_llama2_non_hate(wo_icl_output) and label == "non-hate"):
                                wo_icl_in_space_in_format_right_pred.add(cnt)
                        else:
                            wo_icl_in_space_out_format.add(cnt)
                    else:
                        wo_icl_out_space.add(cnt)


                    if is_llama2_in_space(w_icl_output):
                        w_icl_in_space.add(cnt)
                        if is_llama2_hate(w_icl_output) or is_llama2_non_hate(w_icl_output):
                            w_icl_in_space_in_format.add(cnt)
                            if (is_llama2_hate(w_icl_output) and label == "hate") or (is_llama2_non_hate(w_icl_output) and label == "non-hate"):
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
                        if (is_hate(w_icl_output) and label == "hate") or (
                                is_non_hate(w_icl_output) and label == "non-hate"):
                            cnt_org_oos_icl_right += 1
                    elif model == "mistral":
                        if (is_mistral_hate(w_icl_output) and label == "hate") or (
                                is_mistral_non_hate(w_icl_output) and label == "non-hate"):
                            cnt_org_oos_icl_right += 1
                    elif model == "llama2":
                        if (is_llama2_hate(w_icl_output) and label == "hate") or (
                                is_llama2_non_hate(w_icl_output) and label == "non-hate"):
                            cnt_org_oos_icl_right += 1

                elif cnt in new_OOS_ISIF:  # this part decrease the performance
                    if model in ["chatgpt", "gpt3"]:
                        if (is_hate(wo_icl_output) and label == "hate") or (
                                is_non_hate(wo_icl_output) and label == "non-hate"):
                            cnt_org_right_icl_oos += 1
                    elif model == "mistral":
                        if (is_mistral_hate(wo_icl_output) and label == "hate") or (is_mistral_non_hate(wo_icl_output) and label == "non-hate"):
                            cnt_org_right_icl_oos += 1
                    elif model == "llama2":
                        if (is_llama2_hate(wo_icl_output) and label == "hate") or (
                                is_llama2_non_hate(wo_icl_output) and label == "non-hate"):
                            cnt_org_right_icl_oos += 1

    print("cnt_org_oos_icl_right:", cnt_org_oos_icl_right)
    print("cnt_org_right_icl_oos: -", cnt_org_right_icl_oos)

    # compute format improvement
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
                        if (is_hate(w_icl_output) and label == "hate") or (
                                is_non_hate(w_icl_output) and label == "non-hate"):
                            cnt_org_isoof_icl_right += 1
                    elif model == "mistral":
                        if (is_mistral_hate(w_icl_output) and label == "hate") or (
                                is_mistral_non_hate(w_icl_output) and label == "non-hate"):
                            cnt_org_isoof_icl_right += 1
                    elif model == "llama2":
                        if (is_llama2_hate(w_icl_output) and label == "hate") or (
                                is_llama2_non_hate(w_icl_output) and label == "non-hate"):
                            cnt_org_isoof_icl_right += 1

                if cnt in new_ISOOF_ISIF:
                    if model in ["chatgpt", "gpt3"]:
                        if (is_hate(wo_icl_output) and label == "hate") or (
                                is_non_hate(wo_icl_output) and label == "non-hate"):
                            cnt_org_right_icl_isoof += 1
                    elif model == "mistral":
                        if (is_mistral_hate(wo_icl_output) and label == "hate") or (is_mistral_non_hate(wo_icl_output) and label == "non-hate"):
                            cnt_org_right_icl_isoof += 1
                    elif model == "llama2":
                        if (is_llama2_hate(wo_icl_output) and label == "hate") or (
                                is_llama2_non_hate(wo_icl_output) and label == "non-hate"):
                            cnt_org_right_icl_isoof += 1


    print("cnt_org_isoof_icl_right:", cnt_org_isoof_icl_right)
    print("cnt_org_right_icl_isoof: -", cnt_org_right_icl_isoof)

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
                        if (is_hate(wo_icl_output) and label == "hate") or (
                                is_non_hate(wo_icl_output) and label == "non-hate"):
                            cnt_org_pred_right += 1
                            if (is_hate(w_icl_output) and label == "hate") or (
                                    is_non_hate(w_icl_output) and label == "non-hate"):
                                cnt_icl_pred_right += 1
                                cnt_org_right_icl_right += 1
                            else:
                                cnt_org_right_icl_wrong += 1
                        else:
                            if (is_hate(w_icl_output) and label == "hate") or (
                                    is_non_hate(w_icl_output) and label == "non-hate"):
                                cnt_icl_pred_right += 1
                                cnt_org_wrong_icl_right += 1
                            else:
                                cnt_org_wrong_icl_wrong += 1
                    elif model == "mistral":
                        if (is_mistral_hate(wo_icl_output) and label == "hate") or (
                                is_mistral_non_hate(wo_icl_output) and label == "non-hate"):
                            cnt_org_pred_right += 1
                            if (is_mistral_hate(w_icl_output) and label == "hate") or (
                                    is_mistral_non_hate(w_icl_output) and label == "non-hate"):
                                cnt_icl_pred_right += 1
                                cnt_org_right_icl_right += 1
                            else:
                                cnt_org_right_icl_wrong += 1
                        else:
                            if (is_mistral_hate(w_icl_output) and label == "hate") or (
                                    is_mistral_non_hate(w_icl_output) and label == "non-hate"):
                                cnt_icl_pred_right += 1
                                cnt_org_wrong_icl_right += 1
                            else:
                                cnt_org_wrong_icl_wrong += 1
                    elif model == "llama2":
                        if (is_llama2_hate(wo_icl_output) and label == "hate") or (
                                is_llama2_non_hate(wo_icl_output) and label == "non-hate"):
                            cnt_org_pred_right += 1
                            if (is_llama2_hate(w_icl_output) and label == "hate") or (
                                    is_llama2_non_hate(w_icl_output) and label == "non-hate"):
                                cnt_icl_pred_right += 1
                                cnt_org_right_icl_right += 1
                            else:
                                cnt_org_right_icl_wrong += 1
                        else:
                            if (is_llama2_hate(w_icl_output) and label == "hate") or (
                                    is_llama2_non_hate(w_icl_output) and label == "non-hate"):
                                cnt_icl_pred_right += 1
                                cnt_org_wrong_icl_right += 1
                            else:
                                cnt_org_wrong_icl_wrong += 1

    print(cnt, len(intersect_ISIF), cnt_org_pred_right, cnt_icl_pred_right)
    print("cnt_org_right_icl_right:", cnt_org_right_icl_right)
    print("cnt_org_wrong_icl_wrong:", cnt_org_wrong_icl_wrong)
    print("cnt_org_wrong_icl_right:", cnt_org_wrong_icl_right)
    print("cnt_org_right_icl_wrong:", cnt_org_right_icl_wrong)


def eval(file_name, model, exp = 'acc2'):
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
            if model in ["chatgpt", 'gpt3']:
                if "hate" in output or "neutral" in output or "non" in output:
                    cnt_is += 1
                    if is_hate(output) or is_non_hate(output):
                        cnt_isif += 1
                        if (is_hate(output) and label == "hate") or (is_non_hate(output) and label == "non-hate"):
                            cnt_is_right_pred += 1
                            cnt_isif_right_pred += 1
                    else:
                        cnt_isoof += 1
                        if "not" in output or "non" in output or "no hate" in output or "hate speech: no" in output:
                            if label == "non-hate":
                                cnt_is_right_pred += 1
                        elif "hate" in output or "offensive" in output:
                            if label == "hate":
                                cnt_is_right_pred += 1
                        else:
                            print ("========")
                            print (output)
                            continue
                else:
                    # print("========")
                    # print(output)
                    continue

            elif model == "mistral":
                if exp in ["acc2", "acc3", "flip", "retrieval", "num_k", "reverse_rand", "reverse_retrieval"]:
                    output = output.split('\n')[0]
                    output = output.split('(')[0].strip()
                if is_mistral_in_space(output):
                    cnt_is += 1
                    if is_mistral_hate(output) or is_mistral_non_hate(output):
                        cnt_isif += 1
                        if (is_mistral_non_hate(output) and label == 'non-hate') or (is_mistral_hate(output) and label == "hate"):
                            cnt_is_right_pred += 1
                            cnt_isif_right_pred += 1
                    else:
                        cnt_isoof += 1
                        print("=========")
                        print(output, label)
                        continue
                else:
                    # print ("=========")
                    # print (output, label)
                    continue

            elif model == "llama2":
                if is_llama2_in_space(output):
                    cnt_is += 1
                    if is_llama2_hate(output) or is_llama2_non_hate(output):
                        cnt_isif += 1
                        if (is_llama2_non_hate(output) and label == 'non-hate') or (is_llama2_hate(output) and label == "hate"):
                            cnt_is_right_pred += 1
                            cnt_isif_right_pred += 1
                    else:
                        cnt_isoof += 1
                        print("=========")
                        print(output, label)
                        continue
                else:
                    # print ("=========")
                    # print (output)
                    continue


    print("Total instances:", cnt_total_instance)
    print("Number for ISIF (post-processing) instances:", cnt_isif)
    print("Number for correct predictions in ISIF:", cnt_isif_right_pred)
    print("Number for ISOOF (format issue) instances:", cnt_isoof)
    print("Number for IS (human-eval) instances:", cnt_is)
    print("Number for correct predictions in IS:", cnt_is_right_pred)
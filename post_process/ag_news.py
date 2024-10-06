import jsonlines
import re

def is_sports(sent):
    if "sport" in sent:
        return True
    else:
        return False

def is_business(sent):
    if "business" in sent or "trade" in sent or "economy" in sent or "finance" in sent:
        return True
    else:
        return False

def is_sci_tech(sent):
    if "science" in sent or "technology" in sent:
        return True
    else:
        return False

def is_world(sent):
    if "world" in sent or "international" in sent or "terrorism" in sent or "conflict" in sent or "crime" in sent or "military" in sent or "politics" in sent:
        return True
    else:
        return False

def in_space(output):
    if "sport" in output or "business" in output or "tech" in output or "scien" in output or "world" in output or "econom" in output or "finan" in output or "politi" in output or "terror" in output or "trade" in output or "international" in output or "conflict" in output or "crime" in output or "milit" in output:
        return True
    else:
        return False

def pred_right(output, label):
    if (is_sports(output) and label == "sports") or (is_business(output) and label == "business") or (is_sci_tech(output) and label == "science/technology") or (is_world(output) and label == "world"):
        return True
    else:
        return False

def extract_llama2(sent):
    tmp1 = re.findall("\nlabel:[\w/&\-\(\) ]*", sent)
    tmp2 = re.findall("is:\n[\w/&\-\(\) ]*", sent)
    tmp3 = re.findall("belongs to the \"[\w/&.\-\(\) ]*\"", sent)
    tmp4 = re.findall("would be \"[\w/&.\-\(\) ]*\"", sent)
    tmp5 = re.findall("as \"[\w/&.\-\(\) ]*\"", sent)
    tmp6 = re.findall("as a \"[\w/&.\-\(\) ]*\"", sent)
    tmp7 = re.findall("belonging to the \"[\w/&.\-\(\) ]*\"", sent)
    tmp8 = re.findall("category of \"[\w/&.\-\(\) ]*\"", sent)
    tmp9 = re.findall("news type \"[\w/&.\-\(\) ]*\"", sent)
    tmp10 = re.findall("\"[\w/&\-\(\) ]*\" news type", sent)
    tmp11 = re.findall("is \"[\w/&.\-\(\) ]*\"", sent)
    tmp12 = re.findall("label \"[\w/&.\-\(\) ]*\"", sent)
    tmp13 = re.findall("news type of \"[\w/&.\-\(\) ]*\"", sent)
    tmp14 = re.findall("the news type[\w/&:\-\(\) ]*", sent)
    tmp15 = re.findall("category of[\w/&:\-\(\) ]*", sent)
    if tmp1:
        for i in tmp1:
            if in_space(i):
                return i
        return tmp1[0]
    elif tmp2:
        for i in tmp2:
            if in_space(i):
                return i
        return tmp2[0]
    elif tmp3:
        for i in tmp3:
            if in_space(i):
                return i
        return tmp3[0]
    elif tmp4:
        for i in tmp4:
            if in_space(i):
                return i
        return tmp4[0]
    elif tmp5:
        for i in tmp5:
            if in_space(i):
                return i
        return tmp5[0]
    elif tmp6:
        for i in tmp6:
            if in_space(i):
                return i
        return tmp6[0]
    elif tmp7:
        for i in tmp7:
            if in_space(i):
                return i
        return tmp7[0]
    elif tmp8:
        for i in tmp8:
            if in_space(i):
                return i
        return tmp8[0]
    elif tmp9:
        for i in tmp8:
            if in_space(i):
                return i
        return tmp9[0]
    elif tmp10:
        for i in tmp10:
            if in_space(i):
                return i
        return tmp10[0]
    elif tmp11:
        for i in tmp11:
            if in_space(i):
                return i
        return tmp11[0]
    elif tmp12:
        for i in tmp12:
            if in_space(i):
                return i
        return tmp12[0]
    elif tmp13:
        for i in tmp13:
            if in_space(i):
                return i
        return tmp13[0]
    elif tmp14:
        for i in tmp14:
            if in_space(i):
                return i
        return tmp14[0]
    elif tmp15:
        for i in tmp15:
            if in_space(i):
                return i
        return tmp15[0]
    else:
        # print("=========")
        # print(sent)
        return sent.split('\n')[0].split('.')[0]


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
                    wo_icl_output = extract_llama2(wo_icl_output)
                    w_icl_output = extract_llama2(w_icl_output)
                elif model == "llama2":
                    wo_icl_output = extract_llama2(wo_icl_output)
                    w_icl_output = extract_llama2(w_icl_output)

                assert wo_icl_line["gold_label"].strip().lower() == w_icl_line["gold_label"].strip().lower()
                label = wo_icl_line["gold_label"].strip().lower()

                if model in ["chatgpt", "gpt3", "mistral", "llama2"]:
                    if in_space(wo_icl_output):
                        wo_icl_in_space.add(cnt)
                        if is_sports(wo_icl_output) or is_business(wo_icl_output) or is_sci_tech(wo_icl_output) or is_world(wo_icl_output):
                            wo_icl_in_space_in_format.add(cnt)
                            if pred_right(wo_icl_output, label):
                                wo_icl_in_space_in_format_right_pred.add(cnt)
                        else:
                            wo_icl_in_space_out_format.add(cnt)
                    else:
                        wo_icl_out_space.add(cnt)

                    if in_space(w_icl_output):
                        w_icl_in_space.add(cnt)
                        if is_sports(w_icl_output) or is_business(w_icl_output) or is_sci_tech(w_icl_output) or is_world(w_icl_output):
                            w_icl_in_space_in_format.add(cnt)
                            if pred_right(w_icl_output, label):
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
                    wo_icl_output = extract_llama2(wo_icl_output)
                    w_icl_output = extract_llama2(w_icl_output)
                elif model == "llama2":
                    wo_icl_output = extract_llama2(wo_icl_output)
                    w_icl_output = extract_llama2(w_icl_output)

                assert wo_icl_line["gold_label"].strip().lower() == w_icl_line["gold_label"].strip().lower()
                label = wo_icl_line["gold_label"].strip().lower()

                if cnt in new_ISIF_OOS:  # this part contributes to the performance
                    # print (wo_icl_output, w_icl_output, label)
                    if model in ["chatgpt", "gpt3", "mistral", "llama2"]:
                        if pred_right(w_icl_output, label):
                            cnt_org_oos_icl_right += 1

                elif cnt in new_OOS_ISIF:  # this part decrease the performance
                    if model in ["chatgpt", "gpt3", "mistral", "llama2"]:
                        if pred_right(wo_icl_output, label):
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
                    wo_icl_output = extract_llama2(wo_icl_output)
                    w_icl_output = extract_llama2(w_icl_output)
                elif model == "llama2":
                    wo_icl_output = extract_llama2(wo_icl_output)
                    w_icl_output = extract_llama2(w_icl_output)

                assert wo_icl_line["gold_label"].strip().lower() == w_icl_line["gold_label"].strip().lower()
                label = wo_icl_line["gold_label"].strip().lower()

                if cnt in new_ISIF_ISOOF:
                    # print (wo_icl_output, w_icl_output, label)
                    if model in ["chatgpt", "gpt3", "mistral", "llama2"]:
                        if pred_right(w_icl_output, label):
                            cnt_org_isoof_icl_right += 1

                if cnt in new_ISOOF_ISIF:
                    if model in ["chatgpt", "gpt3", "mistral", "llama2"]:
                        if pred_right(wo_icl_output, label):
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
                    wo_icl_output = extract_llama2(wo_icl_output)
                    w_icl_output = extract_llama2(w_icl_output)
                elif model == "llama2":
                    wo_icl_output = extract_llama2(wo_icl_output)
                    w_icl_output = extract_llama2(w_icl_output)

                assert wo_icl_line["gold_label"].strip().lower() == w_icl_line["gold_label"].strip().lower()
                label = wo_icl_line["gold_label"].strip().lower()

                if cnt in intersect_ISIF:
                    if model in ["chatgpt", "gpt3", "mistral", "llama2"]:
                        if pred_right(wo_icl_output, label):
                            cnt_org_pred_right += 1
                            if pred_right(w_icl_output, label):
                                cnt_icl_pred_right += 1
                                cnt_org_right_icl_right += 1
                            else:
                                cnt_org_right_icl_wrong += 1
                        else:
                            if pred_right(w_icl_output, label):
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
            if model in ["chatgpt", "gpt3", "mistral", "llama2"]:
                if model == "mistral":
                    if exp in ["acc2", "flip", "retrieval", "diverse", "acc3", "num_k", "reverse_rand", "reverse_retrieval"]:
                        output = output.split('\n')[0]
                        output = output.split('(')[0].strip()

                    output = extract_llama2(output)
                elif model == "llama2":
                    output = extract_llama2(output)


                if in_space(output):
                    cnt_is += 1
                    if is_sports(output) or is_business(output) or is_sci_tech(output) or is_world(output):
                        cnt_isif += 1
                        if pred_right(output, label):
                            cnt_is_right_pred += 1
                            cnt_isif_right_pred += 1
                    else:
                        cnt_isoof += 1
                        if "econom" in output or "tech" in output or "financ" in output or "politi" in output or "terror" in output or "scien" in output or "milit" in output:
                            if ("econom" in output and label == "business") or ("tech" in output and label == "science/technology") or ("scien" in output and label == "science/technology") or ("financ" in output and label == "business") or ("politi" in output and label == "world") or ("terror" in output and label == "world") or ("milit" in output and label == "world"):
                                cnt_is_right_pred += 1

                        else:
                            print ("=========")
                            print (output, '\t',label)
                            continue
                else:
                    # print ("=========")
                    # print (output, '\t',label)
                    continue



    print("Total instances:", cnt_total_instance)
    print("Number for ISIF (post-processing) instances:", cnt_isif)
    print("Number for correct predictions in ISIF:", cnt_isif_right_pred)
    print("Number for ISOOF (format issue) instances:", cnt_isoof)
    print("Number for IS (human-eval) instances:", cnt_is)
    print("Number for correct predictions in IS:", cnt_is_right_pred)
import jsonlines
import re

def is_abbreviation(sent):
    if "abbreviation" in sent:
        return True
    else:
        return False

def is_entity(sent):
    if "entity" in sent or "organization" in sent or "color" in sent:
        return True
    else:
        return False

def is_description(sent):
    if "description" in sent or "concept" in sent or "definition" in sent:
        return True
    else:
        return False

def is_human(sent):
    if "human" in sent or "person" in sent or "name" in sent:
        return True
    else:
        return False

def is_location(sent):
    if "locatio" in sent or "place" in sent or "geography" in sent:
        return  True
    else:
        return False

def is_numeric(sent):
    if "numerical"in sent or "quantity" in sent or "date" in sent or "time" in sent or "distance" in sent or "statistics" in sent:
        return True
    else:
        return False

def in_space(output):
    if "abbrevia" in output or "descrip" in output or "abstra" in output or "concept" in output or "defini" in output or "locati" in output or "place" in output or "geogra" in output or "human" in output or "person" in output or "name" in output or "numeri"in output or "quanti" in output or "date" in output or "time" in output or "tempor" in output or "distan" in output or "measure" in output or "dimens" in output or "statis" in output or "organi" in output or "entit" in output or "color" in output:
        return True
    else:
        return False

def pred_right(output, label):
    if label == "locatio":
        label = "location"
    if (is_human(output) and label == "human being") or (is_entity(output) and label == "entity") or (is_numeric(output) and label == "numerical value") or (is_location(output) and label == "location") or (is_description(output) and label == "description and abstract concept") or (is_abbreviation(output) and label == "abbreviation"):
        return True
    else:
        return False

def extract_llama2(sent):
    tmp1 = re.findall("\nlabel:[\w/&\-\(\) ]*", sent)
    tmp2 = re.findall("is:\n[\w/&\-\(\) ]*", sent)
    tmp3 = re.findall("question type:[\w/&\-\(\) ]*", sent)
    tmp4 = re.findall("would be \"[\w/&\-\(\) ]*\"", sent)
    tmp5 = re.findall("or \"[\w/&\-\(\) ]*\"", sent)
    tmp6 = re.findall("is asking[\w/&\-\(\) ]*", sent)
    tmp7 = re.findall("as a \"[\w/&.\-\(\) ]*\"", sent)
    tmp8 = re.findall("category of \"[\w/&.\-\(\) ]*\"", sent)
    tmp9 = re.findall("label is \"[\w/&.\-\(\) ]*\"", sent)
    tmp10 = re.findall("\[[\w/&.:\-\(\) ]*\]", sent)
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
        for i in tmp9:
            if in_space(i):
                return i
        return tmp9[0]
    elif tmp10:
        for i in tmp10:
            if in_space(i):
                return i
        return tmp10[0]
    else:
        # print("**************")
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

                label = wo_icl_line["gold_label"].lower()
                tmp = w_icl_line["gold_label"].lower()
                if label == "locatio":
                    label = "location"
                if tmp == "locatio":
                    tmp = "location"
                assert label == tmp

                if model in ["chatgpt", "gpt3", "mistral", "llama2"]:
                    if in_space(wo_icl_output):
                        wo_icl_in_space.add(cnt)
                        if is_abbreviation(wo_icl_output) or is_entity(wo_icl_output) or is_description(wo_icl_output) or is_human(wo_icl_output) or is_location(wo_icl_output) or is_numeric(wo_icl_output):
                            wo_icl_in_space_in_format.add(cnt)
                            if pred_right(wo_icl_output, label):
                                wo_icl_in_space_in_format_right_pred.add(cnt)
                        else:
                            wo_icl_in_space_out_format.add(cnt)
                    else:
                        wo_icl_out_space.add(cnt)

                    if in_space(w_icl_output):
                        w_icl_in_space.add(cnt)
                        if is_abbreviation(w_icl_output) or is_entity(w_icl_output) or is_description(w_icl_output) or is_human(w_icl_output) or is_location(w_icl_output) or is_numeric(w_icl_output):
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

                label = wo_icl_line["gold_label"].lower()
                tmp = w_icl_line["gold_label"].lower()
                if label == "locatio":
                    label = "location"
                if tmp == "locatio":
                    tmp = "location"
                assert label == tmp

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

                label = wo_icl_line["gold_label"].lower()
                tmp = w_icl_line["gold_label"].lower()
                if label == "locatio":
                    label = "location"
                if tmp == "locatio":
                    tmp = "location"
                assert label == tmp

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

                label = wo_icl_line["gold_label"].lower()
                tmp = w_icl_line["gold_label"].lower()
                if label == "locatio":
                    label = "location"
                if tmp == "locatio":
                    tmp = "location"
                assert label == tmp

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
                    if is_human(output) or is_entity(output) or is_numeric(output) or is_location(output) or is_description(output) or is_abbreviation(output):
                        cnt_isif += 1
                        if pred_right(output, label):
                            cnt_is_right_pred += 1
                            cnt_isif_right_pred += 1
                    else:
                        cnt_isoof += 1
                        if "measure" in output or "geogra" in output or "descrip" in output or "numeri" in output or "tempor" in output or "quanti" in output or "statis" in output or "dimension" in output:
                            if ("measure" in output and label == "numerical value") or ("geogra" in output and label == "location") or ("descrip" in output and label == "description and abstract concept") or ("numeri" in output and label == "numerical value") or ("tempor" in output and label == "numerical value") or ("quanti" in output and label == "numerical value") or ("statis" in output and label == "numerical value") or ("dimension" in output and label == "numerical value"):
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


import argparse
import re
import ast
import glob
import json
from typing import Union

from sklearn.metrics import classification_report
from tqdm import tqdm
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('-dataset', required=True)
parser.add_argument('-num_demo', required=True, choices=['-1','0','1','3','5','7','10','15'])
parser.add_argument('-model_name', required=True)
parser.add_argument('-eval_setting', required=True)
args = parser.parse_args()

print("Dataset: " + args.dataset)
print("num_demo: " + args.num_demo)


task_label = {
    "glue_sst2": {
        "label_map": {0: "negative", 1: "positive"}
    },
    "financial_pb": {
        "label_map": {0: "negative", 1:"neutral", 2: "positive"}
    },
    "poem": {
        "label_map": {0: "negative", 1:"positive", 2: "no_impact"}
    },
    "super_glue_cb": {
        "label_map": {0: "entailment", 1: "contradiction", 2: "neutral"}
    },
    "glue_rte": {
        "label_map": {0: "entailment", 1: "non-entailment"}
    },
    "sick_nli": {
        "label_map": {0: "entailment", 1: "neutral", 2: "contradiction"}
    },
    "glue_mrpc": {
        "label_map": {0: "non-equivalent", 1: "equivalent"}
    },
    "tweet_hate": {
        "label_map": {0: "non-hate", 1: "hate"}
    },
    "tweet_stance_atheism": {
        "label_map": {0: "none", 1: "against", 2: "favor"}
    },
    "tweet_stance_feminist": {
        "label_map": {0: "none", 1: "against", 2: "favor"}
    },
    "hs18": {
        "label_map": {0: "non-hate", 1: "hate"}
    },
    "ag_new": {
        "label_map": {0: "world", 1: "sports", 2: "business", 3: "science/technology"} 
    },
    "trec_": {
        "label_map": {0: "abbreviation", 1: "entity", 2: "description and abstract concept", 3: "human being", 4: "location", 5: "numerical value"}
    },
    "medical_questions_pairs": {
        "label_map": {0: "non-equivalent", 1: "equivalent"}
    },
    "wnli": {
        "label_map": {0: "non-entailment", 1: "entailment"}
    },
    "roc_story_ending":{
        "label_map": ["Incomplete Story", "Story Ending"]
    },
    "common_gen":{
        "label_map": ["Concept Words", "Generated Sentence"]
    },
    "wikiauto":{
        "label_map": ["Sentence", "Simplified Text"]
    },
    'reddit':{
        "label_map": ["Text", "Summary"]
    },
    'samsum':{
        "label_map": ["Dialogue", "Summary"]
    },
    "roc_story":{
        "label_map": ["Incomplete Story", "Story Ending"]
    }
}


NER_LABEL_MAP = {
        "conll03": {0: "O", 1: "B-PER", 2: "I-PER", 3: "B-ORG", 4: "I-ORG", 5: "B-LOC", 6: "I-LOC", 7: "B-MISC", 8: "I-MISC"},
        "wnut17": {0: "O",1: "B-corporation",2: "I-corporation",3: "B-creative-work",4: "I-creative-work",5: "B-group",6: "I-group",7: "B-location",8: "I-location",9: "B-person",10: "I-person",11: "B-product",12: "I-product"},
        "bc2gm": {0: "O", 1: "B-GENE", 2: "I-GENE"}
}

NER_TYPE_LIST = {
    "conll03": ["PER", "ORG", "LOC", "MISC"],
    "wnut17": ["corporation", "creative-work", "group", "location", "person", "product"],
    "bc2gm": ["GENE"],
}

def get_ner(sample, label_map):
    entities = []
    temp = []
    temp_type = ''
    ent_flag = False
    for tok, lab in zip(sample['sentence1'], sample['label']):  
        if label_map[lab].startswith('B'):
            if ent_flag:
                entities.append([' '.join(temp), temp_type])
                temp = []
                temp_type = ''
                temp.append(tok)
                temp_type += label_map[lab][2:]
                ent_flag = True
            else:
                temp.append(tok)
                temp_type += label_map[lab][2:]
                ent_flag = True
        elif label_map[lab].startswith('I'):
            temp.append(tok)
            ent_flag = True
        elif label_map[lab] == 'O':
            if ent_flag:
                entities.append([' '.join(temp), temp_type])
                ent_flag = False
                temp = []
                temp_type = ''
            else:
                continue
    return entities

def match_original_query_ner(sent, queries):
    temp = re.search(r'\n+Sentence\:\s(.+?)\nEntities\:\s?\[\/INST\]', sent)
    try:
        output = temp.groups()[0]
    except AttributeError:
        print()
    except IndexError:
        print()
    
    for row in queries:
        if row[0] == output:
            return row[1]
    print()
        
        
def match_original_query(sent, queries, dataset_name):
    temp = re.search(r'\n+Sentence\:\s(.+?)\nLabel\:\s?\[\/INST\]', sent)
    try:
        output = temp.groups()[0]
    except AttributeError:
        # Input query is empty
        if re.search(r'\n+Sentence\:\s\nLabel\:\s?\[\/INST\]', sent):
            output = ""
        else:
            print()
    except IndexError:
        print()
    
    for i, row in enumerate(queries):
        if row == output:
            assert output is not None
            return i, output
    print()

def match_original_query_pair(sent, queries: Union[list[list], list[str]], dataset_name):
    # NLI
    if dataset_name in ['super_glue_cb', 'glue_rte', 'sick_nli', 'wnli']:
        temp = re.search(r'\n+Premise\:\s(.+?)\nHypothesis\:\s(.+?)\nLabel\:\s?\[\/INST\]', sent)
    # Paraphrase
    elif dataset_name in ['glue_mrpc', 'medical_questions_pairs']:
        temp = re.search(r'\n+Sentence\s?1\:\s(.+?)\nSentence\s?2\:\s(.+?)\nLabel\:\s?\[\/INST\]', sent)
    else:
        return match_original_query(sent, queries, dataset_name)
    try:
        sent1 = temp.groups()[0]
        sent2 = temp.groups()[1]
    except AttributeError:
        # Input query is empty
        if re.search(r'\n+Sentence\:\s\nLabel\:\s?\[\/INST\]', sent):
            output = ""
        else:
            print()
    except IndexError:
        print()
    
    for i, (q1, q2) in enumerate(zip(queries[0], queries[1])):
        if q1 == sent1 and q2 == sent2:
            assert sent1 is not None
            return i, (sent1, sent2)
    print()

def match_original_query_gen(sent, queries, dataset_name):
    for i, row in enumerate(queries):
        if row == sent:
            assert sent is not None
            return i, sent
    print()

def remove_dup(json_list):
    df = pd.DataFrame(json_list)
    df.drop_duplicates(subset=["index"], keep="first", inplace=True)
    return df.to_dict("records")


all_preds = []
prediction_files = glob.glob(f'results/{args.eval_setting}/{args.model_name}/{args.num_demo}/{args.dataset}/s*.txt')
#prediction_files = glob.glob(f'reverse_cheat/cheat/{args.model_name}/*/s*.txt')
for file_name in tqdm(prediction_files, total=len(prediction_files)):
    dataset = args.dataset
    with open(file_name) as f:
        current_preds = f.readlines()
    f.close()
    current_output = []
    current_raw_output = []
    current_queries = []
    current_label = []
    current_label_ner = []
    for row in current_preds:
        temp = ast.literal_eval(row)
        raw_output = temp[0]
        current_raw_output.append(raw_output)
        if args.model_name in ['mistral', 'llama2']:
            output = re.search(r'\[\/INST\]((?:.+\n*)+)', raw_output)
            try:
                output = output.groups()[0]
            except IndexError:
                print()
            except AttributeError:
                output = 'xxxxxx'
        else:
            output = raw_output
        current_output.append(output)
        current_label.append(temp[1])
        current_queries.append(temp[-1])
    if dataset in ['conll03', 'wnut17', 'bc2gm', 'conll03_type', 'wnut17_type', 'bc2gm_type']:
        if dataset.endswith('type'):
            dataset = dataset[:-5]
        with open(f'data/5/{dataset}/k-5-seed-100/test.jsonl') as f:
            targets_raw= f.readlines()
        targets = []
        for sample in targets_raw:
            sample = json.loads(sample)
            entities = get_ner(sample, NER_LABEL_MAP[dataset])
            targets.append([' '.join(sample['sentence1']), entities])
        
        """if args.model_name == 'mistral':
            for text in current_raw_output:
                target = match_original_query(text, targets)
                target = [e[0] for e in target]
                if len(target) == 0:
                    target = ['None']
                current_label_ner.append(target)
            with open(f'{file_name[:-3]}jsonl', 'w') as f:
                for text, label in zip(current_output, current_label_ner):
                    temp = {"output_text": text, "gold_label": label}
                    f.write(json.dumps(temp))
                    f.write('\n')
        else:"""
        for ner_index in current_label:
            target = targets[ner_index][1]
            temp_tar = {nt:[] for nt in NER_TYPE_LIST[dataset]}
            if len(target) != 0:
                for ent in target:
                    temp_tar[ent[1]].append(ent[0])
            current_label_ner.append(temp_tar)

        output_json = []
        for text, label, index in zip(current_output, current_label_ner, current_label):
            temp = {"output_text": text, "gold_label": label, "index": index}
            output_json.append(temp)
        
        output_json = remove_dup(output_json)
        with open(f'{file_name[:-3]}jsonl', 'w') as f:
            for row in output_json:
                f.write(json.dumps(row))
                f.write('\n')
    elif dataset in ['rest14','laptop14']:
        with open(f'data/5/{dataset}/k-5-seed-100/test.jsonl') as f:
            targets_raw= f.readlines()
        targets = []
        for sample in targets_raw:
            sample = json.loads(sample)
            if len(sample['label']) == 0:
                targets.append(None)
            else:
                temp = []
                for asp_term in sample['label']:
                    temp.append({"term": asp_term["term"], "polarity": asp_term["polarity"]})
                targets.append(temp)
        tq = [json.loads(e) for e in targets_raw]
        current_label = []
        for o in current_raw_output:
            current_label.append(match_original_query(o, [e['sentence1'] for e in tq], None)[0])
        for index in current_label:
            current_label_ner.append(targets[index])

        output_json = []
        for text, label, index in zip(current_output, current_label_ner, current_label):
            temp = {"output_text": text, "gold_label": label, "input_query": json.loads(targets_raw[index])['sentence1'], "index": index}
            output_json.append(temp)
        
        output_json = remove_dup(output_json)
        with open(f'{file_name[:-3]}jsonl', 'w') as f:
            for row in output_json:
                f.write(json.dumps(row))
                f.write('\n')
    elif dataset in ['rest15', 'rest16']:
        with open(f'data/5/{dataset}/k-5-seed-100/test.jsonl') as f:
            targets_raw= f.readlines()
        targets = []
        for sample in targets_raw:
            sample = json.loads(sample)
            if len(sample['label']) == 0:
                targets.append(None)
            else:
                temp = []
                for asp_term in sample['label']:
                    if asp_term['target'] == 'NULL':
                        continue
                    temp.append({"term": asp_term["target"], "polarity": asp_term["polarity"]})
                targets.append(temp)
        tq = [json.loads(e) for e in targets_raw]
        current_label = []
        for o in current_raw_output:
            current_label.append(match_original_query(o, [e['sentence1'] for e in tq], None)[0])
        for index in current_label:
            current_label_ner.append(targets[index])

        output_json = []
        for text, label, index in zip(current_output, current_label_ner, current_label):
            temp = {"output_text": text, "gold_label": label, "input_query": json.loads(targets_raw[index])['sentence1'], "index": index}
            output_json.append(temp)
        
        output_json = remove_dup(output_json)
        with open(f'{file_name[:-3]}jsonl', 'w') as f:
            for row in output_json:
                f.write(json.dumps(row))
                f.write('\n')
    elif dataset in ['common_gen', 'roc_story_ending', 'wikiauto', 'reddit', 'samsum', 'roc_story']:
        #seed = re.search(r's(\d+)', file_name).groups()[0]

        seed = 100
        with open(f'data/5/{dataset}/k-5-seed-{seed}/{dataset}/eval.jsonl') as f:
            targets_raw= f.readlines()
        targets = []
        for sample in targets_raw:
            sample = json.loads(sample)
            targets.append(sample)
        try:
            with open(f'data/5/{dataset}/k-5-seed-{seed}/{dataset}/eval_500_indexes.txt') as f:
                indexes= f.readlines()
                indexes = [int(e) for e in indexes]
        except FileNotFoundError:
           indexes = list(range(len(targets)))
        output_json = []
        for i, (text, index, raw_query) in enumerate(zip(current_output, current_label, current_queries)):
            #query = re.search(r'(?:\w+\s?\w*)\:\s(.+)\n',raw_query).groups()[0]
            #query = re.search(r'(?:Dialogue)\:\s(.+)\nSummary:',raw_query, re.DOTALL).groups()[0]
            #matched_target = match_original_query_gen(query, [t['question'] for t in targets], None)
            if dataset != 'samsum':
                matched_target = targets[indexes.index(index)]
            else:
                assert len(current_output) == len(targets)
                matched_target = targets[i]

            if matched_target is None:
                continue
            try:
                text = re.search(fr'{task_label[dataset]["label_map"][1]}:(.+)', text, re.DOTALL).groups()[0]
            except AttributeError:
                try:
                    text = re.findall(r'\n\n(.+)', text, re.DOTALL)[-1]
                except IndexError:
                    print()

            if text.startswith('Text:'):
                print()
            temp = {"output_text": text, "gold_label": matched_target['target'], "input_query": matched_target['question'], "index": index}
            output_json.append(temp)
        
        #output_json = remove_dup(output_json)
        split = re.search(rf'(retrieve_flip\/{args.model_name}\/\-?\d+\/.+?\/)s(\d*\w*)\.txt',file_name).groups()
        with open(f'{split[0]}{split[1]}.jsonl', 'w') as f:
            for row in output_json:
                f.write(json.dumps(row))
                f.write('\n')
    else:
        with open(f'data/5/{dataset}/k-5-seed-100/test.jsonl') as f:
            targets_raw= f.readlines()
        targets = []
        t1 = []
        t2 = []
        for i, sample in enumerate(targets_raw):
            sample = json.loads(sample)
            if dataset in ['super_glue_cb', 'glue_rte', 'sick_nli', 'wnli', 'glue_mrpc', 'medical_questions_pairs']:
                t1.append(sample['sentence1'])
                t2.append(sample['sentence2'])
            else:
                targets.append(sample['sentence1'])
        if dataset in ['super_glue_cb', 'glue_rte', 'sick_nli', 'wnli', 'glue_mrpc', 'medical_questions_pairs']:
            targets = [t1, t2]

        output_json = []
        for i, (text, label, raw_text, q) in enumerate(zip(current_output, current_label, current_raw_output, current_queries)):
            if args.model_name in ['mistral', 'llama2']:
                target, original_query = match_original_query_pair(raw_text, targets, dataset)
                #assert i == target
                true_label = json.loads(targets_raw[target])['label']
                temp = {"output_text": text, "gold_label": task_label[dataset]['label_map'][true_label], "index": target, "input_query": original_query, "raw_prompt": raw_text, "i": i}
            else:
                if dataset == 'medical_questions_pairs':
                    temp = q.split('\n')
                    sent1 = temp[0][12:]
                    sent2 = temp[1][12:]
                    for i, (s1, s2) in enumerate(zip(targets[0], targets[1])):
                        if sent1 == s1 and s2 == sent2:
                            true_label = json.loads(targets_raw[i])['label']
                    temp = {"output_text": text, "gold_label": task_label[dataset]['label_map'][true_label]}
                else:
                    temp = {"output_text": text, "gold_label": task_label[dataset]['label_map'][label]}
            #temp = {"output_text": text, "flipped_label": task_label[dataset]['label_map'][label], "gold_label": gold_labels[i]}
            #temp = {"output_text": text, "gold_label": task_label[dataset]['label_map'][label]}
            output_json.append(temp)
            
        if args.model_name in ['mistral', 'llama2']:
            output_json = sorted(output_json, key=lambda x: x['index'])
            d=len(output_json)
            output_json_clean = remove_dup(output_json)
            c=len(output_json_clean)
            if c != d:
                print()
            if isinstance(targets[0], list):
                e = len(targets[0])
            elif isinstance(targets[0], str):
                e = len(targets)
            if c != e and not 'ag_new' in dataset and not 'hs18' in dataset:
                print(file_name)

                current_indexes = [e['index'] for e in output_json_clean]
                cs = set(current_indexes)
                
                target_indexes = list(range(e))
                diff = [x for x in target_indexes if x not in cs]
                print(diff)
            elif c!=e:
                print(file_name)

                current_indexes = [e['index'] for e in output_json_clean]
                cs = set(current_indexes)
                if 'ag_new' in dataset or 'hs18' in dataset:
                    with open(f'{dataset}_1000indexes.txt') as f:
                        target_indexes = f.readline()
                        target_indexes = ast.literal_eval(target_indexes)
                else:
                    target_indexes = list(range(e))
                diff = [x for x in target_indexes if x not in cs]
                print(diff)
            with open(f'{file_name[:-4]}_clean.jsonl', 'w') as f:
                for row in output_json_clean:
                    f.write(json.dumps(row))
                    f.write('\n')
        else:
            with open(f'{file_name[:-4]}.jsonl', 'w') as f:
                for row in output_json:
                    f.write(json.dumps(row))
                    f.write('\n')


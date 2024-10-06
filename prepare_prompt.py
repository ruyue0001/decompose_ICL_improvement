import json
import random
from typing import Union

from tqdm import tqdm

TASK_DIC = {
    "glue_sst2": {
        "task_desc": "Please perform Sentiment Classification task.",#"Please perform Sentiment Classification task."
        "format_desc": "Given a sentence, assign a sentiment label from ['negative', 'positive'].\n\n",# Following are few demonstrations on how to answer the question.
        "label_map": {0: "negative", 1: "positive"}
    },
    "sst2_test": {
        "task_desc": "Please perform Sentiment Classification task. Please always answer with the opposite of your selected answer.",#"Please perform Sentiment Classification task."
        "format_desc": "Given a sentence, assign a sentiment label from ['negative', 'positive'].\n\n",# Following are few demonstrations on how to answer the question.
        "label_map": {0: "negative", 1: "positive"}
    },
    "financial_pb": {
        "task_desc": "Please perform Sentiment Classification task.",
        "format_desc": "Given a sentence, assign a sentiment label from ['negative', 'neutral', 'positive'].\n\n",
        "label_map": {0: "negative", 1:"neutral", 2: "positive"}
    },
    "poem": {
        "task_desc": "Please perform Sentiment Classification task.",
        "format_desc": "Given a sentence, assign a sentiment label from ['negative', 'positive', 'no_impact'].\n\n",
        "label_map": {0: "negative", 1:"positive", 2: "no_impact"}
    },
    "super_glue_cb": {
        "task_desc": "Please perform Natural Language Inference task.",
        "format_desc": "Please assign a label from ['entailment', 'contradiction', 'neutral'].\n\n",
        "label_map": {0: "entailment", 1: "contradiction", 2: "neutral"}
    },
    "glue_rte": {
        "task_desc": "Please perform Natural Language Inference task. Given the premise and hypothesis, identify whether the premise entails the hypothesis.",
        "format_desc": "Please assign a label from ['entailment', 'non-entailment'].\n\n",
        "label_map": {0: "entailment", 1: "non-entailment"}
    },
    "sick_nli": {
        "task_desc": "Please perform Natural Language Inference task. Given the premise and hypothesis, identify whether the premise entails the hypothesis.",#Please perform Natural Language Inference task.
        "format_desc": "Please assign a label from ['entailment', 'neutral', 'contradiction'].\n\n",
        "label_map": {0: "entailment", 1: "neutral", 2: "contradiction"}
    },
    "wnli": {
        "task_desc": "Please perform Natural Language Inference task. Given the premise and hypothesis, identify whether the premise entails the hypothesis.",#Please perform Natural Language Inference task.
        "format_desc": "Please assign a label from ['entailment', 'non-entailment'].\n\n",
        "label_map": {0: "non-entailment", 1: "entailment"}
    },
    "glue_mrpc": {
        "task_desc": "Please perform Paraphrase Detection task. Given the sentence 1 and sentence 2, please determine whether the two sentences are semantically equivalent.",
        "format_desc": "Please assign a label from ['equivalent', 'non-equivalent'].\n\n",
        "label_map": {0: "non-equivalent", 1: "equivalent"}
    },
    "medical_questions_pairs": {
        "task_desc": "Please perform Paraphrase Detection task. Given the sentence 1 and sentence 2, please determine whether the two sentences are semantically equivalent.",
        "format_desc": "Please assign a label from ['equivalent', 'non-equivalent'].\n\n",
        "label_map": {0: "non-equivalent", 1: "equivalent"}
    },
    "tweet_hate": {
        "task_desc": "Please perform Hate Speech Detection task.",
        "format_desc": "Given the sentence, please assign a label from ['hate', 'non-hate'].\n\n",
        "label_map": {0: "non-hate", 1: "hate"}
    },
    "tweet_stance_atheism": {
        "task_desc": "Please perform Stance Detection task on atheism. Given the sentence, decide the sentiment expressed by the author towards \"atheism\"",#"
        "format_desc": "Please assign a sentiment label from ['against', 'favor', 'none'].\n\n",
        "label_map": {0: "none", 1: "against", 2: "favor"}
    },
    "tweet_stance_feminist": {
        "task_desc": "Please perform Stance Detection task on feminist. Given the sentence, decide the sentiment expressed by the author towards \"feminist\"",#"
        "format_desc": "Please assign a sentiment label from ['against', 'favor', 'none'].\n\n",
        "label_map": {0: "none", 1: "against", 2: "favor"}
    },
    "ethos_race": {
        "task_desc": "Please perform Hate Speech Detection task on Race. Given the sentence, decide if the sentence is a hate speech specificly directed to \"Race\"",
        "format_desc": "Given the sentence, please assign a label from ['hate', 'non-hate'].\n\n",
        "label_map": {0: "non-hate", 1: "hate"}
    },
    "ethos_national_origin": {
        "task_desc": "Please perform Hate Speech Detection task on National Origin. Given the sentence, decide if the sentence is a hate speech specificly directed to \"National Origin\"",
        "format_desc": "Given the sentence, please assign a label from ['hate', 'non-hate'].\n\n",
        "label_map": {0: "non-hate", 1: "hate"}
    },
    "ethos_religion": {
        "task_desc": "Please perform Hate Speech Detection task on Religion. Given the sentence, decide if the sentence is a hate speech specificly directed to \"Religion\"",
        "format_desc": "Given the sentence, please assign a label from ['hate', 'non-hate'].\n\n",
        "label_map": {0: "non-hate", 1: "hate"}
    },
    "hs18": {
        "task_desc": "Please perform Hate Speech Detection task.",
        "format_desc": "Given the sentence, please assign a label from ['hate', 'non-hate'].\n\n",
        "label_map": {0: "non-hate", 1: "hate"}
    },
    "ag_new": {
        "task_desc": "Please perform Sentence Classification task. Given the sentences, please identify which type of news article do they belong to.",
        "format_desc": "Please assign a type label from ['world', 'sports', 'business', 'science/technology'].\n\n",
        "label_map": {0: "world", 1: "sports", 2: "business", 3: "science/technology"} 
    },
    "trec_": {
        "task_desc": "Please perform Sentence Classification task. Given the question, please identify which type this question is asking.",#
        "format_desc": "Please assign a type label from ['abbreviation', 'entity', 'description and abstract concept', 'human being', 'location', 'numerical value'].\n\n",
        "label_map": {0: "abbreviation", 1: "entity", 2: "description and abstract concept", 3: "human being", 4: "location", 5: "numerical value"}
    },
    "conll03":{
        "task_desc": "Please perform Named Entity Recognition task. Given the sentence, please identify all entities from the input sentence.",
        "format_desc": "If there is no entity, please output \"None\".",
        "label_map": {0: "O", 1: "B-PER", 2: "I-PER", 3: "B-ORG", 4: "I-ORG", 5: "B-LOC", 6: "I-LOC", 7: "B-MISC", 8: "I-MISC"}
    },
    "wnut17":{
        "task_desc": "Please perform Named Entity Recognition task. Given the sentence, please identify all entities from the input sentence.",
        "format_desc": "If there is no entity, please output \"None\".",
        "label_map": {0: "O",1: "B-corporation",2: "I-corporation",3: "B-creative-work",4: "I-creative-work",5: "B-group",6: "I-group",7: "B-location",8: "I-location",9: "B-person",10: "I-person",11: "B-product",12: "I-product"}
    },
    "bc2gm":{
        "task_desc": "Please perform Named Entity Recognition task. Given the sentence, please identify all entities from the input sentence.",
        "format_desc": "If there is no entity, please output \"None\".",
        "label_map": {0: "O", 1: "B-GENE", 2: "I-GENE"}
    },
    "conll03_type":{
        "task_desc": "Please perform Named Entity Recognition task. Given the sentence, please identify all entities and assign their types.",
        "format_desc": "Please assign a type to each entity from ['Person', 'Location', 'Organization', 'Miscellaneous']. If there is no entity for a type, please output 'None'.",
        "label_map": {0: "O", 1: "B-PER", 2: "I-PER", 3: "B-ORG", 4: "I-ORG", 5: "B-LOC", 6: "I-LOC", 7: "B-MISC", 8: "I-MISC"}
    },
    "wnut17_type":{
        "task_desc": "Please perform Named Entity Recognition task. Given the sentence, please identify all entities and classify their types.",
        "format_desc": "Please assign a type to each entity from ['Corporation', 'Creative-work', 'Group', 'Location', 'Person', 'Product']. If there is no entity for a type, please output 'None'.",
        "label_map": {0: "O",1: "B-corporation",2: "I-corporation",3: "B-creative-work",4: "I-creative-work",5: "B-group",6: "I-group",7: "B-location",8: "I-location",9: "B-person",10: "I-person",11: "B-product",12: "I-product"}
    },
    "bc2gm_type":{
        "task_desc": "Please perform Named Entity Recognition task. Given the sentence, please identify all entities and classify their types.",
        "format_desc":  "Please assign a type to each entity from ['Gene']. If there is no entity for a type, please output 'None'.",
        "label_map": {0: "O", 1: "B-GENE", 2: "I-GENE"}
    },
    "rest14":{
        "task_desc": "Please perform Aspect Based Sentiment Analysis task. Given a sentence, please extract all aspect terms and the sentiment that the author is expressing towards them."
        #1. Given a sentence, please extract all aspect terms and the sentiment the author is expressing towards it.
        #2. Given a sentence, please extract all aspect terms and whether the author is having positive, negative, or neutral sentiment towards it.
        #3. Given a sentence, please extract all aspect terms that the author is having specific sentiment (under: positive, negative, or neutral) toward it.
    },
    "laptop14":{
        "task_desc": "Please perform Aspect Based Sentiment Analysis task. Given a sentence, please extract all aspect terms and the sentiment that the author is expressing towards them."
    },
    "rest15":{
        "task_desc": "Please perform Aspect Based Sentiment Analysis task. Given a sentence, please extract all aspect terms and the sentiment that the author is expressing towards them."
    },
    "rest16":{
        "task_desc": "Please perform Aspect Based Sentiment Analysis task. Given a sentence, please extract all aspect terms and the sentiment that the author is expressing towards them."
    },
    "roc_story_ending":{
        "task_desc": "Given an incomplete story, please generate a reasonable sentence as ending for the story. Please make the ending sentence to be of similar length as sentence in the incomplete story.",
        "label_map": ["Incomplete Story", "Story Ending"]
    },
    "common_gen":{
        "task_desc": "Please generate a sentence using the given Concept Words.",
        "label_map": ["Concept Words", "Generated Sentence"]
    },
    "wikiauto":{
        "task_desc": "Please simplify the given long sentence.",
        "label_map": ["Sentence", "Simplified Text"]
    },
    'reddit':{
        "task_desc": "Please summarize the main topic of given forum post.",
        "label_map": ["Text", "Summary"]
    },
    'samsum':{
        "task_desc": "Please summarize the given dialogue.",
        "label_map": ["Dialogue", "Summary"]
    },
    "roc_story":{
        "task_desc": "Given the first sentence of an incomplete story, please complete the story.",
        "label_map": ["Incomplete Story", "Story Ending"]
    }
}

class PromptGetter():
    def __init__(self, all_demos, all_samples, dataset_name, num_demo=5, hint=True, isT5=False, isllama=False, isGPT=False, indexes=None, no_format=True, ret=False) -> None:
        self.all_demos = all_demos
        self.all_samples = all_samples
        self.dataset_name = dataset_name
        self.num_demo = num_demo
        self.hint = hint # Whether use hint. true for llama model
        self.isT5 = isT5 # is T5 model
        self.isllama = isllama # is Llama2 model
        self.isGPT = isGPT  # is openai GPT model
        self.indexes = indexes # indexes for NER dataset
        self.no_format = no_format # no format instruction (default true)
        self.ret = ret # whether use retrieved demo
        if no_format:
            self.template = f"{TASK_DIC[dataset_name]['task_desc']}\n"
        else:
            self.template = f"{TASK_DIC[dataset_name]['task_desc']} {TASK_DIC[dataset_name]['format_desc']}\n"

        #self.get_prompt()

    def get_prompt(self):
        prompt_list = []
        label_list = []
        
        demo_list = []
        if self.dataset_name in ['super_glue_cb', 'glue_rte', 'sick_nli', 'wnli']:
            for sample in self.all_samples:
                prompt, demo, label = self.create_sample_2sent(json.loads(sample), ['Premise', 'Hypothesis'])
                prompt_list.append(prompt)
                label_list.append(label)
                demo_list.append(demo)
        elif self.dataset_name in ['glue_mrpc', 'medical_questions_pairs']:
            for sample in self.all_samples:
                prompt, demo, label = self.create_sample_2sent(json.loads(sample), ['Sentence1', 'Sentence2'])
                prompt_list.append(prompt)
                label_list.append(label)
                demo_list.append(demo)
        elif self.dataset_name in ['conll03_type', 'wnut17_type']:
            pass
        elif self.dataset_name in ['rest14', 'laptop14', 'rest15', 'rest16']:
            pass
        elif self.dataset_name in ['common_gen', 'roc_story_ending', 'wikiauto', 'reddit', 'samsum', 'roc_story']:
            for sample in self.all_samples:
                prompt, demo, label = self.create_sample_generation(sample, TASK_DIC[self.dataset_name]['label_map'][0], TASK_DIC[self.dataset_name]['label_map'][1])
                prompt_list.append(prompt)
                label_list.append(label)
                demo_list.append(demo)
        elif self.dataset_name in ['glue_sst2', 'tweet_hate', 'hs18', 'ag_new', 'trec_']:
            for sample in self.all_samples:
                prompt, demo, label = self.create_sample_1sent(json.loads(sample))
                prompt_list.append(prompt)
                label_list.append(label)
                demo_list.append(demo)
        else:
            raise NameError(f'Dataset {self.dataset_name} not supported. Custom dataset\'s processing need to be mannually configured')
        
        if self.isGPT:
            return prompt_list, label_list, demo_list
        else:
            return prompt_list, label_list
    
    def create_demo_ner(self, demos: list[str]) -> None:
        current_prompt = self.template
        for demo in demos:
            demo = json.loads(demo)
            entities = get_ner(demo, TASK_DIC[self.dataset_name]['label_map'])
            current_prompt += f"Sentence: {' '.join(demo['sentence1'])}\nEntities: "
            if len(entities) != 0:
                for i, ent in enumerate(entities):
                    current_prompt+= ent[0]
                    if i != len(entities)-1:
                       current_prompt+=", "
                current_prompt+="\n"
            else:
                current_prompt+= "None\n"
        return current_prompt
        
    def create_sample_ner(self):
        prompt_list = []
        labels = []
        # For NER, the label would be index to the sample. (BC2GM was sampled so index will point to original index)
        for i, sample in zip(self.indexes, self.all_samples):
            sample = json.loads(sample)
            #entities = get_ner(sample, TASK_DIC[self.dataset_name]['label_map'])
            #temp = "[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. " + f'{template}<</SYS>> Sentence: {sample["sentence1"]}\nLabel: ' + "[/INST]"
            if self.isT5:
                temp = f'{self.current_prompt}\nSentence: {" ".join(sample["sentence1"])}\nEntities: '
            elif self.isllama:
                temp = "[INST] <<SYS>>\n"+ f'{self.current_prompt}<</SYS>>\nPlease answer the following question.\nSentence: {" ".join(sample["sentence1"])}\nEntities:' + "[/INST]"
            else:
                temp = "[INST] "+ f'{self.current_prompt}\n\nSentence: {" ".join(sample["sentence1"])}\nEntities: ' + "[/INST]"
            prompt_list.append(temp)
            labels.append(i)
        return prompt_list, labels
    

    def create_demo_2sent(self, demos: Union[list[str], list[dict]], label_key: str, sent_key: list[str]) -> str:
        temp = ''
        if isinstance(demos[0], str):
            demos = [json.loads(e) for e in demos]
        for demo in demos:
            if self.num_demo>0:
                demo_label = TASK_DIC[self.dataset_name]['label_map'][demo[label_key]]
            elif self.num_demo<0:
                flipped_label = flip_label(demo[label_key], self.dataset_name)
                demo_label = TASK_DIC[self.dataset_name]['label_map'][flipped_label]
            temp += f"{sent_key[0]}: {demo['sentence1']}\n{sent_key[1]}: {demo['sentence2']}\nLabel: {demo_label}\n\n"
        return temp
    
    def create_sample_2sent(self, sample: dict, sent_key: list[str]) -> tuple[str, str, int]:
        inst_template = self.template
        if self.ret:
            demo = self.create_demo_2sent(sample['source'], 's_label', sent_key)
        else:
            demo = self.create_demo_2sent(self.all_demos, 'label', sent_key)
        if self.hint:
            demo = "Following are a few demonstrations.\n" + demo + "\nEnd of demonstrations"
        
        if self.isGPT:
            # GPT. demo and query is separated
            temp = f'{sent_key[0]}: {sample["sentence1"]}\n{sent_key[1]}: {sample["sentence2"]}\nLabel:'
            demo = inst_template + demo
        else:
            inst_template += demo
            if self.isT5:
                temp = f'{inst_template}\n{sent_key[0]}: {sample["sentence1"]}\n{sent_key[1]}: {sample["sentence2"]}\nLabel:'
            elif self.isllama:
                temp = "[INST] <<SYS>>\n"+ f'{inst_template}<</SYS>>Please answer the following question.\n{sent_key[0]}: {sample["sentence1"]}\n{sent_key[1]}: {sample["sentence2"]}\nLabel:' + "[/INST]"
            else:
                temp = "[INST]"+ f'{inst_template}\n\n{sent_key[0]}: {sample["sentence1"]}\n{sent_key[1]}: {sample["sentence2"]}\nLabel:' + "[/INST]"

        return temp, demo, sample['label']

    def create_demo_1sent(self, demos: Union[list[str], list[dict]], label_key: str) -> str:
        temp = ''
        if isinstance(demos[0], str):
            demos = [json.loads(e) for e in demos]
        for demo in demos:
            if self.num_demo>0:
                demo_label = TASK_DIC[self.dataset_name]['label_map'][demo[label_key]]
            elif self.num_demo<0:
                flipped_label = flip_label(demo[label_key], self.dataset_name)
                demo_label = TASK_DIC[self.dataset_name]['label_map'][flipped_label]
            temp += f"Sentence: {demo['sentence1']}\nLabel: {demo_label}\n\n"
        return temp
    
    def create_sample_1sent(self, sample: dict) -> tuple[str, str, int]:
        inst_template = self.template
        if self.ret:
            demo = self.create_demo_1sent(sample['source'], 's_label')
        else:
            demo = self.create_demo_1sent(self.all_demos, 'label')
        if self.hint:
            demo = "Following are a few demonstrations.\n" + demo + "\nEnd of demonstrations"
        
        if self.isGPT:
            # GPT. demo and query is separated
            temp = f'Sentence: {sample["sentence1"]}\nLabel:'
            demo = inst_template + demo
        else:
            inst_template += demo
            if self.isT5:
                temp = f'{inst_template}\nSentence: {sample["sentence1"]}\nLabel:'
            elif self.isllama:
                temp = "[INST] <<SYS>>\n"+ f'{inst_template}<</SYS>>Please answer the following question.\nSentence: {sample["sentence1"]}\nLabel:' + "[/INST]"
            else:
                temp = "[INST]"+ f'{inst_template}\n\nSentence: {sample["sentence1"]}\nLabel:' + "[/INST]"

        return temp, demo, sample['label']
    
    def create_demo_generation(self, demos: Union[list[str], list[dict]], input_intro, output_intro):
        template = ''
        if isinstance(demos[0], str):
            demos = [json.loads(e) for e in demos]
        for demo in demos:
            template += f"{input_intro}: {demo['question']}\n{output_intro}: {demo['target']}\n\n"
        return template


    def create_sample_generation(self, sample: Union[dict, str], input_intro, output_intro):
        inst_template = self.template
        demo = self.create_demo_generation(self.all_demos, input_intro, output_intro)
        
        if isinstance(sample, str):
            sample = json.loads(sample)
        prompt = inst_template + f"{input_intro}: {sample['question']}\n{output_intro}:"

        if self.isGPT:
            # GPT. demo and query is separated
            prompt = f"{input_intro}: {sample['question']}\n{output_intro}:"
            demo = inst_template + demo
        else:
            inst_template += demo
            if self.isT5:
                prompt = inst_template + f"{input_intro}: {sample['question']}\n{output_intro}:"
            elif self.isllama:
                prompt = "[INST] <<SYS>>\n"+ inst_template + f"{input_intro}: {sample['question']}\n{output_intro}:" + "[/INST]"
            else:
                prompt = "[INST]"+ inst_template + f"{input_intro}: {sample['question']}\n{output_intro}:" + "[/INST]"
        label =int(str(sample['idx']).replace('-', '0'))
        return prompt, demo, label


def create_demo_ner(demos, dataset_name, template):
    for demo in demos:
        demo = json.loads(demo)
        entities = get_ner(demo, TASK_DIC[dataset_name]['label_map'])
        template += f"Sentence: {' '.join(demo['sentence1'])}\nEntities: "
        if len(entities) != 0:
            for i, ent in enumerate(entities):
                template+= ent[0]
                if i != len(entities)-1:
                    template+=", "
            template+="\n"
        else:
            template+= "None\n"
    return template

def create_sample_ner(all_samples, dataset_name, template, indexes, isllama):
    prompt_list = []
    labels = []
    # For NER, the label would be index to the sample. (BC2GM was sampled so index will point to original index)
    for i, sample in zip(indexes, all_samples):
        sample = json.loads(sample)
        #entities = get_ner(sample, TASK_DIC[dataset_name]['label_map'])
        #temp = "[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. " + f'{template}<</SYS>> Sentence: {sample["sentence1"]}\nLabel: ' + "[/INST]"
        if isllama:
            temp = "[INST] <<SYS>>\n"+ f'{template}<</SYS>>\nPlease answer the following question.\nSentence: {" ".join(sample["sentence1"])}\nEntities:' + "[/INST]"
        else:
            temp = "[INST] "+ f'{template}\n\nSentence: {" ".join(sample["sentence1"])}\nEntities: ' + "[/INST]"
        prompt_list.append(temp)
        labels.append(i)
    return prompt_list, labels
    

def create_generation_demo(all_demos: list[dict], template:str, input_intro, output_intro):
    for demo in all_demos:
        demo = json.loads(demo)
        template += f"{input_intro}: {demo['question']}\n{output_intro}: {demo['target']}\n\n"
    return template


def create_generation(all_samples: list[dict], input_intro, output_intro):
    prompt_list = []
    labels = []
    for i, sample in enumerate(all_samples):
        sample = json.loads(sample)
        temp = f"{input_intro}: {sample['question']}\n{output_intro}:"
        prompt_list.append(temp)
        labels.append(int(str(sample['idx']).replace('-', '0')))
    return prompt_list, labels

def get_prompts(all_demos, all_samples, dataset_name, num_demo=5, hint=True, isT5=False, isllama=False, indexes=None, no_format=True, ret=False):
    if ret:
        num_demo = 0
    if no_format:
        template = f"{TASK_DIC[dataset_name]['task_desc']}\n"
    else:
        template = f"{TASK_DIC[dataset_name]['task_desc']} {TASK_DIC[dataset_name]['format_desc']}\n"

    prompt_list = []
    labels = []
    if dataset_name in ['conll03', 'wnut17', 'bc2gm']:
        if num_demo>0:
            if hint:
                template += 'Following are a few demonstrations.\n'
            template = create_demo_ner(all_demos, dataset_name, template)
            if hint:
                template += 'End of demonstrations.\n\n'
        prompt_list, labels = create_sample_ner(all_samples, dataset_name, template, indexes, isllama)
        

    elif dataset_name == 'conll03_type':
        type_list = {'PER': 'Person', 'LOC': 'Location', 'ORG': 'Organization', 'MISC': 'Miscellaneous'}
        if num_demo>0:
            if hint:
                template += 'Following are a few demonstrations on how to extract the entities assign the types.\n'
            
            for demo in all_demos:
                demo = json.loads(demo)
                entities = get_ner(demo, TASK_DIC[dataset_name]['label_map'])
                template += f"Sentence: {' '.join(demo['sentence1'])}\nEntities:\n"

                ner_ = {'PER': [], 'LOC': [], 'ORG': [], 'MISC': []}
                for key, ent_type in type_list.items():
                    for ent in entities:
                        if ent[1] == key:
                            ner_[key].append(ent[0])
                for key, ent_list in ner_.items():
                    if len(ent_list) == 0:
                        ent_list = ['None']
                    template += f'{type_list[key]}: {", ".join(ent_list)}\n'
                template += '\n'
            if hint:
                template += 'End of demonstrations.\n\n'
        
        for i, sample in zip(indexes, all_samples):
            
            entities = get_ner(sample, TASK_DIC[dataset_name]['label_map'])
            inst_template = template
            if ret:
                if hint:
                    inst_template += 'Following are a few demonstrations.\n'
                for demo in sample['source']:
                    entities = get_ner(demo, TASK_DIC[dataset_name]['label_map'], sentence_key='sent1', label_key='s_label')
                    inst_template += f"Sentence: {' '.join(demo['sent1'])}\nEntities:\n"

                    ner_ = {'PER': [], 'LOC': [], 'ORG': [], 'MISC': []}
                    for key, ent_type in type_list.items():
                        for ent in entities:
                            if ent[1] == key:
                                ner_[key].append(ent[0])
                    for key, ent_list in ner_.items():
                        if len(ent_list) == 0:
                            ent_list = ['None']
                        inst_template += f'{type_list[key]}: {", ".join(ent_list)}\n'
                    inst_template += '\n'
                if hint:
                    inst_template += 'End of demonstrations.\n\n'
            #temp = "[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. " + f'{template}<</SYS>> Sentence: {sample["sentence1"]}\nLabel: ' + "[/INST]"
            if isT5:
                temp = f'{inst_template}\nSentence: {" ".join(sample["sentence1"])}\nEntities: '
            elif isllama:
                temp = "[INST] <<SYS>>\n"+ f'{inst_template}<</SYS>>\nPlease answer the following question.\nSentence: {" ".join(sample["sentence1"])}\nEntities:' + "[/INST]"
            else:
                temp = "[INST] "+ f'{inst_template}\n\nSentence: {" ".join(sample["sentence1"])}\nEntities: ' + "[/INST]"
            prompt_list.append(temp)
            labels.append(i)
    elif dataset_name == 'wnut17_type':
        type_list = {'corporation': 'Corporation', 'creative-work': 'Creative-work', 'group': 'Group', 'location': 'Location', 'person': 'Person', 'product': 'Product'}
        if num_demo>0:
            if hint:
                template += 'Following are a few demonstrations.\n'
            
            for demo in all_demos:
                demo = json.loads(demo)
                entities = get_ner(demo, TASK_DIC[dataset_name]['label_map'])
                template += f"Sentence: {' '.join(demo['sentence1'])}\nEntities:\n"

                ner_ = {'corporation': [], 'creative-work': [], 'group': [], 'location': [], 'person': [], 'product': []}
                for key, ent_type in type_list.items():
                    for ent in entities:
                        if ent[1] == key:
                            ner_[key].append(ent[0])
                for key, ent_list in ner_.items():
                    if len(ent_list) == 0:
                        ent_list = ['None']
                    template += f'{type_list[key]}: {", ".join(ent_list)}\n'
                template += '\n'
            if hint:
                template += 'End of demonstrations.\n\n'
        for i, sample in zip(indexes, all_samples):
            
            entities = get_ner(sample, TASK_DIC[dataset_name]['label_map'])
            inst_template = template
            if ret:
                if hint:
                    inst_template += 'Following are a few demonstrations.\n'
                for demo in sample['source']:
                    entities = get_ner(demo, TASK_DIC[dataset_name]['label_map'], sentence_key='sent1', label_key='s_label')
                    inst_template += f"Sentence: {' '.join(demo['sent1'])}\nEntities:\n"

                    ner_ = {'corporation': [], 'creative-work': [], 'group': [], 'location': [], 'person': [], 'product': []}
                    for key, ent_type in type_list.items():
                        for ent in entities:
                            if ent[1] == key:
                                ner_[key].append(ent[0])
                    for key, ent_list in ner_.items():
                        if len(ent_list) == 0:
                            ent_list = ['None']
                        inst_template += f'{type_list[key]}: {", ".join(ent_list)}\n'
                    inst_template += '\n'
                if hint:
                    inst_template += 'End of demonstrations.\n\n'

            #temp = "[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. " + f'{template}<</SYS>> Sentence: {sample["sentence1"]}\nLabel: ' + "[/INST]"
            if isT5:
                temp = f'{inst_template}\nSentence: {" ".join(sample["sentence1"])}\nEntities: '
            elif isllama:
                temp = "[INST] <<SYS>>\n"+ f'{inst_template}<</SYS>>\nPlease answer the following question.\nSentence: {" ".join(sample["sentence1"])}\nEntities:' + "[/INST]"
            else:
                temp = "[INST] "+ f'{inst_template}\n\nSentence: {" ".join(sample["sentence1"])}\nEntities: ' + "[/INST]"
            prompt_list.append(temp)
            labels.append(i)
    elif dataset_name == 'bc2gm_type':
        type_list = {'GENE': 'Gene'}
        if num_demo>0:
            if hint:
                template += 'Following are a few demonstrations.\n'
            for demo in all_demos:
                demo = json.loads(demo)
                entities = get_ner(demo, TASK_DIC[dataset_name]['label_map'])
                template += f"Sentence: {' '.join(demo['sentence1'])}\nEntities:\n"

                ner_ = {'GENE': []}
                for key, ent_type in type_list.items():
                    for ent in entities:
                        if ent[1] == key:
                            ner_[key].append(ent[0])
                for key, ent_list in ner_.items():
                    if len(ent_list) == 0:
                        ent_list = ['None']
                    template += f'{type_list[key]}: {", ".join(ent_list)}\n'
                template += '\n'
            if hint:
                template += 'End of demonstrations.\n\n'
        for i, sample in zip(indexes, all_samples):
            
            entities = get_ner(sample, TASK_DIC[dataset_name]['label_map'])
            #temp = "[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. " + f'{template}<</SYS>> Sentence: {sample["sentence1"]}\nLabel: ' + "[/INST]"
            if isT5:
                temp = f'{template}\nSentence: {" ".join(sample["sentence1"])}\nEntities: '
            elif isllama:
                temp = "[INST] <<SYS>>\n"+ f'{template}<</SYS>>\nPlease answer the following question.\nSentence: {" ".join(sample["sentence1"])}\nEntities:' + "[/INST]"
            else:
                temp = "[INST] "+ f'{template}\n\nSentence: {" ".join(sample["sentence1"])}\nEntities: ' + "[/INST]"
            prompt_list.append(temp)
            labels.append(i)
    elif dataset_name in ['rest14', 'laptop14']:
        if num_demo>0:
            if hint:
                template += 'Following are a few demonstrations.\n'
            for demo in all_demos:
                demo = json.loads(demo)
                template += f"Sentence: {demo['sentence1']}\nLabel:\n"

                if len(demo['label']) == 0:
                    template += 'None\n'
                else:
                    for i, asp_term in enumerate(demo['label']):
                        template += f"Aspect Term: {asp_term['term']}, Sentiment: {asp_term['polarity']}"
                        if i != len(demo['label']):
                            template+= "\n"
                template+="\n"
            if hint:
                template += '\nEnd of demonstrations.\n'

        for i, sample in zip(indexes, all_samples):
            
            inst_template = template
            if ret:
                if hint:
                    inst_template += 'Following are a few demonstrations.\n'
                for demo in sample['source']:
                    inst_template += f"Sentence: {demo['sent1']}\nLabel:\n"

                    if len(demo['s_label']) == 0:
                        inst_template += 'None\n'
                    else:
                        for i, asp_term in enumerate(demo['s_label']):
                            inst_template += f"Aspect Term: {asp_term['term']}, Sentiment: {asp_term['polarity']}"
                            if i != len(demo['s_label']):
                                inst_template+= "\n"
                    inst_template+="\n"
                if hint:
                    inst_template += 'End of demonstrations.\n\n'
                    
            if isT5:
                temp = f'{inst_template}\nSentence: {sample["sentence1"]}\nLabel:'
            elif isllama:
                temp = "[INST] <<SYS>>\n"+ f'{inst_template}<</SYS>>\nPlease answer the following question.\nSentence: {sample["sentence1"]}\nLabel:' + "[/INST]"
            else:
                temp = "[INST]"+ f'{inst_template}\nSentence: {sample["sentence1"]}\nLabel:' + "[/INST]"
            prompt_list.append(temp)
            labels.append(i)
    elif dataset_name in ['rest15', 'rest16']:
        if num_demo>0:
            if hint:
                template += 'Following are a few demonstrations.\n'
            for demo in all_demos:
                demo = json.loads(demo)
                template += f"Sentence: {demo['sentence1']}\nLabel:\n"
                if len(demo['label']) == 0 or all(d.get('target', None) == 'NULL' for d in demo['label']):
                    template += 'None\n'
                else:
                    for i, asp_term in enumerate(demo['label']):
                        if asp_term['target'] == 'NULL':
                            continue
                        template += f"Aspect Term: {asp_term['target']}, Sentiment: {asp_term['polarity']}"
                        if i != len(demo['label']):
                            template+= "\n"
                template+="\n"
            if hint:
                template += '\nEnd of demonstrations.\n'
        for i, sample in zip(indexes, all_samples):
            
            inst_template = template
            if ret:
                if hint:
                    inst_template += 'Following are a few demonstrations.\n'
                for demo in sample['source']:
                    inst_template += f"Sentence: {demo['sent1']}\nLabel:\n"
                    if len(demo['s_label']) == 0 or all(d.get('target', None) == 'NULL' for d in demo['s_label']):
                        inst_template += 'None\n'
                    else:
                        for i, asp_term in enumerate(demo['s_label']):
                            if asp_term['target'] == 'NULL':
                                continue
                            inst_template += f"Aspect Term: {asp_term['target']}, Sentiment: {asp_term['polarity']}"
                            if i != len(demo['s_label']):
                                inst_template+= "\n"
                    inst_template+="\n"
                if hint:
                    inst_template += 'End of demonstrations.\n\n'
            if isT5:
                temp = f'{inst_template}\nSentence: {sample["sentence1"]}\nLabel:'
            elif isllama:
                temp = "[INST] <<SYS>>\n"+ f'{inst_template}<</SYS>>\nPlease answer the following question.\nSentence: {sample["sentence1"]}\nLabel:' + "[/INST]"
            else:
                temp = "[INST]"+ f'{inst_template}\nSentence: {sample["sentence1"]}\nLabel:' + "[/INST]"
            prompt_list.append(temp)
            labels.append(i)
    elif dataset_name in ['super_glue_cb', 'glue_rte', 'sick_nli', 'wnli']:
        if num_demo>0:
            if hint:
                template += 'Following are a few demonstrations.\n'
            for demo in all_demos:
                demo = json.loads(demo)
                template += f"Premise: {demo['sentence1']}\nHypothesis: {demo['sentence2']}\nLabel: {TASK_DIC[dataset_name]['label_map'][demo['label']]}\n\n"
            if hint:
                template += 'End of demonstrations.\n\n'
        elif num_demo <0:
            if hint:
                template += 'Following are a few demonstrations.\n'
            for demo in all_demos:
                demo = json.loads(demo)
                flipped_label = flip_label(demo['label'], dataset_name)
                template += f"Premise: {demo['sentence1']}\nHypothesis: {demo['sentence2']}\nLabel: {TASK_DIC[dataset_name]['label_map'][flipped_label]}\n\n"
            if hint:
                template += 'End of demonstrations.\n\n'
        for sample in all_samples:
            inst_template = template
            if ret:
                if hint:
                    inst_template += 'Following are a few demonstrations.\n'
                for demo in sample['source']:
                    inst_template+= f"Premise: {demo['sent1']}\nHypothesis: {demo['sent2']}\nLabel: {TASK_DIC[dataset_name]['label_map'][demo['s_label']]}\n\n"
                if hint:
                    inst_template += 'End of demonstrations.\n\n'
            #temp = "[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. " + f'{inst_template}<</SYS>> Sentence: {sample["sentence1"]}\nLabel: ' + "[/INST]"
            if isT5:
                temp = f'{inst_template}\nPremise: {sample["sentence1"]}\nHypothesis: {sample["sentence2"]}\nLabel: '
            elif isllama:
                temp = "[INST] <<SYS>>\n"+ f'{inst_template}<</SYS>>\nPlease answer the following question.\nPremise: {sample["sentence1"]}\nHypothesis: {sample["sentence2"]}\nLabel:' + "[/INST]"
            else:
                temp = "[INST]"+ f'{inst_template}\n\nPremise: {sample["sentence1"]}\nHypothesis: {sample["sentence2"]}\nLabel: ' + "[/INST]"
            prompt_list.append(temp)
            if num_demo >= 0:
                labels.append(sample['label'])
            else:
                labels.append(flip_label(sample['label'], dataset_name))
    elif dataset_name in ['glue_mrpc', 'medical_questions_pairs']:
        if num_demo>0:
            if hint:
                template += 'Following are a few demonstrations.\n'
            for demo in all_demos:
                demo = json.loads(demo)
                template += f"Sentence 1: {demo['sentence1']}\nSentence 2: {demo['sentence2']}\nLabel: {TASK_DIC[dataset_name]['label_map'][demo['label']]}\n\n"
            if hint:
                template += 'End of demonstrations.\n\n'
        elif num_demo<0:
            if hint:
                template += 'Following are a few demonstrations.\n'
            for demo in all_demos:
                demo = json.loads(demo)
                flipped_label = flip_label(demo['label'], dataset_name)
                template += f"Sentence 1: {demo['sentence1']}\nSentence 2: {demo['sentence2']}\nLabel: {TASK_DIC[dataset_name]['label_map'][flipped_label]}\n\n"
            if hint:
                template += 'End of demonstrations.\n\n'
        for sample in all_samples:
            if isinstance(sample, str):
                sample = json.loads(sample)
            inst_template = template
            if ret:
                if hint:
                    inst_template += 'Following are a few demonstrations.\n'
                for demo in sample['source']:
                    inst_template+= f"Sentence 1: {demo['sent1']}\nSentence 2: {demo['sent2']}\nLabel: {TASK_DIC[dataset_name]['label_map'][demo['s_label']]}\n\n"
                if hint:
                    inst_template += 'End of demonstrations.\n\n'

            #temp = "[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. " + f'{inst_template}<</SYS>> Sentence: {sample["sentence1"]}\nLabel: ' + "[/INST]"
            if isT5:
                temp = f'{inst_template}\nSentence 1: {sample["sentence1"]}\nSentence 2: {sample["sentence2"]}\nLabel: '
            elif isllama:
                temp = "[INST] <<SYS>>\n"+ f'{inst_template}<</SYS>>\nPlease answer the following question.\nSentence 1: {sample["sentence1"]}\nSentence 2: {sample["sentence2"]}\nLabel:' + "[/INST]"
            else:
                temp = "[INST]"+ f'{inst_template}\n\nSentence 1: {sample["sentence1"]}\nSentence2: {sample["sentence2"]}\nLabel: ' + "[/INST]"
            prompt_list.append(temp)
            if num_demo >= 0:
                labels.append(sample['label'])
            else:
                labels.append(flip_label(sample['label'], dataset_name))
    elif dataset_name in ['common_gen', 'roc_story_ending', 'wikiauto',  'reddit', 'samsum', 'roc_story']:
        if num_demo>0:
            template = create_generation_demo(all_demos, template, TASK_DIC[dataset_name]['label_map'][0], TASK_DIC[dataset_name]['label_map'][1])
        
        raw_prompt_list, labels = create_generation(all_samples, TASK_DIC[dataset_name]['label_map'][0], TASK_DIC[dataset_name]['label_map'][1])
        for pp in raw_prompt_list:
            inst_template = template
            if hint:
                start_pos = inst_template.index(TASK_DIC[dataset_name]['label_map'][0])
                inst_template = inst_template[:start_pos] + 'Following are a few demonstrations.\n' + inst_template[start_pos:]
            if hint:
                inst_template += 'End of demonstrations.\n\n'
            if isllama and hint:
                temp = "[INST] <<SYS>>\n"+ f'{inst_template}<</SYS>>\nPlease answer the following question.\n{pp}' + "[/INST]"
            elif isllama:
                temp = "[INST] <<SYS>>\n"+ f'{inst_template}<</SYS>>\n{pp}' + "[/INST]"
            else:
                temp = "[INST]"+ f'{inst_template}\n\n{pp}' + "[/INST]"
            prompt_list.append(temp)
    else:
        if num_demo>0:
            if hint:
                template += 'Following are a few demonstrations.\n'
            for demo in all_demos:
                demo = json.loads(demo)
                template += f"Sentence: {demo['sentence1']}\nLabel: {TASK_DIC[dataset_name]['label_map'][demo['label']]}\n\n"
            if hint:
                template += 'End of demonstrations.\n\n'
        elif num_demo <0:
            if hint:
                template += 'Following are a few demonstrations.\n'
            for demo in all_demos:
                demo = json.loads(demo)
                flipped_label = flip_label(demo['label'], dataset_name)
                template += f"Sentence: {demo['sentence1']}\nLabel: {TASK_DIC[dataset_name]['label_map'][flipped_label]}\n\n"
            if hint:
                template += 'End of demonstrations.\n\n'
        for sample in all_samples:
            
            inst_template = template
            if ret:
                if hint:
                    inst_template += 'Following are a few demonstrations.\n'
                for demo in sample['source']:
                    inst_template += f"Sentence: {demo['sent1']}\nLabel: {TASK_DIC[dataset_name]['label_map'][demo['s_label']]}\n\n"
                if hint:
                    inst_template += 'End of demonstrations.\n\n'

            #temp = "[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. " + f'{template}<</SYS>> Sentence: {sample["sentence1"]}\nLabel: ' + "[/INST]"
            if isT5:
                temp = f'{inst_template}\nSentence: {sample["sentence1"]}\nLabel: '
            elif isllama:
                temp = "[INST] <<SYS>>\n"+ f'{inst_template}<</SYS>>\nPlease answer the following question.\nSentence: {sample["sentence1"]}\nLabel:' + "[/INST]"
                #temp = "[INST]\n"+ f'{inst_template} \nGive you answer with the label only.\n\nSentence: {sample["sentence1"]}\nLabel:' + "[/INST]"
            else:
                temp = "[INST]"+ f'{inst_template}\n\nSentence: {sample["sentence1"]}\nLabel: ' + "[/INST]"
            prompt_list.append(temp)
            if num_demo >= 0:
                labels.append(sample['label'])
            else:
                labels.append(flip_label(sample['label'], dataset_name))
    
    return prompt_list, labels

def get_prompts_chat(all_demos, all_samples, dataset_name, num_demo=5, hint=False, indexes=None, no_format=True, ret=False):
    if no_format:
        template = f"{TASK_DIC[dataset_name]['task_desc']}\n"
    else:
        template = f"{TASK_DIC[dataset_name]['task_desc']} {TASK_DIC[dataset_name]['format_desc']}\n"
    if ret:
        num_demo = 0

    prompt_list = []
    sys_message = ""
    labels = []
    demos = []
    if dataset_name in ['conll03', 'wnut17', 'bc2gm']:
        if num_demo>0:
            if hint:
                template += 'Following are a few demonstrations.\n'
            for demo in all_demos:
                demo = json.loads(demo)
                entities = get_ner(demo, TASK_DIC[dataset_name]['label_map'])
                template += f"Sentence: {' '.join(demo['sentence1'])}\nEntities: "
                if len(entities) != 0:
                    for i, ent in enumerate(entities):
                        template+= ent[0]
                        if i != len(entities)-1:
                            template+=", "
                    template+="\n"
                else:
                    template+= "None\n"
            if hint:
                template += '\nEnd of demonstrations.\n'

        sys_message = template
        
        for i, sample in zip(indexes, all_samples):
            sample = json.loads(sample)
            entities = get_ner(sample, TASK_DIC[dataset_name]['label_map'])
            #temp = "[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. " + f'{template}<</SYS>> Sentence: {sample["sentence1"]}\nLabel: ' + "[/INST]"
            temp = f'Sentence: {" ".join(sample["sentence1"])}\nEntities: '
            prompt_list.append(temp)
            labels.append(i)
    elif dataset_name == 'conll03_type':

        type_list = {'PER': 'Person', 'LOC': 'Location', 'ORG': 'Organization', 'MISC': 'Miscellaneous'}
        if num_demo>0:
            if hint:
                template += 'Following are a few demonstrations.\n'
            
            for demo in all_demos:
                demo = json.loads(demo)
                entities = get_ner(demo, TASK_DIC[dataset_name]['label_map'])
                template += f"Sentence: {' '.join(demo['sentence1'])}\nEntities:\n"

                ner_ = {'PER': [], 'LOC': [], 'ORG': [], 'MISC': []}
                for key, ent_type in type_list.items():
                    for ent in entities:
                        if ent[1] == key:
                            ner_[key].append(ent[0])
                for key, ent_list in ner_.items():
                    if len(ent_list) == 0:
                        ent_list = ['None']
                    template += f'{type_list[key]}: {", ".join(ent_list)}\n'
                template += '\n'
            if hint:
                template += 'End of demonstrations.\n\n'
        
        for i, sample in zip(indexes, all_samples):
            sample = json.loads(sample)
            entities = get_ner(sample, TASK_DIC[dataset_name]['label_map'])
            inst_template = template
            if ret:
                if hint:
                    inst_template += 'Following are a few demonstrations.\n'
                for demo in sample['source']:
                    demo = json.loads(demo)
                    entities = get_ner(demo, TASK_DIC[dataset_name]['label_map'])
                    inst_template += f"Sentence: {' '.join(demo['sentence1'])}\nEntities:\n"

                    ner_ = {'PER': [], 'LOC': [], 'ORG': [], 'MISC': []}
                    for key, ent_type in type_list.items():
                        for ent in entities:
                            if ent[1] == key:
                                ner_[key].append(ent[0])
                    for key, ent_list in ner_.items():
                        if len(ent_list) == 0:
                            ent_list = ['None']
                        inst_template += f'{type_list[key]}: {", ".join(ent_list)}\n'
                    inst_template += '\n'
                if hint:
                    inst_template += 'End of demonstrations.\n\n'
            #temp = "[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. " + f'{template}<</SYS>> Sentence: {sample["sentence1"]}\nLabel: ' + "[/INST]"
            temp = f'Sentence: {" ".join(sample["sentence1"])}\nEntities: '
            demos.append(inst_template)
            prompt_list.append(temp)
            labels.append(i)
    elif dataset_name == 'wnut17_type':
        type_list = {'corporation': 'Corporation', 'creative-work': 'Creative-work', 'group': 'Group', 'location': 'Location', 'person': 'Person', 'product': 'Product'}
        if num_demo>0:
            if hint:
                template += 'Following are a few demonstrations.\n'
            
            for demo in all_demos:
                demo = json.loads(demo)
                entities = get_ner(demo, TASK_DIC[dataset_name]['label_map'])
                template += f"Sentence: {' '.join(demo['sentence1'])}\nEntities:\n"

                ner_ = {'corporation': [], 'creative-work': [], 'group': [], 'location': [], 'person': [], 'product': []}
                for key, ent_type in type_list.items():
                    for ent in entities:
                        if ent[1] == key:
                            ner_[key].append(ent[0])
                for key, ent_list in ner_.items():
                    if len(ent_list) == 0:
                        ent_list = ['None']
                    template += f'{type_list[key]}: {", ".join(ent_list)}\n'
                template += '\n'
            if hint:
                template += 'End of demonstrations.\n\n'
        for i, sample in zip(indexes, all_samples):
            sample = json.loads(sample)
            entities = get_ner(sample, TASK_DIC[dataset_name]['label_map'])
            inst_template = template
            if ret:
                if hint:
                    inst_template += 'Following are a few demonstrations.\n'
                for demo in sample['source']:
                    demo = json.loads(demo)
                    entities = get_ner(demo, TASK_DIC[dataset_name]['label_map'])
                    inst_template += f"Sentence: {' '.join(demo['sentence1'])}\nEntities:\n"

                    ner_ = {'corporation': [], 'creative-work': [], 'group': [], 'location': [], 'person': [], 'product': []}
                    for key, ent_type in type_list.items():
                        for ent in entities:
                            if ent[1] == key:
                                ner_[key].append(ent[0])
                    for key, ent_list in ner_.items():
                        if len(ent_list) == 0:
                            ent_list = ['None']
                        inst_template += f'{type_list[key]}: {", ".join(ent_list)}\n'
                    inst_template += '\n'
                if hint:
                    inst_template += 'End of demonstrations.\n\n'

            #temp = "[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. " + f'{template}<</SYS>> Sentence: {sample["sentence1"]}\nLabel: ' + "[/INST]"
            temp = f'Sentence: {" ".join(sample["sentence1"])}\nEntities: '
            prompt_list.append(temp)
            labels.append(i)
            demos.append(inst_template)
    elif dataset_name == 'bc2gm_type':
        type_list = {'GENE': 'Gene'}
        if num_demo>0:
            if hint:
                template += 'Following are a few demonstrations.\n'
            for demo in all_demos:
                demo = json.loads(demo)
                entities = get_ner(demo, TASK_DIC[dataset_name]['label_map'])
                template += f"Sentence: {' '.join(demo['sentence1'])}\nEntities:\n"

                ner_ = {'GENE': []}
                for key, ent_type in type_list.items():
                    for ent in entities:
                        if ent[1] == key:
                            ner_[key].append(ent[0])
                for key, ent_list in ner_.items():
                    if len(ent_list) == 0:
                        ent_list = ['None']
                    template += f'{type_list[key]}: {", ".join(ent_list)}\n'
                template += '\n'
            if hint:
                template += '\nEnd of demonstrations.\n'
        sys_message = template

        for i, sample in zip(indexes, all_samples):
            sample = json.loads(sample)
            entities = get_ner(sample, TASK_DIC[dataset_name]['label_map'])
            #temp = "[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. " + f'{template}<</SYS>> Sentence: {sample["sentence1"]}\nLabel: ' + "[/INST]"
            temp = f'Sentence: {" ".join(sample["sentence1"])}\nEntities: '
            prompt_list.append(temp)
            labels.append(i)
    elif dataset_name in ['rest14', 'laptop14']:
        if num_demo>0:
            if hint:
                template += 'Following are a few demonstrations.\n'
            for demo in all_demos:
                demo = json.loads(demo)
                template += f"Sentence: {demo['sentence1']}\nLabel:\n"

                if len(demo['label']) == 0:
                    template += 'None\n'
                else:
                    for i, asp_term in enumerate(demo['label']):
                        template += f"Aspect Term: {asp_term['term']}, Sentiment: {asp_term['polarity']}"
                        if i != len(demo['label']):
                            template+= "\n"
                template+="\n"
            if hint:
                template += '\nEnd of demonstrations.\n'

        sys_message = template
        for i, sample in zip(indexes, all_samples):
            sample = json.loads(sample)
            #temp = "[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. " + f'{template}<</SYS>> Sentence: {sample["sentence1"]}\nLabel: ' + "[/INST]"
            temp = f'Sentence: {sample["sentence1"]}\nLabel:'
            prompt_list.append(temp)
            labels.append(i)
    elif dataset_name in ['rest15', 'rest16']:
        if num_demo>0:
            if hint:
                template += 'Following are a few demonstrations.\n'
            for demo in all_demos:
                demo = json.loads(demo)
                template += f"Sentence: {demo['sentence1']}\nLabel:\n"
                if len(demo['label']) == 0 or all(d.get('target', None) == 'NULL' for d in demo['label']):
                    template += 'None\n'
                else:
                    for i, asp_term in enumerate(demo['label']):
                        if asp_term['target'] == 'NULL':
                            continue
                        template += f"Aspect Term: {asp_term['target']}, Sentiment: {asp_term['polarity']}"
                        if i != len(demo['label']):
                            template+= "\n"
                template+="\n"
            if hint:
                template += '\nEnd of demonstrations.\n'

        sys_message = template
        for i, sample in zip(indexes, all_samples):
            sample = json.loads(sample)
            #if len(sample['label']) != 0 and all(d.get('target', None) == 'NULL' for d in sample['label']):
            #    print()
            #temp = "[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. " + f'{template}<</SYS>> Sentence: {sample["sentence1"]}\nLabel: ' + "[/INST]"
            temp = f'Sentence: {sample["sentence1"]}\nLabel:'
            prompt_list.append(temp)
            labels.append(i)
    elif dataset_name in ['super_glue_cb', 'glue_rte', 'sick_nli', 'wnli']:
        if num_demo>0:
            if hint:
                template += 'Following are a few demonstrations.\n'
            for demo in all_demos:
                demo = json.loads(demo)
                template += f"Premise: {demo['sentence1']}\nHypothesis: {demo['sentence2']}\nLabel: {TASK_DIC[dataset_name]['label_map'][demo['label']]}\n\n"
            if hint:
                template += '\nEnd of demonstrations.\n'
        elif num_demo<0:
            if hint:
                template += 'Following are a few demonstrations.\n'
            for demo in all_demos:
                demo = json.loads(demo)
                flipped_label = flip_label(demo['label'], dataset_name)
                template += f"Premise: {demo['sentence1']}\nHypothesis: {demo['sentence2']}\nLabel: {TASK_DIC[dataset_name]['label_map'][flipped_label]}\n\n"
            if hint:
                template += '\nEnd of demonstrations.\n'
        

        for sample in all_samples:
            sys_message = template
            sample = json.loads(sample)
            if ret:
                if hint:
                    sys_message += 'Following are a few demonstrations.\n'
                for demo in sample['source']:
                    sys_message+= f"Premise: {demo['sent1']}\nHypothesis: {demo['sent2']}\nLabel: {TASK_DIC[dataset_name]['label_map'][demo['s_label']]}\n\n"
                if hint:
                    sys_message += '\nEnd of demonstrations.\n'
            #temp = "[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. " + f'{template}<</SYS>> Sentence: {sample["sentence1"]}\nLabel: ' + "[/INST]"
            temp = f'Premise: {sample["sentence1"]}\nHypothesis: {sample["sentence2"]}\nLabel: '
            prompt_list.append(temp)
            labels.append(sample['label'])
            demos.append(sys_message)
    elif dataset_name in ['glue_mrpc', 'medical_questions_pairs']:
        if num_demo>0:
            if hint:
                template += 'Following are a few demonstrations.\n'
            for demo in all_demos:
                demo = json.loads(demo)
                template += f"Sentence 1: {demo['sentence1']}\nSentence 2: {demo['sentence2']}\nLabel: {TASK_DIC[dataset_name]['label_map'][demo['label']]}\n\n"
            if hint:
                template += '\nEnd of demonstrations.\n'
        elif num_demo<0:
            if hint:
                template += 'Following are a few demonstrations.\n'
            for demo in all_demos:
                demo = json.loads(demo)
                flipped_label = flip_label(demo['label'], dataset_name)
                template += f"Sentence 1: {demo['sentence1']}\nSentence 2: {demo['sentence2']}\nLabel: {TASK_DIC[dataset_name]['label_map'][flipped_label]}\n\n"
            if hint:
                template += '\nEnd of demonstrations.\n'
        for sample in all_samples:
            sys_message = template
            sample = json.loads(sample)
            if ret:
                if hint:
                    sys_message += 'Following are a few demonstrations.\n'
                for demo in sample['source']:
                    sys_message+= f"Sentence 1: {demo['sent1']}\nSentence 2: {demo['sent2']}\nLabel: {TASK_DIC[dataset_name]['label_map'][demo['s_label']]}\n\n"
                if hint:
                    sys_message += '\nEnd of demonstrations.\n'
            #temp = "[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. " + f'{template}<</SYS>> Sentence: {sample["sentence1"]}\nLabel: ' + "[/INST]"
            temp = f'Sentence 1: {sample["sentence1"]}\nSentence 2: {sample["sentence2"]}\nLabel: '
            prompt_list.append(temp)
            labels.append(sample['label'])
            demos.append(sys_message)
    elif dataset_name in ['common_gen', 'roc_story_ending', 'wikiauto',  'reddit', 'samsum', 'roc_story']:
        if num_demo>0:
            template = create_generation_demo(all_demos, template, TASK_DIC[dataset_name]['label_map'][0], TASK_DIC[dataset_name]['label_map'][1])
        sys_message = template
        prompt_list, labels = create_generation(all_samples, TASK_DIC[dataset_name]['label_map'][0], TASK_DIC[dataset_name]['label_map'][1])
    else:
        if num_demo>0:
            if hint:
                template += 'Following are a few demonstrations.\n'
            for demo in all_demos:
                demo = json.loads(demo)
                template += f"Sentence: {demo['sentence1']}\nLabel: {TASK_DIC[dataset_name]['label_map'][demo['label']]}\n\n"
            if hint:
                template += 'End of demonstrations.\n\n'
        elif num_demo <0:
            if hint:
                template += 'Following are a few demonstrations.\n'
            for demo in all_demos:
                demo = json.loads(demo)
                flipped_label = flip_label(demo['label'], dataset_name)
                template += f"Sentence: {demo['sentence1']}\nLabel: {TASK_DIC[dataset_name]['label_map'][flipped_label]}\n\n"
            if hint:
                template += 'End of demonstrations.\n\n'
        
        for sample in all_samples:
            sample = json.loads(sample)
            sys_message = template
            if ret:
                if hint:
                    sys_message += 'Following are a few demonstrations.\n'
                for demo in sample['source']:
                    sys_message+= f"Sentence: {demo['sent1']}\nLabel: {TASK_DIC[dataset_name]['label_map'][demo['s_label']]}\n\n"
                if hint:
                    sys_message += '\nEnd of demonstrations.\n'
            #temp = "[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. " + f'{template}<</SYS>> Sentence: {sample["sentence1"]}\nLabel: ' + "[/INST]"
            temp = f'Sentence: {sample["sentence1"]}\nLabel: '
            prompt_list.append(temp)
            labels.append(sample['label'])
            demos.append(sys_message)
    
    return prompt_list, labels, sys_message, demos


def get_ner(sample, label_map, sentence_key='sentence1', label_key='label'):
    entities = []
    temp = []
    temp_type = ''
    ent_flag = False
    for tok, lab in zip(sample[sentence_key], sample[label_key]):  
        if label_map[lab].startswith('B'):
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

def flip_label(label: int, dataset):
    # 2 class
    if dataset in ['glue_rte', 'wnli', 'glue_sst2', 'glue_mrpc', 'tweet_hate', 'medical_questions_pairs', 'hs18', 'sst2_test']:
        if label == 0:
            out_label = 1
        elif label == 1:
            out_label = 0
    # 3 class with neutral
    if dataset in ['tweet_stance_atheism', 'tweet_stance_feminist']:
        # neutral
        if label == 0:
            out_label = random.choice([1,2])
        elif label == 1:
            out_label = 2
        elif label == 2:
            out_label = 1
    # Multi-class
    if dataset == 'ag_new':
        label_space = list(range(4))
        label_space.remove(label)
        out_label = random.choice(label_space)
    if dataset == 'trec_':
        label_space = list(range(6))
        label_space.remove(label)
        out_label = random.choice(label_space)
    return out_label




if __name__ == "__main__":
    with open(f'./data/5/wnli/k-5-seed-100/train.jsonl') as f:
        all_demos = f.readlines()
    #with open(f'./data/5/wnli/k-5-seed-100/test.jsonl') as f:
    #    all_samples = f.readlines()
    with open(f'./data/retrieval/wnli/test.jsonl') as f:
        all_samples = f.readlines()
    #entities = get_prompts(all_demos, all_samples, 'rest14', isT5=False,isllama=True, num_demo=5, hint=False, indexes=list(range(len(all_samples))))
    prompt_getter = PromptGetter(all_demos, all_samples, dataset_name='wnli', num_demo=-5, hint=False, isT5=False, isllama=False, isGPT=True, indexes=None, no_format=True, ret=False) # 5, True, False, True, False, ret=True)
    #entities = get_prompts_chat(all_demos, all_samples, 'rest14', num_demo=5, hint=False, indexes=list(range(len(all_samples))))
    print()


"""
conll
'0': O
'1': B-PER
'2': I-PER
'3': B-ORG
'4': I-ORG
'5': B-LOC
'6': I-LOC
'7': B-MISC
'8': I-MISC

wnut
'0': O
'1': B-corporation
'2': I-corporation
'3': B-creative-work
'4': I-creative-work
'5': B-group
'6': I-group
'7': B-location
'8': I-location
'9': B-person
'10': I-person
'11': B-product
'12': I-product

"""
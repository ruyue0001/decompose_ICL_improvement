import os
import json
import math
import random
import logging
import argparse
from typing import Union

import torch
from transformers import AutoModel, AutoTokenizer, set_seed
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
#from dppy.finite_dpps import FiniteDPP
import faiss
#from langchain_community.retrievers import BM25Retriever

from generate_icl import task2keys
from prepare_prompt import TASK_DIC


dataset_list = [("glue", "sst2"), ("glue", "rte"), ("tweet_eval", "hate"), ("huggingface", "trec"), ("huggingface", "ag_news"), ("glue", "mrpc"), ("glue", "wnli"), ("huggingface", "medical_questions_pairs"), ("huggingface", "hate_speech18")]


class Scorer():
    """
    A class for embedding sentences, calculating similarities, and retriving sentences by SimCSE.
    """
    def __init__(self, model_name_or_path: str,
                use_transformer=False,
                device: str = None,
                num_cells: int = 100,
                num_cells_in_search: int = 10,
                pooler = None):
        
        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        if use_transformer:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            self.model = AutoModel.from_pretrained(model_name_or_path)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.index = None
        self.index_cheat = [None, None, None,None,None,None]
        self.is_faiss_index = False
        self.num_cells = num_cells
        self.num_cells_in_search = num_cells_in_search

        if pooler is not None:
            self.pooler = pooler
        elif "unsup" in model_name_or_path:
            self.logger.info("Use `cls_before_pooler` for unsupervised models. If you want to use other pooling policy, specify `pooler` argument.")
            self.pooler = "cls_before_pooler"
        else:
            self.pooler = "cls"

    def encode(self, sentence: Union[str, list[str]], 
                device: str = None, 
                return_numpy: bool = False,
                normalize_to_unit: bool = True,
                keepdim: bool = False,
                batch_size: int = 64,
                max_length: int = 128,
                is_source: bool = False) -> Union[np.ndarray, torch.Tensor]:

        target_device = self.device if device is None else device
        self.model = self.model.to(target_device)
        
        single_sentence = False
        if isinstance(sentence, str):
            sentence = [sentence]
            single_sentence = True

        embedding_list = [] 
        with torch.no_grad():
            total_batch = len(sentence) // batch_size + (1 if len(sentence) % batch_size > 0 else 0)
            if is_source:
                for batch_id in tqdm(range(total_batch)):
                    inputs = self.tokenizer(
                        sentence[batch_id*batch_size:(batch_id+1)*batch_size], 
                        padding=True, 
                        truncation=True, 
                        max_length=max_length, 
                        return_tensors="pt"
                    )
                    inputs = {k: v.to(target_device) for k, v in inputs.items()}
                    outputs = self.model(**inputs, return_dict=True)
                    if self.pooler == "cls":
                        embeddings = outputs.pooler_output
                    elif self.pooler == "cls_before_pooler":
                        embeddings = outputs.last_hidden_state[:, 0]
                    else:
                        raise NotImplementedError
                    if normalize_to_unit:
                        embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
                    embedding_list.append(embeddings.cpu())
            else:
                for batch_id in range(total_batch):
                    inputs = self.tokenizer(
                        sentence[batch_id*batch_size:(batch_id+1)*batch_size], 
                        padding=True, 
                        truncation=True, 
                        max_length=max_length, 
                        return_tensors="pt"
                    )
                    inputs = {k: v.to(target_device) for k, v in inputs.items()}
                    outputs = self.model(**inputs, return_dict=True)
                    if self.pooler == "cls":
                        embeddings = outputs.pooler_output
                    elif self.pooler == "cls_before_pooler":
                        embeddings = outputs.last_hidden_state[:, 0]
                    else:
                        raise NotImplementedError
                    if normalize_to_unit:
                        embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
                    embedding_list.append(embeddings.cpu())
        embeddings = torch.cat(embedding_list, 0)
        
        if single_sentence and not keepdim:
            embeddings = embeddings[0]
        
        if return_numpy and not isinstance(embeddings, np.ndarray):
            return embeddings.numpy()
        return embeddings
    
    def build_index(self, sentences_or_file_path: Union[str, list[str]], 
                        use_faiss: bool = False,
                        faiss_fast: bool = False,
                        device: str = None,
                        batch_size: int = 64):

        # if the input sentence is a string, we assume it's the path of file that stores various sentences
        if isinstance(sentences_or_file_path, str):
            sentences = []
            with open(sentences_or_file_path, "r") as f:
                logging.info("Loading sentences from %s ..." % (sentences_or_file_path))
                for line in tqdm(f):
                    sentences.append(line.rstrip())
            sentences_or_file_path = sentences
        
        self.logger.info("Encoding embeddings for sentences...")
        embeddings = self.encode(sentences_or_file_path, device=device, batch_size=batch_size, normalize_to_unit=True, return_numpy=True, is_source=True)

        self.logger.info("Building index...")
        self.index = {"sentences": sentences_or_file_path}

        if use_faiss:
            #quantizer = faiss.IndexFlatIP(embeddings.shape[1])  
            #if faiss_fast:
            #    index = faiss.IndexIVFFlat(quantizer, embeddings.shape[1], min(self.num_cells, len(sentences_or_file_path)), faiss.METRIC_INNER_PRODUCT) 
            #else:
            #    index = quantizer

            index = faiss.IndexIDMap(faiss.IndexFlatIP(embeddings.shape[1]))

            '''if (self.device == "cuda" and device != "cpu") or device == "cuda":
                if hasattr(faiss, "StandardGpuResources"):
                    print("Use GPU-version faiss")
                    #res = faiss.StandardGpuResources()
                    #res.setTempMemory(20 * 1024 * 1024 * 1024)
                    index = faiss.index_cpu_to_gpu(0, index)
                else:
                    print("Use CPU-version faiss")
            else: 
                print("Use CPU-version faiss")'''

            if faiss_fast:            
                index.train(embeddings.astype(np.float32))
            index.add_with_ids(embeddings.astype(np.float32), np.arange(len(sentences_or_file_path)))
            index.nprobe = min(self.num_cells_in_search, len(sentences_or_file_path))
            self.is_faiss_index = True
        else:
            index = embeddings
            self.is_faiss_index = False
        self.index["index"] = index
        self.logger.info("Finished")
    
    def build_index_cheat(self, sentences_or_file_path: Union[str, list[str]], 
                        use_faiss: bool = False,
                        faiss_fast: bool = False,
                        device: str = None,
                        batch_size: int = 64,
                        polarity:int=0):

        # if the input sentence is a string, we assume it's the path of file that stores various sentences
        if isinstance(sentences_or_file_path, str):
            sentences = []
            with open(sentences_or_file_path, "r") as f:
                logging.info("Loading sentences from %s ..." % (sentences_or_file_path))
                for line in tqdm(f):
                    sentences.append(line.rstrip())
            sentences_or_file_path = sentences
        
        self.logger.info("Building index...")
        self.index_cheat[polarity] = {"sentences": sentences_or_file_path}

        if 'ret' in args.strategy:
            self.logger.info("Encoding embeddings for sentences...")
            embeddings = self.encode(sentences_or_file_path, device=device, batch_size=batch_size, normalize_to_unit=True, return_numpy=True, is_source=True)

            if use_faiss:
                #quantizer = faiss.IndexFlatIP(embeddings.shape[1])  
                #if faiss_fast:
                #    index = faiss.IndexIVFFlat(quantizer, embeddings.shape[1], min(self.num_cells, len(sentences_or_file_path)), faiss.METRIC_INNER_PRODUCT) 
                #else:
                #    index = quantizer

                index = faiss.IndexIDMap(faiss.IndexFlatIP(embeddings.shape[1]))

                '''if (self.device == "cuda" and device != "cpu") or device == "cuda":
                    if hasattr(faiss, "StandardGpuResources"):
                        print("Use GPU-version faiss")
                        #res = faiss.StandardGpuResources()
                        #res.setTempMemory(20 * 1024 * 1024 * 1024)
                        index = faiss.index_cpu_to_gpu(0, index)
                    else:
                        print("Use CPU-version faiss")
                else: 
                    print("Use CPU-version faiss")'''

                if faiss_fast:            
                    index.train(embeddings.astype(np.float32))
                index.add_with_ids(embeddings.astype(np.float32), np.arange(len(sentences_or_file_path)))
                index.nprobe = min(self.num_cells_in_search, len(sentences_or_file_path))
                self.is_faiss_index = True
            else:
                index = embeddings
                self.is_faiss_index = False
            self.index_cheat[polarity]["index"] = index

        self.logger.info("Finished")

    def build_index_bm25(self, sentences, top_k):
        self.index = BM25Retriever.from_texts(sentences, k=top_k)

    def similarity(self, queries: Union[str, list[str]], 
                    keys: Union[str, list[str], np.ndarray], 
                    device: str = None) -> Union[float, np.ndarray]:
        
        query_vecs = self.encode(queries, device=device, return_numpy=True) # suppose N queries
        
        if not isinstance(keys, np.ndarray):
            key_vecs = self.encode(keys, device=device, return_numpy=True) # suppose M keys
        else:
            key_vecs = keys

        # check whether N == 1 or M == 1
        single_query, single_key = len(query_vecs.shape) == 1, len(key_vecs.shape) == 1 
        if single_query:
            query_vecs = query_vecs.reshape(1, -1)
        if single_key:
            key_vecs = key_vecs.reshape(1, -1)
        
        # returns an N*M similarity array
        similarities = cosine_similarity(query_vecs, key_vecs)
        
        if single_query:
            similarities = similarities[0]
            if single_key:
                similarities = float(similarities[0])
        
        return similarities

    def search(self, queries: Union[str, list[str]], 
                device: str = None, 
                threshold: float = 0.6,
                top_k: int = 5) -> Union[list[tuple[str, float]], list[list[tuple[str, float]]]]:
        

        if isinstance(queries, list):
            combined_results = []
            for query in queries:
                results = self.search(query, device, threshold, top_k)
                combined_results.append(results)
            return combined_results
        
        similarities = self.similarity(queries, self.index["index"]).tolist()
        id_and_score = []
        for i, s in enumerate(similarities):
            if s >= threshold:
                id_and_score.append((i, s))

        if top_k > 0:
            id_and_score = sorted(id_and_score, key=lambda x: x[1], reverse=True)[:top_k]
        elif top_k < 0:
            id_and_score = sorted(id_and_score, key=lambda x: x[1], reverse=True)[top_k:]
        results = [(self.index["sentences"][idx], idx, score) for idx, score in id_and_score]
        return results
    
    def search_cheat(self, queries,polarity,device=None, threshold=0, top_k=5):
        similarities = self.similarity(queries, self.index_cheat[polarity]["index"]).tolist()
        id_and_score = []
        for i, s in enumerate(similarities):
            if s >= threshold:
                id_and_score.append((i, s))

        if top_k > 0:
            id_and_score = sorted(id_and_score, key=lambda x: x[1], reverse=True)[:top_k]
        elif top_k < 0:
            id_and_score = sorted(id_and_score, key=lambda x: x[1], reverse=True)[top_k:]
        results = [(self.index_cheat[polarity]["sentences"][idx], idx, score) for idx, score in id_and_score]
        return results

    def search_random(self, queries: Union[str, list[str]], 
                source_len: int,
                device: str = None, 
                top_k: int = 5,
                ) -> Union[list[tuple[str, float]], list[list[tuple[str, float]]]]:
        

        if isinstance(queries, list):
            combined_results = []
            for query in queries:
                results = self.search_random(query, source_len, device, top_k)
                combined_results.append(results)
            return combined_results
        
        '''similarities = self.similarity(queries, self.index["index"]).tolist()
        id_and_score = []
        for i, s in enumerate(similarities):
            if s >= threshold:
                id_and_score.append((i, s))'''

        results = random.sample(list(range(source_len)), top_k)
        results = [[0, e] for e in results]
        return results
    
    def search_cheat_rand(self, queries, source, polarity, top_k):
        results = random.sample(source[polarity], top_k)
        results = [[0, e] for e in results]
        return results
    
    def search_reverse_cheat_rand(self, queries, source, polarity, top_k):
        label_space = 6 - self.index_cheat.count(None)
        temp = []
        for i in range(label_space):
            if i == polarity:
                continue
            temp.extend(source[i])
        results = random.sample(temp, top_k)
        results = [[0, e] for e in results]
        return results

    def search_reverse_cheat_ret(self, queries,polarity,device=None, threshold=0, top_k=5):
        label_space = 6 - self.index_cheat.count(None)

        id_and_score = []

        for polar_label in range(label_space):
            if polar_label == polarity:
                continue
            similarities = self.similarity(queries, self.index_cheat[polar_label]["index"]).tolist()
            for i, s in enumerate(similarities):
                if s >= threshold:
                    id_and_score.append((i, s, polar_label))

        if top_k > 0:
            id_and_score = sorted(id_and_score, key=lambda x: x[1], reverse=True)[:top_k]
        elif top_k < 0:
            id_and_score = sorted(id_and_score, key=lambda x: x[1], reverse=True)[top_k:]
        results = [(self.index_cheat[label]["sentences"][idx], idx, score, label) for idx, score, label in id_and_score]
        return results
    
    def search_diverse_label(self, queries, threshold, top_k, label_space, train_dataset):
        similarities = self.similarity(queries, self.index["index"]).tolist()
        id_and_score = []
        for i, s in enumerate(similarities):
            if s >= threshold:
                id_and_score.append((i, s))

        id_and_score = sorted(id_and_score, key=lambda x: x[1], reverse=True)
        ids = [idx for idx, score in id_and_score]
        results = []
        temp = []
        for i in ids:
            if len(results) == top_k:
                break
            if len(results) <= top_k - label_space:
                results.append(i)
                temp.append(train_dataset[i]['coarse_label'])
            else:
                if train_dataset[i]['coarse_label'] in temp:
                    continue
                else:
                    results.append(i)
                    temp.append(train_dataset[i]['coarse_label'])

        results = [(self.index["sentences"][idx], idx) for idx in results]         
        #results = [(self.index["sentences"][idx], idx, score) for idx, score in id_and_score]
        return results
    

def loader(benchmark_name, task_name, train_size=None, validation_size=None):

    train_dataset = None
    test_dataset = None
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    set_seed(1)
    # load dataset
    if benchmark_name == 'huggingface':
        # TREC, AGNews
        if task_name in ['sick', 'poem_sentiment', 'sick']:
            test_dataset = load_dataset(task_name, split='validation')
        elif task_name in ['climate_fever']:
            # no train set, split test set.
            print(f'No train set for {task_name}. Split test set {train_size} / {validation_size}')
            dataset = load_dataset(task_name, split='test')
            dataset = dataset.train_test_split(train_size=train_size, test_size=validation_size, shuffle=True)
            train_dataset, test_dataset = dataset['train'], dataset['test']

            # train_dataset = load_dataset(task_name, split=f'test[:{train_size}]')
            # test_dataset = load_dataset(task_name, split=f'test[-{validation_size}:]')
        elif task_name in ['medical_questions_pairs', 'hate_speech18']:
            # no validation set, split train set.
            print(f'No validation set for {task_name}. Split train set {train_size} / {validation_size}')
            dataset = load_dataset(task_name, split='train')
            if task_name == 'hate_speech18':
                dataset = dataset.filter(lambda example: example[label_key] in [0, 1])
            dataset = dataset.train_test_split(train_size=train_size, test_size=validation_size, shuffle=True)
            train_dataset, test_dataset = dataset['train'], dataset['test']
        elif task_name == 'rest14':
            train_dataset = load_dataset('alexcadillon/SemEval2014Task4', 'restaurants', split='train')
            test_dataset = load_dataset('alexcadillon/SemEval2014Task4', 'restaurants', split='test')
        elif task_name == 'laptop14':
            train_dataset = load_dataset('alexcadillon/SemEval2014Task4', 'laptops', split='train')
            test_dataset = load_dataset('alexcadillon/SemEval2014Task4', 'laptops', split='test')
        elif task_name == 'rest15':
            train_dataset = load_dataset('Yaxin/SemEval2015Task12Raw', 'restaurants', split='train')
            test_dataset = load_dataset('Yaxin/SemEval2015Task12Raw', 'restaurants', split='test')
        elif task_name == 'rest16':
            train_dataset = load_dataset('Yaxin/SemEval2016Task5Raw', 'restaurants_english', split='train')
            test_dataset = load_dataset('Yaxin/SemEval2016Task5Raw', 'restaurants_english', split='test')
        else:
            test_dataset = load_dataset(task_name, split='test')

        if train_dataset is None:
            train_dataset = load_dataset(task_name, split='train')
    elif benchmark_name == 'ethos':
        dataset = load_dataset(benchmark_name, 'multilabel', split='train')
        dataset = dataset.train_test_split(train_size=train_size, test_size=validation_size, shuffle=True)
        train_dataset, test_dataset = dataset['train'], dataset['test']

        # train_dataset = load_dataset(benchmark_name, 'multilabel', split=f'train[:{train_size}]')
        # test_dataset = load_dataset(benchmark_name, 'multilabel', split=f'train[-{validation_size}:]')
    elif benchmark_name == 'financial_phrasebank':
        dataset = load_dataset(benchmark_name, task_name, split='train')
        dataset = dataset.train_test_split(train_size=train_size, test_size=validation_size, shuffle=True)
        train_dataset, test_dataset = dataset['train'], dataset['test']

        # train_dataset = load_dataset(benchmark_name, task_name, split=f'train[:{train_size}]')
        # test_dataset = load_dataset(benchmark_name, task_name, split=f'train[-{validation_size}:]')

    else:
        # GLUE, SuperGLUE, tweet_eval
        train_dataset = load_dataset(benchmark_name, task_name, split='train')
        if task_name == 'mnli':
            test_dataset = load_dataset(benchmark_name, task_name, split='validation_matched')
        else:
            test_dataset = load_dataset(benchmark_name, task_name, split='validation')
    
    return train_dataset, test_dataset



if __name__ == "__main__":
    #Scorer('roberta-large', pooler="cls_before_pooler")

       # set random seet to 1 to sample same train/validation set


    # key for input sentence

    parser = argparse.ArgumentParser()
    parser.add_argument('-strategy', required=True, choices=['ret', 'ret-bm25', 'rand', 'homo-ret', 'homo-rand', 'hetero-ret', 'hetero-rand', 'diverse'], type=str, default='ret')
    parser.add_argument(
        "--n_samples",
        type=int,
        default=5,
        help="Number of samples for in-context learning."
    )
    #parser.add_argument('-dataset', required=True)
    args = parser.parse_args()

    TOPK = args.n_samples


    dataset_name_map = {
        'sst2': 'glue_sst2',
        'rte': 'glue_rte',
        'hate': 'tweet_hate',
        'stance_atheism': 'tweet_stance_atheism',
        'stance_feminist': 'tweet_stance_feminist',
        'trec': 'trec_',
        'ag_news': 'ag_new',
        'mrpc': 'glue_mrpc',
        'wnli': 'wnli',
        'medical_questions_pairs': 'medical_questions_pairs',
        'hate_speech18': 'hs18',
        'conll2003': 'conll03',
        'wnut_17': 'wnut17',
        'bc2gm_corpus': 'bc2gm',
        'rest14': 'rest14',
        'rest15': 'rest15',
        'rest16': 'rest16',
        'laptop14': 'laptop14'
    }

    #for source_domain in domain_list:
    for (benchmark_name, task_name) in dataset_list:
            #if target_domain == source_domain:
        if task_name == 'medical_questions_pairs':
            TOPK = 6

        keys_dict = task2keys.get(task_name)
        sentence1_key, sentence2_key = keys_dict.get('input_key')
        label_key = keys_dict.get('label_key')
        if 'train_size' in keys_dict.keys():
            train_size = keys_dict.get('train_size')
            validation_size = keys_dict.get('validation_size')
        else:
            train_size = None
            validation_size = None
        train_dataset, test_dataset = loader(benchmark_name, task_name, train_size, validation_size)
        scorer = Scorer('princeton-nlp/sup-simcse-roberta-large', pooler="cls", use_transformer=True)
        #target_domain = 'bc5cdr'

        if task_name not in ['conll2003', 'wnut_17', 'bc2gm_corpus', 'rest14', 'rest15', 'rest16', 'laptop14']:
            label_map = TASK_DIC[dataset_name_map[task_name]]['label_map']

        # NLI
        if task_name in ["rte", "wnli"]:
            train_text_label = [f"Premise: {sample[sentence1_key]} Hypothesis: {sample[sentence2_key]} Label: {label_map[sample[label_key]]}" for sample in train_dataset]
        # Paraphrase
        elif task_name in ['mrpc', 'medical_questions_pairs']:
            train_text_label = [f"Sentence1: {sample[sentence1_key]} Sentence2: {sample[sentence2_key]} Label: {label_map[sample[label_key]]}" for sample in train_dataset]
        elif task_name in ['conll2003', 'wnut_17', 'bc2gm_corpus']:
            train_text_label = [' '.join(sample[sentence1_key]) for sample in train_dataset]
        elif task_name in ['rest14', 'rest15', 'rest16', 'laptop14']:
            train_text_label = [sample[sentence1_key] for sample in train_dataset]
        else:
            train_text_label = [f"Sentence: {sample[sentence1_key]} Label: {label_map[sample[label_key]]}" for sample in train_dataset]

        if args.strategy == 'ret':
            scorer.build_index(train_text_label, use_faiss=False)
        elif args.strategy == 'ret_bm25':
            scorer.build_index_bm25(train_text_label, top_k=TOPK)
        
        if 'cheat' in args.strategy:
            train_ds_by_label = {lab: [i for i, e in enumerate(train_dataset) if e[label_key] == lab] for lab in range(len(label_map))}
            if task_name in ['rte', 'wnli']:
                for lab in range(len(label_map)):
                    scorer.build_index_cheat([f'Premise: {train_dataset[i][sentence1_key]} Hypothesis: {train_dataset[i][sentence2_key]} Label: {label_map[train_dataset[i][label_key]]}' for i in train_ds_by_label[lab]], polarity=lab)
            elif task_name in ['mrpc', 'medical_questions_pairs']:
                for lab in range(len(label_map)):
                    scorer.build_index_cheat([f'Sentence1: {train_dataset[i][sentence1_key]} Sentence2: {train_dataset[i][sentence2_key]} Label: {label_map[train_dataset[i][label_key]]}' for i in train_ds_by_label[lab]], polarity=lab)
            else:
                for lab in range(len(label_map)):
                    scorer.build_index_cheat([f'Sentence1: {train_dataset[i][sentence1_key]} Label: {label_map[train_dataset[i][label_key]]}' for i in train_ds_by_label[lab]], polarity=lab)

        query_w_demos = []

        if task_name == 'ag_news':
            with open(f'ag_new_1000indexes.txt', 'r') as f:
                #f.write(str(sample_index))
                sample_index = f.readlines()
                import ast
                sample_index = ast.literal_eval(sample_index[0])
            test_dataset = [test_dataset[i] for i in sample_index]
        for query_id, q_sent in tqdm(enumerate(test_dataset), total=len(test_dataset)):
            if task_name in ["rte", "wnli"]:
                temp = {'sentence1': q_sent[sentence1_key], 'sentence2': q_sent[sentence2_key], 'label': q_sent[label_key]}
                query = f"Premise: {q_sent[sentence1_key]} Hypothesis: {q_sent[sentence2_key]}"
            elif task_name in ['mrpc', 'medical_questions_pairs']:
                temp = {'sentence1': q_sent[sentence1_key], 'sentence2': q_sent[sentence2_key], 'label': q_sent[label_key]}
                query = f"Sentence1: {q_sent[sentence1_key]} Sentence2: {q_sent[sentence2_key]}"
            else:
                temp = {'sentence1': q_sent[sentence1_key], 'label': q_sent[label_key]}
                query = f"Sentence: {q_sent[sentence1_key]}"

            if args.strategy == 'ret':
                results = scorer.search(query, threshold=0, top_k=TOPK)
            elif args.strategy == 'ret_bm25':
                results = scorer.index.get_relevant_documents(query)
            elif args.strategy == 'diverse':
                results = scorer.search_diverse_label(query, threshold=0, top_k=TOPK, label_space=5, train_dataset=train_dataset)
            elif args.strategy == 'homo-ret':
                results = scorer.search_cheat(query, polarity=q_sent[label_key], threshold=0, top_k=TOPK, device=None)
                results_correct_index = []
                for i,e in enumerate(results):
                    tt = train_ds_by_label[q_sent[label_key]][e[1]]
                    results_correct_index.append((e[0], tt, e[2]))
                results = results_correct_index
            elif args.strategy == 'homo-rand':
                results = scorer.search_cheat_rand(query, train_ds_by_label, polarity=q_sent[label_key], top_k=TOPK)
            elif args.strategy == 'hetero-ret':
                results = scorer.search_reverse_cheat_ret(query, polarity=q_sent[label_key], threshold=0, top_k=TOPK, device=None)
                #results = scorer.search_cheat(query, polarity=q_sent[label_key], threshold=0, top_k=TOPK, device=None)
                results_correct_index = []
                for i,e in enumerate(results):
                    tt = train_ds_by_label[e[3]][e[1]]
                    results_correct_index.append((e[0], tt, e[2]))
                results = results_correct_index
            elif args.strategy == 'hetero-rand':
                results = scorer.search_reverse_cheat_rand(query, train_ds_by_label, polarity=q_sent[label_key], top_k=TOPK)
            
            #results = scorer.search(query, threshold=0, top_k=TOPK)
            #random.seed(13)
            #results = scorer.search_same_label(query, train_ds_by_label, TOPK)

            #results = [[0, train_text_label.index(e.page_content)] for e in results]
            """results = scorer.search_cheat(query, polarity=q_sent[label_key], threshold=0, top_k=TOPK, device=None)
            results_correct_index = []
            for i,e in enumerate(results):
                tt = train_ds_by_label[q_sent[label_key]][e[1]]
                results_correct_index.append((e[0], tt, e[2]))
            results = results_correct_index"""
            if sentence2_key:
                if task_name != 'medical_questions_pairs':
                    demo = [{'sid': si[1], 'sent1': train_dataset[si[1]][sentence1_key], 'sent2': train_dataset[si[1]][sentence2_key],
                            's_label': train_dataset[si[1]][label_key]} for si in results]
                else:
                    for i, si in enumerate(results):
                        if train_dataset[si[1]][sentence1_key] == q_sent[sentence1_key]:
                            results.pop(i)
                    if len(results) == TOPK+1:
                        results.pop(-1)
                    demo = [{'sid': si[1], 'sent1': train_dataset[si[1]][sentence1_key], 'sent2': train_dataset[si[1]][sentence2_key],
                            's_label': train_dataset[si[1]][label_key]} for si in results]
                    
            else:
                demo = [{'sid': si[1], 'sent1': train_dataset[si[1]][sentence1_key], 's_label': train_dataset[si[1]][label_key]} for si in results]

            #source = [{'sid': si[1], 's_sent': source_texts[filtered_source_id[si[1]]], 
            #        's_label': get_entity_type(source_texts[filtered_source_id[si[1]]], source_labels_type[filtered_source_id[si[1]]])} for si in results]
            temp.update({'source': demo})
            query_w_demos.append(temp)

        dir_base = f'./data_new/{args.strategy}/{dataset_name_map[task_name]}/'
        if not os.path.exists(dir_base):
            os.makedirs(dir_base)
        with open(f'{dir_base}/test.jsonl', 'w') as f:
            for row in query_w_demos:
                f.write(json.dumps(row))
                f.write('\n')
        f.close()

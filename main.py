import os
os.environ['HF_HOME'] = '/home/wuyin/hf_cache/'
os.environ['CUDA_VISIBLE_DEVICES'] = "2,3"
import time
import json
import argparse
import ast
import random
random.seed(1)

parser = argparse.ArgumentParser()
parser.add_argument('-num_demo', required=True, choices=[-1, 0, 1, 3, 5, 7, 10,15], type=int)
parser.add_argument('-dataset', required=True)
parser.add_argument('-seed', required=False, default='100')
parser.add_argument('-model_name', required=True)
parser.add_argument('-hint', action='store_true')
parser.add_argument('-ret', action='store_true')
parser.add_argument('-ret_setting', choices=['ret', 'ret-bm25', 'homo-ret', 'homo-rand', 'hetero-ret', 'hetero-rand', 'diverse'], type=str, default='ret')
parser.add_argument('-no_format', action='store_true')
parser.add_argument('-batch_size', default=8)
parser.add_argument('-gen_style', required=False)

#parser.add_argument('-dataset', required=True)
args = parser.parse_args()

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import tqdm
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import pandas as pd
#from knockknock import teams_sender
#from tqdm import tqdm

from prepare_prompt import get_prompts, TASK_DIC, PromptGetter


def main():
    logger = get_logger(__name__)

    accelerator = Accelerator()

    access_token = 'your_huggingface_token'

    model_collections = {
        'mistral':'mistralai/Mistral-7B-Instruct-v0.2', 
        'llama2': 'meta-llama/Llama-2-13b-chat-hf', 
        'flant5': 'google/flan-t5-xxl'
        }

    if args.model_name == 'flant5':
        model = AutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-xxl', torch_dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-xxl')
    elif args.model_name == 'mistral':
        model = AutoModelForCausalLM.from_pretrained(model_collections[args.model_name], torch_dtype=torch.bfloat16 , attn_implementation="flash_attention_2", token=access_token)
        tokenizer = AutoTokenizer.from_pretrained(model_collections[args.model_name], token=access_token)
        #from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
        #tokenizer = MistralTokenizer.v1()
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    elif args.model_name == 'llama2':
        model = AutoModelForCausalLM.from_pretrained(model_collections[args.model_name], torch_dtype=torch.bfloat16 , attn_implementation="flash_attention_2", token=access_token)
        tokenizer = AutoTokenizer.from_pretrained(model_collections[args.model_name], token=access_token)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = 'left'
    else:
        raise NameError("Check model name.")

    torch.manual_seed(int(args.seed))
    torch.cuda.manual_seed_all(int(args.seed))

    #device = accelerator.device

    if args.num_demo <= 0:
        num_demo = 5
    else:
        num_demo = args.num_demo

    if args.dataset in ['conll03_type', 'wnut17_type', 'bc2gm_type']:
        dataset_name = args.dataset[:-5]
    elif args.dataset == 'ag_new':
        dataset_name = args.dataset
    elif args.dataset == 'llama_test':
        dataset_name = 'glue_sst2'
    else:
        dataset_name = args.dataset
    
    if args.ret:
        with open(f'./data_new/{args.ret_setting}/{dataset_name}/test.jsonl') as f:
            all_samples = f.readlines()
        all_demos = None
    elif args.dataset in ['common_gen', 'roc_story_ending', 'wikiauto', 'reddit', 'samsum', 'roc_story']:
        with open(f'./data/{num_demo}/{dataset_name}/k-{num_demo}-seed-{args.seed}/{dataset_name}/{args.gen_style}.jsonl') as f:
            all_demos = f.readlines()
        with open(f'./data/{num_demo}/{dataset_name}/k-{num_demo}-seed-{args.seed}/{dataset_name}/eval.jsonl') as f:
            all_samples = f.readlines()
    else:
        with open(f'./data/{num_demo}/{dataset_name}/k-{num_demo}-seed-{args.seed}/train.jsonl') as f:
            all_demos = f.readlines()
        with open(f'./data/{num_demo}/{dataset_name}/k-{num_demo}-seed-{args.seed}/test.jsonl') as f:
            all_samples = f.readlines()


    if args.dataset in ['ag_new', 'hs18']:
        #sample_index = random.sample(list(range(len(all_samples))), 1000)
        #sample_index = sorted(sample_index)
        #all_samples = [all_samples[i] for i in sample_index]
        with open(f'{dataset_name}_1000indexes.txt', 'r') as f:
            sample_index = f.readlines()
            sample_index = ast.literal_eval(sample_index[0])
        all_samples = [all_samples[i] for i in sample_index]

    #all_samples = [json.loads(e) for e in all_samples]

    # if args.dataset in ['bc2gm', 'bc2gm_type']:
    #     prompt_list, labels = get_prompts(all_demos, all_samples, args.dataset, num_demo=args.num_demo, hint=args.hint, isT5=args.model_name=='flant5', isllama=args.model_name=='llama2', indexes=sample_index, no_format=args.no_format)
    # elif args.dataset in ['conll03', 'wnut17', 'conll03_type', 'wnut17_type', 'rest14', 'rest15', 'rest16', 'laptop14']:
    #     prompt_list, labels = get_prompts(all_demos, all_samples, args.dataset, num_demo=args.num_demo, hint=args.hint, isT5=args.model_name=='flant5', isllama=args.model_name=='llama2',indexes=list(range(len(all_samples))), no_format=args.no_format, ret=args.ret)
    # else:
    #     prompt_list, labels = get_prompts(all_demos, all_samples, dataset_name, num_demo=args.num_demo, hint=args.hint, isT5=args.model_name=='flant5', isllama=args.model_name=='llama2', no_format=args.no_format, ret=args.ret)

    prompt_getter = PromptGetter(all_demos, all_samples, dataset_name=args.dataset, num_demo=int(args.num_demo), hint=args.hint, isT5=args.model_name=='flant5', isllama=args.model_name=='llama2', isGPT=False, indexes=None, no_format=args.no_format, ret=args.ret)

    prompt_list, labels, _ = prompt_getter.get_prompt()

    prompt_list = tokenizer(prompt_list, padding=True, return_tensors="pt")
    labels = torch.LongTensor(labels)

    ds = TensorDataset(prompt_list['input_ids'], prompt_list['attention_mask'], labels)

    dl = DataLoader(ds, batch_size=args.batch_size)

    model, dl = accelerator.prepare(model, dl)

    model.eval()

    #model = model.to(device)

    #if accelerator.is_main_process:
    all_predictions = []


    pbar = tqdm(total=len(dl))

    for batch in dl:
        with torch.no_grad():
            generate_ids = model.generate(batch[0], attention_mask=batch[1], max_new_tokens=256)#, temperature=0.5, do_sample=True, num_beams=5)
        #accelerator.wait_for_everyone()

        #generate_ids = accelerator.gather(generate_ids)
        output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        #logger.info(output, main_process_only=False)
        all_predictions.extend(accelerator.gather_for_metrics(list(zip(output, batch[0].detach().cpu(), batch[2].detach().cpu().tolist()))))
        #if accelerator.is_main_process:
        pbar.update()
        #accelerator.print(output)


    accelerator.wait_for_everyone()
    out_dir = 'results'
    if args.num_demo > 0:
        if args.no_format:
            out_dir += 'ICL'
        else:
            out_dir += 'DI_ICL'
    elif args.num_demo == 0:
        if args.no_format:
            out_dir += 'zeroshot'
        else:
            out_dir += 'DI'
    else:
        out_dir += 'Incorrect_Label'
    
    if args.ret:
        out_dir += f'retrieval_series/{args.ret_setting}'

    if args.gen_style:
        if args.gen_style == 'train_short':
            gen_style = 'basic'
        else:
            gen_style = args.gen_style
        out_dir += f'generation/{gen_style}'

    if accelerator.is_main_process:
        all_pred_with_input = []
        for e in all_predictions:
            temp = tokenizer.batch_decode(e[1].unsqueeze(0), skip_special_tokens=True, clean_up_tokenization_spaces=False)
            all_pred_with_input.append([e[0], e[2], temp[0]])
        dir_base = f"./{out_dir}/{args.model_name}/{args.num_demo}/{args.dataset}/"
        #dir_base = f"./{out_dir}/test/{args.model_name}/{args.dataset}/"
        if not os.path.exists(dir_base):
            os.makedirs(dir_base)
        with open(f'./{dir_base}/s{args.seed}.txt', 'w') as f:
            for row in all_pred_with_input:
                f.write(str(row))
                f.write('\n')

    #return f"{args.model_name}/{args.num_demo}/{args.dataset}/s{args.seed}"

if __name__ == "__main__":
    main()
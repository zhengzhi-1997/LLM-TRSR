from datasets import load_dataset
from trl import SFTTrainer
import pandas as pd
import re
import random
from tqdm import tqdm
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, DataCollatorForLanguageModeling
from transformers import GPTNeoXForCausalLM, BloomForCausalLM, OPTForCausalLM
import argparse
import json
from tqdm import tqdm
import time
from datasets import load_from_disk
import pandas as pd
import os
from torch.utils.data import DataLoader
import re
import torch
import transformers
import numpy as np
from datasets import Dataset
from torch.nn.functional import softmax
from sklearn.metrics import ndcg_score
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import json
from peft import PeftModel

def set_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--data_path', default='/code/zhengzhi/LLM4LongSeqRec/LLM4Rec/dataset/MIND/hierarchical_12k', type=str, help='')
    # parser.add_argument('--output_dir', default='/code/zhengzhi/LLM4LongSeqRec/LLM4Rec/output/MIND/hierarchical_12k', type=str, help='')
    parser.add_argument('--device', default='0', type=str, help='')
    parser.add_argument('--model_dir', default="public/opt/opt-125m", type=str, help='')
    parser.add_argument('--data_path', default='/code/zhengzhi/LLM4LongSeqRec/LLM4Rec/dataset', type=str, help='')
    parser.add_argument('--dataset', default='MIND', type=str, help='MIND or amazon')
    # parser.add_argument('--data_type', default='recurrent_12k', type=str, help='hierarchical or recurrent')
    parser.add_argument('--output_dir', default='/code/zhengzhi/LLM4LongSeqRec/LLM4Rec/output', type=str, help='')
    # parser.add_argument('--prompt_type', default='full', type=str, help='full, summary, recent')
    # parser.add_argument('--max_seq_length', default=1024, type=int, help='max_seq_length for LLM')
    parser.add_argument('--summary_type', default='hierarchical', type=str, help='hierarchical, recurrent, no_summary')
    parser.add_argument('--summary_size', default='30b', type=str, help='30b, 13b, 7b')
    parser.add_argument('--item_num', default=5, type=int, help='item num in historical sequence')
    parser.add_argument('--batch_size', default=1, type=int, help='')
    parser.add_argument('--local_rank', type=int, default=0, help='')
    parser.add_argument('--lora', type=str, default='no', help='use lora or not')
    parser.add_argument('--N', default=0, type=int, help='evaluate on which epoch')
    return parser.parse_args()


def recall_at_k(y_true, y_score, k):
    top_k_indices = np.argsort(-y_score, axis=1)[:, :k]
    recalls = []
    for i in range(y_true.shape[0]):
        try:
            true_labels = y_true[i, top_k_indices[i]]
        except:
            print(y_true.shape, y_score.shape, top_k_indices.shape)
            exit(0)
        recalls.append(np.sum(true_labels) / np.sum(y_true[i]))
    return np.mean(recalls)

def mrr_at_k(y_true, y_score, k):
    mrrs = []
    for i in range(y_true.shape[0]):
        sorted_indices = np.argsort(-y_score[i])
        rank = 0
        for j in range(k):
            if y_true[i, sorted_indices[j]] == 1:
                rank = j + 1
                break
        mrrs.append(1.0 / rank if rank > 0 else 0)
    return np.mean(mrrs)

def ndcg_at_k(y_true, y_score, k):

    def dcg_at_k(r, k):
        """计算DCG@k的值"""
        r = np.asfarray(r)[:k]
        return np.sum(r / np.log2(np.arange(2, r.size + 2)))
    
    ndcgs = []
    for i in range(y_true.shape[0]):
        sorted_indices = np.argsort(-y_score[i])
        predicted_relevance = y_true[i, sorted_indices][:k]
        dcg_value = dcg_at_k(predicted_relevance, k)
        
        sorted_true_relevance = np.sort(y_true[i])[::-1]  # 从大到小排序
        idcg_value = dcg_at_k(sorted_true_relevance, k)
        
        ndcgs.append(dcg_value / (idcg_value + 1e-7))  # 防止除以0
    
    return np.mean(ndcgs)

def get_results(tokenizer, model, dataloader, args):
    model.eval()
    yes_id, no_id= tokenizer.convert_tokens_to_ids('yes'), tokenizer.convert_tokens_to_ids('no')
    yes_logits, no_logits, labels = [], [], []
    with torch.no_grad():
        step = 0
        total_elements = len(dataloader)
        start_time = time.time()  # record the start time
        for batch in tqdm(dataloader):
            step += 1
            if (step + 1) % 10 == 0:
                elapsed_time = time.time() - start_time  # time elapsed from start until now
                avg_time_per_loop = elapsed_time / (step+1)  # average time per loop
                remaining_elements = total_elements - (step+1)  # remaining number of elements
                estimated_time_remaining = avg_time_per_loop * remaining_elements  # estimated time for the rest of loops
                print(f"Processed {step+1} steps. Estimated time remaining: {estimated_time_remaining / 60} minutes.")

            labels.append(batch['label'])
            batch_size = len(batch['label'])
            
            input_ids = batch['input_ids'].to("cuda:{}".format(args.device))
            # try:
            #     logits = model(input_ids).logits
            # except:
            #     print(input_ids)
            #     exit(0)
            logits = model(input_ids).logits
            for i in range(batch_size):
                yes_logits.append(logits[i, batch['last_position'][i], yes_id].item())
                no_logits.append(logits[i, batch['last_position'][i], no_id].item())
                
            # yes_logits, no_logits = torch.tensor(yes_logits).reshape(-1), torch.tensor(no_logits).reshape(-1)
            # print(yes_logits)
            # exit(0)

    yes_logits, no_logits = torch.tensor(yes_logits).reshape(-1), torch.tensor(no_logits).reshape(-1)
    yes_probs = F.softmax(torch.stack([yes_logits, no_logits], dim=-1), dim=-1)[:, 0].cpu().numpy()
    labels = torch.concat(labels).cpu().numpy()
    y_true = labels.reshape(-1, 21)
    y_score = yes_probs.reshape(-1, 21)
    k_list = [3, 5, 10]
    results = {}
    for k in k_list:
        results[f'recall@{k}'] = recall_at_k(y_true, y_score, k)
        results[f'mrr@{k}'] = mrr_at_k(y_true, y_score, k)
        results[f'ndcg@{k}'] = ndcg_at_k(y_true, y_score, k)
    return results


def main():
    args = set_args()
   
    if args.summary_size == '30b':
        if args.summary_type == 'recurrent':
            data_path = os.path.join(args.data_path, args.dataset, 'recurrent_12k')
        else:
            data_path = os.path.join(args.data_path, args.dataset, 'hierarchical_12k')
    else:
        if args.summary_type == 'recurrent':
            data_path = os.path.join(args.data_path, args.dataset, f'recurrent_3k_{args.summary_size}')
        else:
            data_path = os.path.join(args.data_path, args.dataset, f'hierarchical_3k_{args.summary_size}')
    
    model_name = os.path.basename(args.model_dir)
    if args.lora == 'yes':
        model_name += '_lora'
   
    if args.summary_size == '30b':
        output_dir = os.path.join(
            args.output_dir, model_name, args.dataset, args.summary_type, f'item_num_{args.item_num}'
        )
    else:
        output_dir = os.path.join(
            args.output_dir, model_name, args.dataset, f'{args.summary_type}_{args.summary_size}', f'item_num_{args.item_num}'
        )
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    # if 'pythia' in args.model_dir or 'Llama-2' in args.model_dir:
    #     tokenizer.pad_token = tokenizer.eos_token
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    if 'pythia' in args.model_dir:
        tokenizer.pad_token = tokenizer.eos_token
    if 'Llama' in args.model_dir:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'right'

   

    def encode(item):
        temp_data = {}
        if args.dataset == 'amazon':
            prompt_text = "### User:\nGiven a preference summary of the user, and information related to the historical items the user has purchased, predict whether the user will click on the given next product. Note that the preference summary captures long-term interests, while the recent items indicate short term interests. Both of them should be holistically considered for a more comprehensive understanding of user behavior. Please output 'yes' or 'no'."
        if args.dataset == 'MIND':
            prompt_text = "### User:\nGiven a news reading preference summary of the user, and information related to the historical news the user has read, predict whether the user will click on the given next news. Note that the preference summary captures long-term interests, while the recent five news indicate short term interests. Both of them should be holistically considered for a more comprehensive understanding of user behavior. Please output 'yes' or 'no'."

        if args.summary_type == 'no_summary':
            preference_text = '\n[Preference Summary]\n' + 'NULL'
        else:
            preference_text = '\n[Preference Summary]\n' + item['summary']

        history_item_list = item['all_history_items'].split('\n')
        
        if args.item_num == 0:
            if args.dataset == 'amazon':
                history_item_text = '\n[Historical Items]\n' + 'NULL'
            if args.dataset == 'MIND':
                history_item_text = '\n[Historical News]\n' + 'NULL'
        else:
            
            if args.dataset == 'amazon':
                history_item_text = '\n[Historical Items]\n' + '\n'.join(history_item_list[-args.item_num:])
            if args.dataset == 'MIND':
                history_item_text = '\n[Historical News]\n' + '\n'.join(history_item_list[-args.item_num:])
            

        if args.dataset == 'amazon':
            next_item_text = '\n[Next Item]\n' + item['next_item']
        if args.dataset == 'MIND':
            next_item_text = '\n[Next News]\n' + item['next_item']

        text = prompt_text + preference_text + history_item_text + next_item_text + '\n### Assistant:\n'

        input_ids = tokenizer.encode(text, return_tensors="pt", padding='max_length', max_length=2048, truncation=True)
        # input_ids = tokenizer.encode(text, return_tensors="pt", padding='max_length', truncation=True)
        last_position = min(len(tokenizer.encode(text)) - 1, 2047)
        temp_data['input_ids'] = input_ids
        temp_data['label'] = item['label']
        temp_data['last_position'] = last_position
        return temp_data
    
    def collate_fn(batch):
        input_ids = [torch.tensor(item['input_ids'][0]) for item in batch]
        label = [item['label'] for item in batch]
        last_position = [item['last_position'] for item in batch]
        return {
            'input_ids': torch.stack(input_ids).to("cuda:{}".format(args.device)),
            'label': torch.tensor(label).to("cuda:{}".format(args.device)),
            'last_position': torch.tensor(last_position).to("cuda:{}".format(args.device))
        }
    
    df_val = pd.read_parquet(os.path.join(data_path, 'validation.parquent'))
    # df_val = pd.read_parquet(os.path.join(data_path, 'test.parquent'))
    val_dataset = Dataset.from_pandas(df_val)
    # # 小数据量测试
    # val_dataset = val_dataset.select(range(2100))

    val_dataset = val_dataset.map(encode, batched=False, num_proc=16)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # 对所有checkpoint进行遍历
    # output_dir = os.path.join(args.output_dir, os.path.basename(args.model_dir), args.prompt_type)
    all_items = os.listdir(output_dir)
    
    # 过滤出以"checkpoint"为开头的文件夹名
    # checkpoint_folders = sorted([folder for folder in all_items if folder.startswith("checkpoint") and os.path.isdir(os.path.join(output_dir, folder))])
    def sort_key(s):
        prefix, num = s.split('-')
        return int(num)
    checkpoint_folders = sorted(
        [folder for folder in all_items if folder.startswith("checkpoint") and os.path.isdir(os.path.join(output_dir, folder))],
        key=sort_key
        )
    # 对第N个folder进行测试
    folder = checkpoint_folders[args.N]
    print(folder)
    
    if args.lora == 'no':
        if args.model_dir == 'public/EleutherAI/pythia-160m':
            model = GPTNeoXForCausalLM.from_pretrained(os.path.join(output_dir, folder), torch_dtype=torch.bfloat16).to("cuda:{}".format(args.device))
        elif args.model_dir == 'public/bigscience/bloom-560m':
            model = BloomForCausalLM.from_pretrained(os.path.join(output_dir, folder), torch_dtype=torch.bfloat16).to("cuda:{}".format(args.device))
        elif args.model_dir == 'public/opt/opt-350m':
            model = OPTForCausalLM.from_pretrained(os.path.join(output_dir, folder), torch_dtype=torch.bfloat16).to("cuda:{}".format(args.device))
        else:
            model = AutoModelForCausalLM.from_pretrained(os.path.join(output_dir, folder), torch_dtype=torch.bfloat16).to("cuda:{}".format(args.device))
    elif args.lora == 'yes':
        model = AutoModelForCausalLM.from_pretrained(args.model_dir, torch_dtype=torch.bfloat16)
        model = PeftModel.from_pretrained(model, os.path.join(output_dir, folder)).to("cuda:{}".format(args.device))

    results = get_results(tokenizer, model, val_dataloader, args)
    if not os.path.exists(os.path.join(output_dir, 'val')):
        os.makedirs(os.path.join(output_dir, 'val'))
    with open(os.path.join(output_dir, f'val/{folder}.json'), 'w') as f:
        json.dump(results, f)

if __name__ == "__main__":
    main()

    
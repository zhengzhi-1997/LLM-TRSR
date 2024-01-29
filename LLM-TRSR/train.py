# import trl
# print(trl.__version__)
# exit(0)
from datasets import load_dataset
from trl import SFTTrainer
import pandas as pd
import re
import random
from tqdm import tqdm
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, DataCollatorForLanguageModeling, GPTNeoXForCausalLM, BloomForCausalLM
from transformers import OPTForCausalLM
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
from peft import LoraConfig, get_peft_model

def set_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--data_path', default='/code/zhengzhi/LLM4LongSeqRec/LLM4Rec/dataset/MIND/hierarchical_12k', type=str, help='')
    # parser.add_argument('--output_dir', default='/code/zhengzhi/LLM4LongSeqRec/LLM4Rec/output/MIND/hierarchical_12k', type=str, help='')
    parser.add_argument('--model_dir', default="public/opt/opt-125m", type=str, help='')
    parser.add_argument('--data_path', default='/code/zhengzhi/LLM4LongSeqRec/LLM4Rec/dataset', type=str, help='')
    parser.add_argument('--dataset', default='MIND', type=str, help='MIND or amazon')
    # parser.add_argument('--data_type', default='hierarchical_12k', type=str, help='hierarchical_12k or recurrent_12k')
    parser.add_argument('--output_dir', default='/code/zhengzhi/LLM4LongSeqRec/LLM4Rec/output', type=str, help='')
    # parser.add_argument('--prompt_type', default='full', type=str, help='full, summary, recent')
    # parser.add_argument('--max_seq_length', default=1024, type=int, help='max_seq_length for LLM')
    parser.add_argument('--summary_type', default='hierarchical', type=str, help='hierarchical, recurrent, no_summary')
    parser.add_argument('--summary_size', default='30b', type=str, help='30b, 13b, 7b')
    parser.add_argument('--item_num', default=5, type=int, help='item num in historical sequence')
    parser.add_argument('--num_train_epochs', default=5, type=int, help='')
    parser.add_argument('--lr', default=1e-4, type=float, help='')
    parser.add_argument('--per_device_train_batch_size', default=8, type=int, help='')
    parser.add_argument('--gradient_accumulation_steps', default=8, type=int, help='')
    parser.add_argument('--local_rank', type=int, default=0, help='')
    parser.add_argument('--lora', type=str, default='no', help='use lora or not')
    parser.add_argument('--lora_r', type=int, default=8, help='')
    parser.add_argument("--deepspeed", type=str, default='/code/zhengzhi/LLM4LongSeqRec/LLM4Rec/ds_zero2.json', help="Path to deepspeed config file.")
    return parser.parse_args()

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

    df_train = pd.read_parquet(os.path.join(data_path, 'train.parquent'))
    train_dataset = Dataset.from_pandas(df_train)
   
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    if 'pythia' in args.model_dir:
        tokenizer.pad_token = tokenizer.eos_token
    if 'Llama' in args.model_dir:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'right'
    

    if 'Llama' in args.model_dir or 'llama' in args.model_dir:
        peft_config = LoraConfig(r=args.lora_r,
                            lora_alpha=16,
                            target_modules=['q_proj','k_proj','v_proj','o_proj'],
                            lora_dropout=0.05,
                            bias="none",
                            task_type="CAUSAL_LM",
                            inference_mode=False,
                            )
    if 'pythia' in args.model_dir:
        peft_config = LoraConfig(r=args.lora_r,
                            lora_alpha=16,
                            target_modules=['query_key_value'],
                            lora_dropout=0.05,
                            bias="none",
                            task_type="CAUSAL_LM",
                            inference_mode=False,
                            )


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

        if item['label'] == 1:
            text += 'yes'
        else:
            text += 'no'

        temp_data['text'] = text
        return temp_data



    train_dataset = train_dataset.map(encode, batched=False, num_proc=16)

    training_args = TrainingArguments(
            # output_dir=os.path.join(args.output_dir, os.path.basename(args.model_dir), args.prompt_type),
            output_dir=output_dir,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.lr,
            weight_decay=5e-4,
            adam_beta1=0.9,
            adam_beta2=0.95,
            # fp16=True,
            bf16=True,
            num_train_epochs=args.num_train_epochs,
            # logging_dir=os.path.join(args.output_dir, '/logs'),
            logging_strategy="steps",
            logging_steps=10,
            save_strategy="epoch",
            report_to='none',
            deepspeed=args.deepspeed
        )
    if args.lora == 'no':
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            dataset_text_field="text",
            max_seq_length=2048,
            tokenizer=tokenizer
        )
    elif args.lora == 'yes':
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            dataset_text_field="text",
            max_seq_length=2048,
            peft_config=peft_config,
            tokenizer=tokenizer
        )
    trainer.train()

if __name__ == "__main__":
    main()

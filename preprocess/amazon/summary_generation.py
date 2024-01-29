import pandas as pd
import re
import random
from tqdm import tqdm
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
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
import math

def set_seed(seed):
    """_summary_

    Parameters
    ----------
    seed
        _description_
    """
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split_id', type=int, default=0)
    parser.add_argument('--model_size', type=str, default='30b')
    parser.add_argument('--output_dir', type=str, default='/code/zhengzhi/LLM4LongSeqRec/preprocessed_data/amazon', help='')
    parser.add_argument('--generation_method', type=str, default='recurrent', help='hierarchical or recurrent')
    parser.add_argument('--block_size', type=int, default=5, help='number of items in a summarization block')
    parser.add_argument('--max_block_num', type=int, default=5, help='max block number')
    parser.add_argument('--batch_size', type=int, default=1, help='')
    return parser.parse_args()

def summary_generation_recurrent(args, df, df_product2sentence):
    '''
    循环式生成summary
    '''
    if args.model_size == '30b':
        tokenizer = AutoTokenizer.from_pretrained("public/upstage/llama-30b-instruct-2048")
        model = AutoModelForCausalLM.from_pretrained(
            "public/upstage/llama-30b-instruct-2048",
            device_map="auto",
            torch_dtype=torch.float16,
            load_in_8bit=True,
            trust_remote_code=True
        )
    elif args.model_size == '13b':
        tokenizer = AutoTokenizer.from_pretrained("public/meta-llama/Llama-2-13b-hf")
        model = AutoModelForCausalLM.from_pretrained(
            "public/meta-llama/Llama-2-13b-hf",
            device_map="auto",
            torch_dtype=torch.float16,
            load_in_8bit=True,
            trust_remote_code=True
        )
    elif args.model_size == '7b':
        tokenizer = AutoTokenizer.from_pretrained("public/meta-llama/Llama-2-7b-hf")
        model = AutoModelForCausalLM.from_pretrained(
            "public/meta-llama/Llama-2-7b-hf",
            device_map="auto",
            torch_dtype=torch.float16,
            load_in_8bit=True,
            trust_remote_code=True
        )

    tokenizer.pad_token_id = tokenizer.eos_token_id
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype='auto',
        trust_remote_code=True,
        device_map="auto"
    )
    for N in range(args.max_block_num):
        print(f'block {N} generation start')
        start_time = time.time()
        block_list = [df.at[i, 'prev_items'][(N)*args.block_size: (N+1)*args.block_size] for i in range(len(df))]
        sentence_list = [[df_product2sentence[df_product2sentence['id']==item]['sentence'].item() for item in block] for block in block_list]
        # 首个block的summary生成
        if N == 0:
            instruction = '''
            Given the above historical purchase data of a user, including the titles, descriptions, and attributes of the items they have bought, craft a concise summary that captures the user's preferences, personality, and shopping habits. \n
            '''
            prompt_list = ['### User:\n' + instruction + '[historical purchase data]:\n' + '\n'.join(sentences) + '\n### Assistant:\n' for sentences in sentence_list]
        else:
            instruction = '''
            Given the following summary of a user's preferences and a list of recent purchases they have made, analyze whether the user's shopping preferences and habits have changed. Taking into account both the existing summary of user preferences and the user’s most recent purchase records, generate an updated concise summary that captures the user's preferences, personality, and shopping habits. Note that the newly generated summary of user preferences should be consistent with the format of the previous Preference Summary. It should serve as a complete summary of the user, rather than a separate narrative of the user's original summary and current preferences. \n
            '''
            prompt_list = ['### User:\n' + instruction + '[Previous Preference Summary]:\n' + df.at[i, f'summary_{N-1}'] + '\n[Recent purchase data]:\n' + '\n'.join(sentence_list[i]) + '\n### Assistant:\n' for i in range(len(sentence_list))]
        if args.model_size == '30b':
            summary_list = [item[0]['generated_text'] for item in pipeline(
                    prompt_list,
                    max_length=2048,
                    do_sample=True,
                    num_return_sequences=1,
                    eos_token_id=tokenizer.eos_token_id,
                    return_full_text=False,
                    batch_size=args.batch_size
                )]
        else:
            summary_list = [item[0]['generated_text'] for item in pipeline(
                    prompt_list,
                    max_new_tokens=150,
                    do_sample=True,
                    num_return_sequences=1,
                    eos_token_id=tokenizer.eos_token_id,
                    return_full_text=False,
                    batch_size=args.batch_size
                )]
        df[f'summary_{N}'] = summary_list
        end_time = time.time()
        elapsed_time_seconds = end_time - start_time
        elapsed_time_minutes = elapsed_time_seconds / 60
        print(f"block {N} generation time: {elapsed_time_minutes:.2f} minutes")
    df['final_summary'] = [df.at[i, f'summary_{(df.at[i, "item_count"] - 1) // 5}'] for i in range(len(df))]
    return df

def summary_generation_hierarchical(args, df, df_product2sentence):
    '''
    循环式生成summary
    '''
    if args.model_size == '30b':
        tokenizer = AutoTokenizer.from_pretrained("public/upstage/llama-30b-instruct-2048")
        model = AutoModelForCausalLM.from_pretrained(
            "public/upstage/llama-30b-instruct-2048",
            device_map="auto",
            torch_dtype=torch.float16,
            load_in_8bit=True,
            trust_remote_code=True
        )
    elif args.model_size == '13b':
        tokenizer = AutoTokenizer.from_pretrained("public/meta-llama/Llama-2-13b-hf")
        model = AutoModelForCausalLM.from_pretrained(
            "public/meta-llama/Llama-2-13b-hf",
            device_map="auto",
            torch_dtype=torch.float16,
            load_in_8bit=True,
            trust_remote_code=True
        )
    elif args.model_size == '7b':
        tokenizer = AutoTokenizer.from_pretrained("public/meta-llama/Llama-2-7b-hf")
        model = AutoModelForCausalLM.from_pretrained(
            "public/meta-llama/Llama-2-7b-hf",
            device_map="auto",
            torch_dtype=torch.float16,
            load_in_8bit=True,
            trust_remote_code=True
        )

    tokenizer.pad_token_id = tokenizer.eos_token_id
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype='auto',
        trust_remote_code=True,
        device_map="auto"
    )
    list_dict = {}
    for i in range(args.max_block_num):
        list_dict[f'summary_list_{i}'] = []
    list_dict['final_summary_list'] = []

    start_time = time.time()  # 开始时间
    # DataFrame的总行数
    total_rows = len(df)

    for i in range(len(df)):
        prev_items = df.at[i, 'prev_items']
        N = math.ceil(len(prev_items) / args.block_size)
        block_summary_list = []
        # 对每一个块提取summary
        for n in range(N):
            block = prev_items[(n)*args.block_size: (n+1)*args.block_size]
            sentences = [df_product2sentence[df_product2sentence['id']==item]['sentence'].item() for item in block]
            instruction = '''
            Given the above historical purchase data of a user, including the titles, descriptions, and attributes of the items they have bought, craft a concise summary that captures the user's preferences, personality, and shopping habits. \n
            '''
            prompt = '### User:\n' + instruction + '[historical purchase data]:\n' + '\n'.join(sentences) + '\n### Assistant:\n'
            if args.model_size == '30b':
                summary = pipeline(
                    prompt,
                    max_length=2048,
                    do_sample=True,
                    num_return_sequences=1,
                    eos_token_id=tokenizer.eos_token_id,
                    return_full_text=False,
                    batch_size=args.batch_size
                )[0]['generated_text']
            else:
                summary = pipeline(
                    prompt,
                    max_new_tokens=150,
                    do_sample=True,
                    num_return_sequences=1,
                    eos_token_id=tokenizer.eos_token_id,
                    return_full_text=False,
                    batch_size=args.batch_size
                )[0]['generated_text']

            block_summary_list.append(summary)
        
        for j in range(len(block_summary_list)):
            list_dict[f'summary_list_{j}'].append(block_summary_list[j])
        for j in range(len(block_summary_list), args.max_block_num):
            list_dict[f'summary_list_{j}'].append('')

        # 提取所有块的综合summary
        instruction = '''
        Given a series of user preferences summaries arranged in chronological order, generate a concise summary that encapsulates the user's overall preference. Note that the newly generated summary of user preferences should be consistent with the format of the given summaries.\n
        '''
        summaries = [f'[Preference Summary {j}:\n]' + block_summary_list[j] for j in range(len(block_summary_list))]
        prompt = '### User:\n' + instruction + '\n'.join(summaries) + '\n### Assistant:\n'
        if args.model_size == '30b':
            final_summary = pipeline(
                prompt,
                max_length=2048,
                do_sample=True,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id,
                return_full_text=False,
                batch_size=args.batch_size
            )[0]['generated_text']
        else:
            final_summary = pipeline(
                prompt,
                max_new_tokens=150,
                do_sample=True,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id,
                return_full_text=False,
                batch_size=args.batch_size
            )[0]['generated_text']

        list_dict['final_summary_list'].append(final_summary)

        # 每处理5行，输出预计剩余时间
        if (i + 1) % 5 == 0:
            elapsed_time = time.time() - start_time  # 已经过去的时间（单位：秒）
            rows_left = total_rows - i - 1  # 剩余的行数
            time_per_row = elapsed_time / (i + 1)  # 每行需要的平均时间（单位：秒）
            estimated_time_left = rows_left * time_per_row  # 预计剩余时间（单位：秒）
            
            # 将预计剩余时间从秒转换为分钟和秒
            estimated_minutes_left = int(estimated_time_left // 60)
            estimated_seconds_left = int(estimated_time_left % 60)
            
            print(f"已处理 {i + 1} 行，预计剩余时间：{estimated_minutes_left} 分 {estimated_seconds_left} 秒")


    for i in range(args.max_block_num):
        df[f'summary_{i}'] = list_dict[f'summary_list_{i}']
    df['final_summary'] = list_dict['final_summary_list']
    return df



if __name__ == '__main__':
    # recurrent可优化点：1. df_product2sentence先采样再检索  2. batchsize=1，用for循环进行处理，如果遇到不需要计算的情况就跳过
    args = set_args()
    set_seed(42)
    df_product2sentence = pd.read_parquet('/code/zhengzhi/LLM4LongSeqRec/preprocessed_data/amazon/product2sentence.parquet')
    df = pd.read_parquet(f'/code/zhengzhi/LLM4LongSeqRec/preprocessed_data/amazon/sessions_splits/split_{args.split_id}.parquet')
    # if args.generation_method == 'recurrent':
    #     df = summary_generation_recurrent(args, df, df_product2sentence)
    #     df.to_parquet(f'/code/zhengzhi/LLM4LongSeqRec/preprocessed_data/amazon/recurrent_summary_splits/split_{args.split_id}.parquet')
    #     # df = df[:32]
    #     # df = summary_generation_recurrent(args, df, df_product2sentence)
    #     # df.to_csv(f'/code/zhengzhi/LLM4LongSeqRec/summary_test/split_0_recurrent_batch_size_{args.batch_size}.csv', index=0)
    # elif args.generation_method == 'hierarchical':
    #     df = summary_generation_hierarchical(args, df, df_product2sentence)
    #     df.to_parquet(f'/code/zhengzhi/LLM4LongSeqRec/preprocessed_data/amazon/hierarchical_summary_splits/split_{args.split_id}.parquet')

    if args.model_size == '30b':
        if args.generation_method == 'recurrent':
            df = summary_generation_recurrent(args, df, df_product2sentence)
            df.to_parquet(f'/code/zhengzhi/LLM4LongSeqRec/preprocessed_data/amazon/recurrent_summary_splits/split_{args.split_id}.parquet')
            
        elif args.generation_method == 'hierarchical':
            df = summary_generation_hierarchical(args, df, df_product2sentence)
            df.to_parquet(f'/code/zhengzhi/LLM4LongSeqRec/preprocessed_data/amazon/hierarchical_summary_splits/split_{args.split_id}.parquet')
    elif args.model_size == '7b':
        if args.generation_method == 'recurrent':
            df = summary_generation_recurrent(args, df, df_product2sentence)
            df.to_parquet(f'/code/zhengzhi/LLM4LongSeqRec/preprocessed_data/amazon/recurrent_summary_splits_7b/split_{args.split_id}.parquet')
            
        elif args.generation_method == 'hierarchical':
            df = summary_generation_hierarchical(args, df, df_product2sentence)
            df.to_parquet(f'/code/zhengzhi/LLM4LongSeqRec/preprocessed_data/amazon/hierarchical_summary_splits_7b/split_{args.split_id}.parquet')
    elif args.model_size == '13b':
        if args.generation_method == 'recurrent':
            df = summary_generation_recurrent(args, df, df_product2sentence)
            df.to_parquet(f'/code/zhengzhi/LLM4LongSeqRec/preprocessed_data/amazon/recurrent_summary_splits_13b/split_{args.split_id}.parquet')
            
        elif args.generation_method == 'hierarchical':
            df = summary_generation_hierarchical(args, df, df_product2sentence)
            df.to_parquet(f'/code/zhengzhi/LLM4LongSeqRec/preprocessed_data/amazon/hierarchical_summary_splits_13b/split_{args.split_id}.parquet')
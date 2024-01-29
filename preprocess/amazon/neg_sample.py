import pandas as pd
import re
import random
from tqdm import tqdm
import os
from transformers.pipelines.pt_utils import KeyDataset

# 前面的build dataset过程中没有进行负采样，导致后续的训练和测试过程可能出现负采样结果不统一的问题。
# 为此，增加统一负采样过程，为每一个split中的每一个正样本统一负采样20个负样本，并与已有数据合并。
# 该方法为临时方法，在处理新数据集时应注意此问题，将负采样过程统一写到最开始的build_dataset中

df_products = pd.read_parquet('/code/zhengzhi/LLM4LongSeqRec/preprocessed_data/amazon/products.parquet')
df_product2sentence = pd.read_parquet('/code/zhengzhi/LLM4LongSeqRec/preprocessed_data/amazon/product2sentence.parquet')
df_sess = pd.read_parquet('/code/zhengzhi/LLM4LongSeqRec/preprocessed_data/amazon/sessions.parquet')

# 对已有的切片进行处理
split_size = 200
num_splits = len(df_sess) // split_size
for i in tqdm(range(num_splits)):
    split_df = pd.read_parquet(f'/code/zhengzhi/LLM4LongSeqRec/preprocessed_data/amazon/sessions_splits/split_{i}.parquet')
    neg_items_list = []
    for j in range(len(split_df)):
        neg_items = list(df_products['id'].sample(n=20, random_state=1000*i+j))
        neg_items_list.append(neg_items)
    split_df['neg_items'] = neg_items_list
    split_df.to_parquet(f'/code/zhengzhi/LLM4LongSeqRec/preprocessed_data/amazon/sessions_splits/split_{i}.parquet')

# 对recurrent侧进行处理
for i in tqdm(range(60)):
    split_df = pd.read_parquet(f'/code/zhengzhi/LLM4LongSeqRec/preprocessed_data/amazon/sessions_splits/split_{i}.parquet')
    recurrent_split_df = pd.read_parquet(f'/code/zhengzhi/LLM4LongSeqRec/preprocessed_data/amazon/recurrent_summary_splits/split_{i}.parquet')
    recurrent_split_df['neg_items'] = split_df['neg_items']
    recurrent_split_df.to_parquet(f'/code/zhengzhi/LLM4LongSeqRec/preprocessed_data/amazon/recurrent_summary_splits/split_{i}.parquet')
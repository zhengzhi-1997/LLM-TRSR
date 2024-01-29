import pandas as pd
import re
import random
from tqdm import tqdm
import os
from transformers.pipelines.pt_utils import KeyDataset
from sklearn.utils import shuffle

df_sess = pd.read_csv('/code/zhengzhi/LLM4LongSeqRec/raw_data/MIND/MINDlarge_train/behaviors.tsv', sep='\t', header=None, 
                        names=['Impression ID', 'User ID', 'Time', 'History', 'Impressions'])
df_products = pd.read_csv('/code/zhengzhi/LLM4LongSeqRec/raw_data/MIND/MINDlarge_train/news.tsv', sep='\t', header=None, 
                        names=['id', 'Category', 'SubCategory', 'Title', 'Abstract', 'URL', 'Title Entities', 'Abstract Entities']) 
df_sess = df_sess.dropna(subset=['History']).reset_index(drop=True)
# 删除重复数据
df_sess = df_sess.drop_duplicates(subset=['User ID'], keep='first').reset_index(drop=True)
# 转化history为字符串
df_sess['History'] = df_sess['History'].apply(lambda x: x.split(' '))
# 统计history列表长度，取长度大于等于11，小于等于26
df_sess['history_count'] = df_sess['History'].apply(lambda x: len(x))
df_sess = df_sess[(df_sess['history_count'] >= 11) & (df_sess['history_count'] <= 26)].reset_index(drop=True)
# 与amazon数据格式对齐
df_sess['next_item'] = df_sess['History'].apply(lambda x: x[-1])
df_sess['prev_items'] = df_sess['History'].apply(lambda x: x[:-1])
df_sess['item_count'] = df_sess['prev_items'].apply(lambda x: len(x))
# 最多10w数据
df_sess = shuffle(df_sess)
df_sess = df_sess[:100000]
# 进行负采样
neg_items_list = []
for i in range(len(df_sess)):
    neg_items = list(df_products['id'].sample(n=20, random_state=i))
    neg_items_list.append(neg_items)
df_sess['neg_items'] = neg_items_list
# 只保留必要的列
df_sess = df_sess[['prev_items', 'next_item', 'item_count', 'neg_items']]
# 将每个item用一句话来描述，构建item2sentence的对应表
def generate_sentence(row):
    # columns_to_include = ['id', 'Category', 'SubCategory', 'Title', 'Abstract']
    columns_to_include = ['id', 'Category', 'SubCategory', 'Title']
    # 为每个选择的列生成 "column_name: value" 格式的字符串
    items = [f"{col}: {row[col]}" for col in columns_to_include if pd.notna(row[col])]
    result_string = '; '.join(items)
    return result_string

df_products['sentence'] = df_products.apply(generate_sentence, axis=1)
df_product2sentence = df_products[['id', 'sentence']]

# 保存数据集
df_products.to_parquet('/code/zhengzhi/LLM4LongSeqRec/preprocessed_data/MIND/products.parquet')
df_product2sentence.to_parquet('/code/zhengzhi/LLM4LongSeqRec/preprocessed_data/MIND/product2sentence.parquet')
df_sess.to_parquet('/code/zhengzhi/LLM4LongSeqRec/preprocessed_data/MIND/sessions.parquet')

# 切片，每个split保存200行数据
split_size = 200
num_splits = len(df_sess) // split_size
print(num_splits)
for i in tqdm(range(num_splits)):
    start_idx = i * split_size
    end_idx = (i + 1) * split_size
    # 获取当前切片
    split_df = df_sess.iloc[start_idx:end_idx].reset_index(drop=True)
    # 保存数据
    split_df.to_parquet(f'/code/zhengzhi/LLM4LongSeqRec/preprocessed_data/MIND/sessions_splits/split_{i}.parquet')
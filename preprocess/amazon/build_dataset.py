import pandas as pd
import re
import random
from tqdm import tqdm
import os
from transformers.pipelines.pt_utils import KeyDataset

df_products = pd.read_csv('/code/zhengzhi/LLM4LongSeqRec/raw_data/Amazon-M2/training/products_train.csv')
df_products = df_products.rename(columns={'desc': 'description'})
df_sess = pd.read_csv('/code/zhengzhi/LLM4LongSeqRec/raw_data/Amazon-M2/training/sessions_train.csv')

def str_to_list(s):
    return re.findall(r"'(.*?)'", s)

df_sess['prev_items'] = df_sess['prev_items'].apply(str_to_list)

# 只取UK区
df_products = df_products[df_products['locale']=='UK'].reset_index(drop=True)
df_sess = df_sess[df_sess['locale']=='UK'].reset_index(drop=True)
# 统计item列表长度
df_sess['item_count'] = df_sess['prev_items'].apply(lambda x: len(x))
# 筛选序列长度，训练集中序列长度需大于等于10，小于等于25
df_sess = df_sess[(df_sess['item_count'] >= 10) & (df_sess['item_count'] <= 25)].reset_index(drop=True)
# 删除不必要的列
df_products = df_products.drop(columns=['locale'])
df_sess = df_sess.drop(columns=['locale'])
# 将每个item用一句话来描述，构建item2sentence的对应表
def generate_sentence(row):
    columns_to_include = ['id', 'title', 'price', 'brand', 'color', 'size', 'model', 'material', 'author', 'description']
    # 为每个选择的列生成 "column_name: value" 格式的字符串
    items = [f"{col}: {row[col]}" for col in columns_to_include if pd.notna(row[col])]
    result_string = '; '.join(items)
    return result_string

df_products['sentence'] = df_products.apply(generate_sentence, axis=1)
df_product2sentence = df_products[['id', 'sentence']]

# 保存数据集
df_products.to_parquet('/code/zhengzhi/LLM4LongSeqRec/preprocessed_data/amazon/products.parquet')
df_product2sentence.to_parquet('/code/zhengzhi/LLM4LongSeqRec/preprocessed_data/amazon/product2sentence.parquet')
df_sess.to_parquet('/code/zhengzhi/LLM4LongSeqRec/preprocessed_data/amazon/sessions.parquet')

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
    split_df.to_parquet(f'/code/zhengzhi/LLM4LongSeqRec/preprocessed_data/amazon/sessions_splits/split_{i}.parquet')



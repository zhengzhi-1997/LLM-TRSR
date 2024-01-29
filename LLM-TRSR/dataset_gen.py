import pandas as pd
from tqdm import tqdm


def preprocess(df, df_product, df_type):
    summary_list, recent_five_items_list, all_history_items_list, next_item_list, label_list = [], [], [], [], []
    for i in tqdm(range(len(df))):
        prev_items, neg_items, next_item, summary = df.at[i, 'prev_items'], df.at[i, 'neg_items'], df.at[i, 'next_item'], df.at[i, 'final_summary']
        # 将最近点击的5个item和所有item转化为字符串
        recent_five_items = '\n'.join([df_product[df_product['id']==item]['sentence'].item() for item in prev_items[-5:]])
        all_history_items = '\n'.join([df_product[df_product['id']==item]['sentence'].item() for item in prev_items])
        # 将正负样本转化为字符串并append到列表中，train数据中正负样本比例为1：1，val、test data中正负比例为1:20
        # 处理正样本
        pos_item = df_product[df_product['id']==next_item]['sentence'].item()
        summary_list.append(summary)
        recent_five_items_list.append(recent_five_items)
        all_history_items_list.append(all_history_items)
        next_item_list.append(pos_item)
        label_list.append(1)
        # 增加负样本
        neg_num = None
        if df_type == 'train':
            neg_num = 1
        else:
            neg_num = 20
        for j in range(neg_num):
            neg_item = df_product[df_product['id']==neg_items[j]]['sentence'].item()
            summary_list.append(summary)
            recent_five_items_list.append(recent_five_items)
            all_history_items_list.append(all_history_items)
            next_item_list.append(neg_item)
            label_list.append(0)
    new_df = pd.DataFrame()
    new_df['summary'], new_df['recent_five_items'], new_df['all_history_items'], new_df['next_item'], new_df['label'] = \
        summary_list, recent_five_items_list, all_history_items_list, next_item_list, label_list
    return new_df

dataset_list = ['MIND']
type_list = ['hierarchical', 'recurrent']
for dataset in dataset_list:
    for type in type_list:
        df_product = pd.read_parquet(f'/code/zhengzhi/LLM4LongSeqRec/preprocessed_data/{dataset}/product2sentence.parquet')
        # hierarchical 12k
        df_list = [
            pd.read_parquet(f'/code/zhengzhi/LLM4LongSeqRec/preprocessed_data/{dataset}/{type}_summary_splits_7b/split_{i}.parquet') for i in range(15)
        ]
        df = pd.concat(df_list, ignore_index=True)
        # 从prev_items列提取所有的item id
        prev_items_set = set(item for sublist in df['prev_items'] for item in sublist)
        # 从next_item列提取所有的item id
        next_items_set = set(df['next_item'])
        # 从neg_items列提取所有的item id
        neg_items_set = set(item for sublist in df['neg_items'] for item in sublist)
        # 合并三个set
        all_items = prev_items_set.union(next_items_set).union(neg_items_set)
        df_product = df_product[df_product['id'].isin(all_items)]

        # df_train = df[:10000].reset_index(drop=True)
        # df_val = df[10000: 11000].reset_index(drop=True)
        # df_test = df[11000:].reset_index(drop=True)
        df_train = df[:2400].reset_index(drop=True)
        df_val = df[2400: 2700].reset_index(drop=True)
        df_test = df[2700:].reset_index(drop=True)

        df_train = preprocess(df_train, df_product, 'train')
        df_val = preprocess(df_val, df_product, 'val')
        df_test = preprocess(df_test, df_product, 'test')

        df_train.to_parquet(f'/code/zhengzhi/LLM4LongSeqRec/LLM4Rec/dataset/{dataset}/{type}_3k_7b/train.parquent')
        df_val.to_parquet(f'/code/zhengzhi/LLM4LongSeqRec/LLM4Rec/dataset/{dataset}/{type}_3k_7b/validation.parquent')
        df_test.to_parquet(f'/code/zhengzhi/LLM4LongSeqRec/LLM4Rec/dataset/{dataset}/{type}_3k_7b/test.parquent')
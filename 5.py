import pandas as pd
import numpy as np
from tqdm import tqdm
from pandarallel import pandarallel

import builtins

builtins.pd = pd  # 关键补丁


def calc_cat_match_features(row):
    user_cats = row['user_pref_cats']
    seller_cats = row['seller_cats']
    # 修正：如果不是set就转成空集
    if not isinstance(user_cats, set):
        user_cats = set()
    if not isinstance(seller_cats, set):
        seller_cats = set()
    inter = user_cats & seller_cats
    union = user_cats | seller_cats
    return pd.Series({
        'cat_overlap_cnt': len(inter),
        'cat_overlap_ratio_user': len(inter) / len(user_cats) if len(user_cats) else 0,
        'cat_overlap_ratio_seller': len(inter) / len(seller_cats) if len(seller_cats) else 0,
        'cat_jaccard_sim': len(inter) / len(union) if len(union) else 0,
        'is_user_maincat_in_seller': int(row['user_main_cat'] in seller_cats) if row['user_main_cat'] else 0,
        'seller_maincat_in_user_prefs': int(row['seller_main_cat'] in user_cats) if row['seller_main_cat'] else 0,
    })

if __name__ == '__main__':
    pandarallel.initialize(progress_bar=False)  # 关闭pandarallel自带进度条
    builtins.pd = pd  # 关键补丁

    print(0)
    tqdm.pandas()
    print(1)
    # ========== 1. 读取数据 ==========
    log_df = pd.read_csv('user_log_format1.csv')
    user_info = pd.read_csv('user_info_format1.csv')
    print(2)

    # ========== 2. 用户-商家对 ==========
    user_seller = log_df[['user_id', 'seller_id']].drop_duplicates()
    print(3)

    # ========== 3. 用户画像特征 ==========
    user_info['is_gender_known'] = user_info['gender'].apply(lambda x: 1 if x in [0, 1] else 0)
    user_info['is_age_known'] = user_info['age_range'].apply(lambda x: 0 if pd.isnull(x) else 1)
    user_info = pd.get_dummies(user_info, columns=['gender', 'age_range'], dummy_na=True)
    print(4)

    # ========== 4. 用户全局行为特征 ==========
    user_total_actions = log_df.groupby('user_id')['action_type'].value_counts().unstack().fillna(0)
    user_total_actions.columns = [f'user_action_type_{int(x)}_cnt' for x in user_total_actions.columns]
    user_total_actions['user_total_action_cnt'] = user_total_actions.sum(axis=1)

    user_total_seller = log_df.groupby('user_id')['seller_id'].nunique().rename('user_unique_seller_cnt')
    user_total_cat = log_df.groupby('user_id')['cat_id'].nunique().rename('user_unique_cat_cnt')
    user_total_brand = log_df.groupby('user_id')['brand_id'].nunique().rename('user_unique_brand_cnt')

    user_pref_all = pd.concat([user_total_actions, user_total_seller, user_total_cat, user_total_brand],
                              axis=1).reset_index()

    print(5)
    # ========== 5. 用户-商家行为统计 ==========
    behavior_types = [0, 1, 2, 3]
    agg_dict = {'item_id': 'count'}
    for act in behavior_types:
        log_df[f'action_{act}'] = (log_df['action_type'] == act).astype(int)
        agg_dict[f'action_{act}'] = 'sum'
    user_seller_actions = log_df.groupby(['user_id', 'seller_id']).agg(agg_dict).rename(
        columns={'item_id': 'interaction_cnt'}
    ).reset_index()

    # 统计独立品类/品牌数
    user_seller_catcnt = log_df.groupby(['user_id', 'seller_id'])['cat_id'].nunique().reset_index().rename(
        columns={'cat_id': 'cat_cnt'})
    user_seller_brandcnt = log_df.groupby(['user_id', 'seller_id'])['brand_id'].nunique().reset_index().rename(
        columns={'brand_id': 'brand_cnt'})
    # 最近/最早交互时间
    user_seller_time = log_df.groupby(['user_id', 'seller_id'])['time_stamp'].agg(['min', 'max']).reset_index()
    user_seller_time.columns = ['user_id', 'seller_id', 'first_action_time', 'last_action_time']

    user_seller_feat = user_seller.merge(user_seller_actions, on=['user_id', 'seller_id'], how='left')
    user_seller_feat = user_seller_feat.merge(user_seller_catcnt, on=['user_id', 'seller_id'], how='left')
    user_seller_feat = user_seller_feat.merge(user_seller_brandcnt, on=['user_id', 'seller_id'], how='left')
    user_seller_feat = user_seller_feat.merge(user_seller_time, on=['user_id', 'seller_id'], how='left')

    print(6)
    # ========== 6. 用户-商家品类契合特征 ==========
    user_cat_counts = log_df.groupby(['user_id', 'cat_id']).size().reset_index(name='action_count')
    user_pref_cats = user_cat_counts[user_cat_counts['action_count'] >= 2].groupby('user_id')['cat_id'].apply(
        set).reset_index()
    user_pref_cats.columns = ['user_id', 'user_pref_cats']

    user_main_cat = user_cat_counts.sort_values(['user_id', 'action_count'], ascending=[True, False])
    user_main_cat = user_main_cat.groupby('user_id').first().reset_index()[['user_id', 'cat_id']]
    user_main_cat.columns = ['user_id', 'user_main_cat']

    seller_cats = log_df.groupby('seller_id')['cat_id'].apply(set).reset_index()
    seller_cats.columns = ['seller_id', 'seller_cats']

    seller_cat_counts = log_df.groupby(['seller_id', 'cat_id']).size().reset_index(name='sell_count')
    seller_main_cat = seller_cat_counts.sort_values(['seller_id', 'sell_count'], ascending=[True, False])
    seller_main_cat = seller_main_cat.groupby('seller_id').first().reset_index()[['seller_id', 'cat_id']]
    seller_main_cat.columns = ['seller_id', 'seller_main_cat']

    user_seller_feat = user_seller_feat.merge(user_pref_cats, on='user_id', how='left')
    user_seller_feat = user_seller_feat.merge(user_main_cat, on='user_id', how='left')
    user_seller_feat = user_seller_feat.merge(seller_cats, on='seller_id', how='left')
    user_seller_feat = user_seller_feat.merge(seller_main_cat, on='seller_id', how='left')

    print("user_seller_feat中user_id=16的有：")
    print(user_seller_feat[user_seller_feat['user_id'] == 16])

    # ========== 分批处理 ==========
    ### pandarallel.initialize(progress_bar=True)
    batch_size = 1000_000
    total = len(user_seller_feat)
    part_files = []
    num_batches = (total + batch_size - 1) // batch_size
    print("准备分批前user_id=16的行：", user_seller_feat[user_seller_feat['user_id'] == 16])

    print(f"共需分批：{num_batches} 批")
    for i in tqdm(range(0, total, batch_size), desc="分批生成特征", unit="batch"):
        batch = user_seller_feat.iloc[i:i + batch_size].copy()
        batch['user_pref_cats'] = batch['user_pref_cats'].apply(lambda x: x if isinstance(x, set) else set())
        batch['seller_cats'] = batch['seller_cats'].apply(lambda x: x if isinstance(x, set) else set())
        batch_cat_match = batch.parallel_apply(calc_cat_match_features, axis=1)
        batch = pd.concat([batch, batch_cat_match], axis=1)
        part_file = f'user_seller_features_part_{i // batch_size}.csv'
        batch.to_csv(part_file, index=False)
        part_files.append(part_file)

    import pandas as pd

    all_batches = [pd.read_csv(f) for f in part_files]
    all_feat = pd.concat(all_batches, axis=0, ignore_index=True)
    print("分批拼接后user_id=16的有：")
    print(all_feat[all_feat['user_id'] == 16])

    # ========== 7. 合并批次、拼接剩余特征 ==========
    print("正在拼接所有批次...")
    all_batches = []
    for f in tqdm(part_files, desc="拼接分块", unit="part"):
        all_batches.append(pd.read_csv(f))
    user_seller_feat_full = pd.concat(all_batches, axis=0, ignore_index=True)

    print("user_seller_feat_full中user_id=16的有：")
    print(user_seller_feat_full[user_seller_feat_full['user_id'] == 16])

    # 合并用户画像与全局特征
    user_seller_feat_full = user_seller_feat_full.merge(user_info, on='user_id', how='left')
    user_seller_feat_full = user_seller_feat_full.merge(user_pref_all, on='user_id', how='left')

    # 清理临时set/主cat列
    user_seller_feat_full.drop(columns=['user_pref_cats', 'seller_cats', 'user_main_cat', 'seller_main_cat'],
                               inplace=True)
    print(8)
    print("保存前总行数：", user_seller_feat_full.shape[0])
    print("user_id=16是否在？", (user_seller_feat_full['user_id'] == 16).any())
    print(user_seller_feat_full[user_seller_feat_full['user_id'] == 16])

    # ========== 8. 保存 ==========
    user_seller_feat_full.to_csv('user_seller_features333.csv', index=False)
    print('全部特征提取完成，已保存至 user_seller_features.csv，特征维度：', user_seller_feat_full.shape[1] - 2)

    df_check = pd.read_csv('user_seller_features222.csv')
    print("保存后 user_id=16 是否在？", (df_check['user_id'] == 16).any())
    print(df_check[df_check['user_id'] == 16])
    print("保存后总行数：", df_check.shape[0])


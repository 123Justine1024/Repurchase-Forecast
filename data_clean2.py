from sklearn.preprocessing import MinMaxScaler
import pandas as pd

df = pd.read_csv('user_seller_feat_full_cleaned.csv')


# 2. 不需要归一化的字段（主键、标签、one-hot、0/1特征、已归一化特征等）
not_norm_cols = [
    'user_id', 'seller_id', 'label',
    'cat_overlap_ratio_user', 'cat_overlap_ratio_seller', 'cat_jaccard_sim',
    'is_user_maincat_in_seller', 'seller_maincat_in_user_prefs', 'is_age_known'

]
# 自动筛除one-hot
not_norm_cols += [col for col in df.columns if 'gender_' in col or 'age_range_' in col]

# 3. 归一化剩下的连续型特征
norm_cols = [col for col in df.columns if col not in not_norm_cols]

scaler = MinMaxScaler()
df[norm_cols] = scaler.fit_transform(df[norm_cols])

# 4. 保存归一化后的结果
df.to_csv('user_seller_feat_full_cleaned_normalized.csv', index=False)
print("归一化完成，已保存为 user_seller_feat_full_cleaned_normalized.csv")

import pandas as pd

# 加载特征表（归一化+清洗后）
features = pd.read_csv('user_seller_feat_full_cleaned_normalized.csv')
features.rename(columns={'seller_id': 'merchant_id'}, inplace=True)  # 字段名统一

# 加载训练和测试集
train = pd.read_csv('train_format1.csv')  # 应该有 user_id, seller_id, label
test = pd.read_csv('test_format1.csv')    # 应该有 user_id, seller_id，无 label

# 训练集合并
train_merged = train.merge(features, on=['user_id', 'merchant_id'], how='left')

# 测试集合并
test_merged = test.merge(features, on=['user_id', 'merchant_id'], how='left')

print("合并后训练集 shape:", train_merged.shape)
print("合并后测试集 shape:", test_merged.shape)

# 检查是否有未合上的行（极少见，但建议确认）
print("训练集未匹配特征的样本数：", train_merged.isnull().any(axis=1).sum())
print("测试集未匹配特征的样本数：", test_merged.isnull().any(axis=1).sum())

train_merged.to_csv('train_with_features.csv', index=False)
test_merged.to_csv('test_with_features.csv', index=False)
print('新训练/测试集已保存，可直接用于后续建模。')

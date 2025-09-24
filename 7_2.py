from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import pandas as pd

df = pd.read_csv('train_with_features.csv')

# 你实际的R、M、L列名，举例用
R_col = 'last_action_time_gap_days'     # 最近一次行为距离样本日的天数
M_col = 'action_3'    # 总交易金额/行为活跃度等
L_col = 'first_action_time_gap_days'      # 行为跨度、活跃天数等
# 请将上面三列名替换为你的特征名

# 1. 要降维的全部数值特征（剔除主键、标签、R/M/L）
not_pca_cols = ['user_id', 'merchant_id', 'label', R_col, M_col, L_col]
pca_cols = [col for col in df.columns if col not in not_pca_cols]

X_pca_input = df[pca_cols]

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_pca_input)

# PCA降到17维
pca = PCA(n_components=17, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# 新特征名
pca_feat_names = [f'pca_{i+1}' for i in range(17)]
final_feat_names = pca_feat_names + [R_col, M_col, L_col]
# 取R、M、L原始特征
X_rml = df[[R_col, M_col, L_col]].values

# 拼接主成分和R/M/L
X_final = np.concatenate([X_pca, X_rml], axis=1)

# 构建DataFrame，保留主键和label
result = pd.DataFrame(X_final, columns=final_feat_names)
result.insert(0, 'user_id', df['user_id'].values)
result.insert(1, 'merchant_id', df['merchant_id'].values)
if 'label' in df.columns:
    result['label'] = df['label'].values

result.to_csv('train_with_features_pca20.csv', index=False)
print('降维后20维特征表已保存为 train_with_features_pca20.csv')

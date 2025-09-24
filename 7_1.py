from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix
import numpy as np
import pandas as pd

# 读取数据和特征部分略...
# 1. 读数据
train = pd.read_csv('train_with_features_pca20.csv')

# 2. 提取特征
not_feat_cols = ['user_id', 'merchant_id', 'label']
features = [col for col in train.columns if col not in not_feat_cols]
X_train = train[features]
y_train = train['label']
print('正负样本比例：', y_train.value_counts(normalize=True))
# 1. 随机过采样
ros = RandomOverSampler(random_state=42)
X_train_res, y_train_res = ros.fit_resample(X_train, y_train)
print("过采样后正负样本数：", pd.Series(y_train_res).value_counts())

# 2. 训练模型
base_clf = DecisionTreeClassifier(max_depth=2, class_weight='balanced', random_state=42)
ada_clf = AdaBoostClassifier(estimator=base_clf, n_estimators=100, random_state=42)
ada_clf.fit(X_train_res, y_train_res)

# 3. 训练集评估（可用原训练集或过采样后的训练集，通常更关注在原集上的表现）
y_pred_train = ada_clf.predict(X_train)
acc = accuracy_score(y_train, y_pred_train)
rec = recall_score(y_train, y_pred_train)
pre = precision_score(y_train, y_pred_train)
tn, fp, fn, tp = confusion_matrix(y_train, y_pred_train).ravel()
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
gmean = np.sqrt(sensitivity * specificity)
print(f"训练集 Accuracy : {acc:.4f}")
print(f"训练集 G-mean   : {gmean:.4f}")
print(f"训练集 Recall   : {rec:.4f}")
print(f"训练集 Precision: {pre:.4f}")

import pandas as pd
from sklearn.model_selection import train_test_split

train = pd.read_csv('train_with_features_pca20.csv')

# 提取特征
not_feat_cols = ['user_id', 'merchant_id', 'label']
features = [col for col in train.columns if col not in not_feat_cols]
X = train[features]
y = train['label']

# 划分 80%训练+20%测试（2/8分）
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print('训练集样本:', len(X_train), '验证集样本:', len(X_valid))
print('训练集正负比例：', y_train.value_counts(normalize=True))
print('验证集正负比例：', y_valid.value_counts(normalize=True))
from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler(random_state=42)
X_train_res, y_train_res = ros.fit_resample(X_train, y_train)
print("过采样后正负样本数：", pd.Series(y_train_res).value_counts())
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

base_clf = DecisionTreeClassifier(max_depth=2, class_weight='balanced', random_state=42)
ada_clf = AdaBoostClassifier(estimator=base_clf, n_estimators=100, random_state=42)
ada_clf.fit(X_train_res, y_train_res)
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix
import numpy as np

y_pred_valid = ada_clf.predict(X_valid)
acc = accuracy_score(y_valid, y_pred_valid)
rec = recall_score(y_valid, y_pred_valid)
pre = precision_score(y_valid, y_pred_valid)
tn, fp, fn, tp = confusion_matrix(y_valid, y_pred_valid).ravel()
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
gmean = np.sqrt(sensitivity * specificity)

print(f"验证集 Accuracy : {acc:.4f}")
print(f"验证集 G-mean   : {gmean:.4f}")
print(f"验证集 Recall   : {rec:.4f}")
print(f"验证集 Precision: {pre:.4f}")

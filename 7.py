import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, accuracy_score

# 1. 读数据
train = pd.read_csv('train_with_features_pca20.csv')
test = pd.read_csv('test_with_features.csv')

# 2. 提取特征
not_feat_cols = ['user_id', 'merchant_id', 'label']
features = [col for col in train.columns if col not in not_feat_cols]
X_train = train[features]
y_train = train['label']
print('正负样本比例：', y_train.value_counts(normalize=True))

# 3. 训练模型
base_clf = DecisionTreeClassifier(max_depth=2, random_state=42)
ada_clf = AdaBoostClassifier(estimator=base_clf,n_estimators=50, learning_rate=0.8, random_state=42)
ada_clf.fit(X_train, y_train)

# 4. 训练集评估
y_pred_train = ada_clf.predict(X_train)
y_prob_train = ada_clf.predict_proba(X_train)[:,1]
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix
import numpy as np

# 假设你已经有
# y_train（真实标签）, y_pred_train（模型预测标签）

# Accuracy
acc = accuracy_score(y_train, y_pred_train)
# Recall
rec = recall_score(y_train, y_pred_train)
# Precision
pre = precision_score(y_train, y_pred_train)
# 混淆矩阵，计算G-mean
tn, fp, fn, tp = confusion_matrix(y_train, y_pred_train).ravel()
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
gmean = np.sqrt(sensitivity * specificity)

print(f"训练集 Accuracy : {acc:.4f}")
print(f"训练集 G-mean   : {gmean:.4f}")
print(f"训练集 Recall   : {rec:.4f}")
print(f"训练集 Precision: {pre:.4f}")

print('训练集准确率:', accuracy_score(y_train, y_pred_train))
print('训练集AUC:', roc_auc_score(y_train, y_prob_train))

# 5. 测试集预测



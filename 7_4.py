import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# 1. 数据加载
df = pd.read_csv('train_with_features_pca20.csv')
not_feat_cols = ['user_id', 'merchant_id', 'label']
feature_cols = [col for col in df.columns if col not in not_feat_cols]

# 2. 特征和标签提取
X = df[feature_cols].values   # shape (N, 20)
y = df['label'].values

# 3. 划分训练集和验证集（2/8分，test_size=0.2）
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Reshape成 (样本数, 20, 1) 适合BLSTM-C
X_train_dl = X_train.reshape(-1, 20, 1)
X_valid_dl = X_valid.reshape(-1, 20, 1)
y_train_dl = y_train
y_valid_dl = y_valid

print('X_train_dl.shape:', X_train_dl.shape)
print('X_valid_dl.shape:', X_valid_dl.shape)
print('y_train_dl分布:', np.bincount(y_train_dl))
print('y_valid_dl分布:', np.bincount(y_valid_dl))

# 5. 构建 BLSTM-C 模型
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Bidirectional, LSTM, Conv1D, GlobalMaxPooling1D, Dense, Dropout

seq_len = 20      # 输入序列长度
feature_dim = 1   # 每步一个特征

inputs = Input(shape=(seq_len, feature_dim))
x = Bidirectional(LSTM(64, return_sequences=True))(inputs)
x = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(x)
x = GlobalMaxPooling1D()(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.5)(x)
outputs = Dense(1, activation='sigmoid')(x)

model = Model(inputs, outputs)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# 6. 训练模型
history = model.fit(
    X_train_dl, y_train_dl,
    epochs=10,
    batch_size=128,
    validation_data=(X_valid_dl, y_valid_dl)
)

# 7. 验证集评估
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix
y_pred_prob = model.predict(X_valid_dl)[:, 0]
y_pred = (y_pred_prob > 0.5).astype(int)

acc = accuracy_score(y_valid_dl, y_pred)
rec = recall_score(y_valid_dl, y_pred)
pre = precision_score(y_valid_dl, y_pred)
tn, fp, fn, tp = confusion_matrix(y_valid_dl, y_pred).ravel()
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
gmean = np.sqrt(sensitivity * specificity)

print(f"验证集 Accuracy : {acc:.4f}")
print(f"验证集 G-mean   : {gmean:.4f}")
print(f"验证集 Recall   : {rec:.4f}")
print(f"验证集 Precision: {pre:.4f}")

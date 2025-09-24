import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']   # 指定中文字体
plt.rcParams['axes.unicode_minus'] = False     # 正常显示负号

# 1. 读取数据集并抽样查看
user_info = pd.read_csv('user_info_format1.csv')
user_log = pd.read_csv('user_log_format1.csv')
train = pd.read_csv('train_format1.csv')
test = pd.read_csv('test_format1.csv')

print("【用户信息表样例】\n", user_info.head())
print("【用户行为日志样例】\n", user_log.head())
print("【训练集样例】\n", train.head())
print("【测试集样例】\n", test.head())

# 2. 查看数据类型和数据大小
print("\n【用户信息表info】")
print(user_info.info())
print("用户信息表行列数:", user_info.shape)

print("\n【用户行为日志表info】")
print(user_log.info())
print("用户行为日志表行列数:", user_log.shape)

print("\n【训练集info】")
print(train.info())
print("训练集行列数:", train.shape)

print("\n【测试集info】")
print(test.info())
print("测试集行列数:", test.shape)

# 3. 缺失值统计
print("\n【用户信息表缺失值统计】")
print(user_info.isnull().sum())
print("年龄缺失数量:", user_info['age_range'].isnull().sum())
print("性别缺失数量:", user_info['gender'].isnull().sum())

print("\n【用户行为日志表缺失值统计】")
print(user_log.isnull().sum())

# 4. 数据分布和正负样本分布
print("\n【训练集label分布】")
print(train['label'].value_counts())
print(11111111111111111111111)

train['label'].value_counts().plot(kind='bar')
plt.title('训练集正负样本分布')
plt.xlabel('label（0=非复购，1=复购）')
plt.ylabel('数量')
plt.show()
print(22222222222222222222222222)
# 5. 复购相关因素分析
# 5.1 店铺分析（不同店铺复购关系可视化、店铺复购分布）
merchant_stats = train.groupby('merchant_id')['label'].agg(['count', 'sum', 'mean']).reset_index()
merchant_stats.rename(columns={'sum':'repeat_buyers', 'mean':'repeat_ratio'}, inplace=True)

# 前20店铺的复购率
top20_merchants = merchant_stats.sort_values('repeat_ratio', ascending=False).head(20)
plt.figure(figsize=(10,5))
plt.bar(top20_merchants['merchant_id'].astype(str), top20_merchants['repeat_ratio'])
plt.xticks(rotation=90)
plt.title('Top20店铺复购率')
plt.xlabel('merchant_id')
plt.ylabel('复购率')
plt.show()

# 店铺复购率分布
plt.hist(merchant_stats['repeat_ratio'], bins=30)
plt.title('店铺复购率分布')
plt.xlabel('复购率')
plt.ylabel('店铺数')
plt.show()

# 5.2 用户分析（复购分布）
user_stats = train.groupby('user_id')['label'].agg(['count', 'sum', 'mean']).reset_index()
user_stats.rename(columns={'sum':'repeat_times', 'mean':'repeat_ratio'}, inplace=True)

plt.hist(user_stats['repeat_ratio'], bins=30)
plt.title('用户复购率分布')
plt.xlabel('复购率')
plt.ylabel('用户数')
plt.show()

# 5.3 用户性别分析（复购关系、分布）
train_userinfo = train.merge(user_info, on='user_id', how='left')
gender_repeat = train_userinfo.groupby('gender')['label'].mean()
gender_repeat.plot(kind='bar')
plt.title('不同性别用户复购率')
plt.xlabel('gender（0女1男2/空未知）')
plt.ylabel('复购率')
plt.show()

gender_count = train_userinfo.groupby('gender')['label'].count()
gender_count.plot(kind='bar')
plt.title('不同性别样本数')
plt.xlabel('gender')
plt.ylabel('样本数量')
plt.show()

# 5.4 用户年龄分析（复购关系、分布）
age_repeat = train_userinfo.groupby('age_range')['label'].mean()
age_repeat.plot(kind='bar')
plt.title('不同年龄段复购率')
plt.xlabel('年龄段')
plt.ylabel('复购率')
plt.show()

age_count = train_userinfo.groupby('age_range')['label'].count()
age_count.plot(kind='bar')
plt.title('不同年龄段样本数')
plt.xlabel('年龄段')
plt.ylabel('样本数量')
plt.show()

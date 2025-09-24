import pandas as pd
import numpy as np

df = pd.read_csv('user_seller_features222.csv')

# 去除重复
df.drop_duplicates(subset=['user_id', 'seller_id'], inplace=True)

# 缺失值填充（可选，0/均值/中位数/特殊值均可，这里用0）
df.fillna(0, inplace=True)

#import pandas as pd
from datetime import datetime


# 样本参考日，格式 MMDD，转换为 datetime
SAMPLE_DAY = '1111'
SAMPLE_DATE = datetime.strptime('2016' + SAMPLE_DAY, '%Y%m%d')  # 假设都用2016年

def mmdd_to_date(mmdd):
    """将mmdd格式（如'0402'或402）转为2016年日期"""
    mmdd = str(int(mmdd)).zfill(4)  # 补齐4位
    return datetime.strptime('2016'+mmdd, '%Y%m%d')

# 计算距离样本参考日的天数
for col in ['first_action_time', 'last_action_time']:
    if col in df.columns:
        df[col + '_gap_days'] = df[col].apply(lambda x: (SAMPLE_DATE - mmdd_to_date(x)).days)

# 可选：删除原始时间字段
df.drop(['first_action_time', 'last_action_time'], axis=1, inplace=True)

# 保存
df.to_csv('user_seller_feat_full_cleaned.csv', index=False)
print('已完成first/last_action_time与参考日的天数差计算，并保存。')

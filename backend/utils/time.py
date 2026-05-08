# 有关时间转换
from datetime import datetime
import pandas as pd

def transform_timestamp(timestamp : str) -> datetime:
    # 字符串转化为时间类型，精确到小时，支持多种格式
    try:
        return datetime.strptime(str(timestamp), "%m/%d/%Y %H:%M:%S")
    except ValueError:
        return pd.to_datetime(timestamp).to_pydatetime()

if __name__ == '__main__':
    print(transform_timestamp("03/23/2026 10:00:00"))
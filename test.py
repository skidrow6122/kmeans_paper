import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
import numpy as np

############# 0. 데이터 로드
ds_receipt = "boxbee_nothern_seoul_receipt_v3.xlsx"
df_receipt = pd.read_excel(ds_receipt)
df_receipt = df_receipt[['id', 'tozipcode', 'tolatitude', 'tolongitude', 'ordertype']]
df_receiptxy = df_receipt[['id','tolatitude', 'tolongitude']]
#print(df_receiptxy.info())

# 최종 결과 기록 df 선언
df_result = df_receipt[['id', 'tolatitude', 'tolongitude' ]]
#df_result.loc['result'] = ""
df_result['result'] = ""
print(df_result)


############### 1. 다른 df 로부터 특정 조건 만족 범위 필터링 해서 새 데이터 프레임으로
df_target = df_result.loc[df_result['id'] == '2c9f9c687bdc96f4017c0f860f110e4e']

############## 2.  특정 조건 만족하는 행의 특정 열값만 찾아서 바꾸기
df_target.loc[df_target.result == '', ('result')] = 1
print(df_target)


# 2개 데이터 프레임 merge
# df_result.update(df_target)
# print(df_result)


############## 3.  한줄씩 읽어가며 result 에 쓰기
def fix_missing3(df_result, df_target, id, result):
    for i in df_target.index:
        val = df_target.loc[i, result]
        id = df_target.loc[i, id]
        df_result.loc[df_result.id == id, result] = df_result.loc[df_result.id == id, result] + str(val)


fix_missing3(df_result, df_target, 'id', 'result')


print(df_result)


############### 5. 다른 df 로부터 특정 조건 만족 범위 필터링 해서 새 데이터 프레임으로 2
df_target2 = df_result.loc[df_result['id'] == '2c9f9c687bdc96f4017c0f860f110e4e']

############## 6.  특정 조건 만족하는 행의 특정 열값만 찾아서 바꾸기 2
df_target2.loc[df_target2.result == '1', ('result')] = 5

fix_missing3(df_result, df_target2, 'id', 'result')
print(df_result)
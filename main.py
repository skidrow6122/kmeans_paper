# ## 1. 라이브러리 import
# import inline as inline
# import matplotlib
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split
# from dateutil.relativedelta import relativedelta
# from datetime import *
#
# ############################
# ##### 2. 데이터 읽기 #########
# ############################
# #2-1. 서비스 데이터 읽기 : 전체 69,708 rows
# # 고객 상품(이용권)별 재결제(해지) 이력 정보
# ds_service = "train_service.csv"
# df_service = pd.read_csv(ds_service, parse_dates=['registerdate','enddate'], infer_datetime_format=True)
#
# # 데이터에 대한 전반적인 정보를 표시 dataframe을 구성하는 행과 열의 크기, 컬럼명, 컬럼을 구성하는 값의 자료형 등을 출력
# #print(df_service.info())
# # 데이터 샘플 3개 출력
# print(df_service.sample(3))
#
# #2-2. 시청 이력(train_bookmark) 데이터 읽기 : 전체 412,036 rows
# ds_bookmark = "train_bookmark.csv"
# df_bookmark = pd.read_csv(ds_bookmark, parse_dates=['dates'], infer_datetime_format=True)
#
# # 데이터에 대한 전반적인 정보를 표시 dataframe을 구성하는 행과 열의 크기, 컬럼명, 컬럼을 구성하는 값의 자료형 등을 출력
# #df_bookmark.info()
# # 데이터 샘플 3개 출력
# print(df_bookmark.sample(3))
#
# # service 파일 컬럼별 unique values 확인
# for column in df_service.columns.values.tolist():
# print(column)
# print(df_service[column].unique())
# print("")
#
# # bookmark 파일 컬럼별 unique values 확인
# for column in df_bookmark.columns.values.tolist():
# print(column)
# print(df_bookmark[column].unique())
# print("")
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


############# 1. 데이터 로드
ds_receipt = "boxbee_nothern_seoul_receipt_v3.xlsx"
df_receipt = pd.read_excel(ds_receipt)
df_receipt = df_receipt[['id', 'tozipcode', 'tolatitude', 'tolongitude', 'ordertype']]
df_receiptxy = df_receipt[['tolatitude', 'tolongitude']]
print(df_receiptxy.info())

sns.lmplot(x = 'tolatitude', y='tolongitude', data=df_receiptxy, fit_reg=False, scatter_kws={"s": 2})
plt.title("k-means plot")
plt.xlabel('tolatitude')
plt.ylabel('tolongitude')
#plt.show()

############# 2. 엘보우 메소드
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
import numpy as np

elbowX = df_receiptxy.values
inertias = []
clusterCount = []
K = range(1,11)
for k in K:
    print("{0} 개 k 검증 시작".format(k))
    kmeansModel = KMeans(n_clusters=k, random_state=21, init='k-means++').fit(elbowX)
    #kmeansModel.fit(elbowX)
    inertias.append(kmeansModel.inertia_)

## 클러스터 안에 데이터 포인트 세기
    labels = kmeansModel.predict(elbowX)
    clusterCount = np.bincount(labels)
    print("{0} 개 k 의 클러스터 내부 데이터 포인트 개수는 {1}".format(k,clusterCount))

plt.plot(K, inertias, 'bx-')
plt.title('optimal k by Elbow Method')
plt.xlabel('k')
plt.ylabel('Inertias')
plt.xlim([0, 10])
plt.ylim([0, 7])
#plt.show()



############# 3. k 확정 후 kmeans - k는 4
kmeans = KMeans(n_clusters=10)
kmeans.fit(elbowX)
kmeans.labels_
df_receiptxy['cluster_id'] = kmeans.labels_
print(df_receiptxy.info())
print(df_receiptxy)
sns.lmplot(x = 'tolatitude', y='tolongitude', data=df_receiptxy ,fit_reg=False, scatter_kws={'s':2}, hue='cluster_id')
plt.show()


# ############# 4. 클러스터링 된 데이터 export
# df_receiptxy.to_excel('C:/newPython/projects/Kmeans/KmeansCode/result.xlsx', sheet_name='new_name')
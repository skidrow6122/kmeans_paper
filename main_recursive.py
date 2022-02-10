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
print(df_receiptxy.info())

sns.lmplot(x = 'tolatitude', y='tolongitude', data=df_receiptxy, fit_reg=False, scatter_kws={"s": 2})
plt.title("k-means plot")
plt.xlabel('tolatitude')
plt.ylabel('tolongitude')
#plt.show()

# 최종 결과 기록 df 선언
df_result = df_receipt[['id', 'tolatitude', 'tolongitude' ]]
df_result['result'] = ""

######## 클러스터링 결과 merge 함수
def applyToResult (df_result, df_target, id, result):
    for i in df_target.index:
        print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ {} 번째 수행'.format(i))
        print(df_target.loc[df_target['id'] == '2c9f9c687bdc96f4017c0f860f110e4e'])
        val = df_target.loc[i, result]
        id = df_target.loc[i, id]
        print('###########################')
        df_result.loc[df_result.id == id, result] = df_result.loc[df_result.id == id, result] + str(val)



################################### 2. 리컬시브 대상 추출 함수############################
################################### 3. elbow K 찾기 함수 ###############################
################################### 재호출

def findRecursiveTarget(k, target):
    # 2 시작
    for i in k :
        df_temp = target.loc[target['result'] == k]

        # 3시작
        inertias = []
        diff = []
        tempK = range(1, 11)
        for j in tempK:
            print("{0} 개 k 검증 시작".format(k))
            kmeansModel2 = KMeans(n_clusters=j, random_state=21, init='k-means++').fit(df_temp[['tolatitude', 'tolongitude']])
            inertias.append(kmeansModel2.inertia_)

            for y in inertias :
                if y == 0 :
                    diff.append(0)
                diff.append(inertias[y-1] - inertias[y])
            tmp = max(diff)
            elbow_k = diff.index(tmp)

            # recursive clustering
            clustering(elbow_k, df_temp)
#########################################################################################


################ 1. 클러스터링####################  ############################
################ 4. 나뉘어진 클러스터 별 데이터 포인트 개수 찾아서 이상값 숫자 리턴 함수
################ 5. 이상값 기준 분기처리 하여 종료 판단 함수 ########################
def clustering (k, target) :
    KmeansModel = KMeans(n_clusters=k, random_state=21, init='k-means++').fit(target[['tolatitude', 'tolongitude']])

    # 4시작
    clusterCount = []
    dataPoints = KmeansModel.predict(target[['tolatitude', 'tolongitude']])
    clusterCount = np.bincount(dataPoints)
    print("{0} 개 k 의 클러스터 내부 데이터 포인트 개수는 {1}".format(k, clusterCount))
    invalidCount = 0

    for z in range(len(clusterCount)):
        if clusterCount[z] > 35:
            invalidCount = invalidCount + 1
    print("invalidCount 는 {0}".format(invalidCount))

    # 5시작
    if invalidCount >= clusterCount.size/2 :
        # target 에 기록 , result 에 기록
        KmeansModel.labels_
        target['result'] = KmeansModel.labels_
        print(target.loc[target['id'] == '2c9f9c687bdc96f4017c0f860f110e4e'])
        applyToResult(df_result, target, 'id', 'result')

        # 하위 리컬시브 대상 추출 해서 elbow K 찾고 재호출 함수 호출
        findRecursiveTarget(k, target)

    elif invalidCount < clusterCount.size/2 and invalidCount > 1 :
        # K +1
        k = k +1

        # 하위 리컬시브 대상 추출 해서 elbow K 찾고 재호출 함수 호출
        findRecursiveTarget(k, target)

    else :
        # 기록
        KmeansModel.labels_
        target['result'] = KmeansModel.labels_
        applyToResult(df_result, target, 'id', 'result')

#########################################################################################


# 최초 10개로 클러스터링
clustering(10, df_receiptxy[['id', 'tolatitude', 'tolongitude']])


# 최종값 출력
print(df_result.info())
print(df_result)
sns.lmplot(x = 'tolatitude', y='tolongitude', data=df_receiptxy ,fit_reg=False, scatter_kws={'s':2}, hue='result')
plt.show()



# ############# 4. 클러스터링 된 데이터 export
# df_receiptxy.to_excel('C:/newPython/projects/Kmeans/KmeansCode/result.xlsx', sheet_name='new_name')
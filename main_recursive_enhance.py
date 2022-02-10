import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import numpy as np

###########################################################
############# Section 1. Data set load ####################
###########################################################
pd.set_option('display.max_columns', 5000)
pd.set_option('mode.chained_assignment', None)
ds_receipt = "boxbee_nothern_seoul_receipt_v3.xlsx"
df_receipt = pd.read_excel(ds_receipt)
df_receipt = df_receipt[['id', 'tozipcode', 'tolatitude', 'tolongitude', 'ordertype']]
df_receiptxy = df_receipt[['id','tolatitude', 'tolongitude']]
print(df_receiptxy.info())

InitialScatter = sns.lmplot(x = 'tolatitude', y='tolongitude', data=df_receiptxy,
                            fit_reg=False, scatter_kws={"s": 2}, height=6)
InitialScatter.fig.subplots_adjust(top=.90)
InitialScatter.ax.set_title("(K-means) Initial Destination scatter chart")
plt.show()

df_result = df_receipt[['id', 'tolatitude', 'tolongitude' ]]
df_result['result'] = ""
checkPointer = 0
print('[INFO] Data load process completed successfully !!')

###########################################################
########## Section 2. function definition #################
###########################################################
# merging data points to the result data frame
def applyToResult (df_result, df_target, id, result):
    for i in df_target.index:
        val = df_target['result'][i]
        id = df_target['id'][i]
        df_result.loc[df_result.id == id, result] = df_result.loc[df_result.id == id, result] + str(val)

#find Recursive clustering target dataframe
def findRecursiveTarget(k, target, term):
# find target
    groupKrange = range(0,k)
    for i in groupKrange :
        df_temp = target.loc[target['result'] == i]
        if term == "recursive" :
            again_k = (df_temp.shape[0] // 35) + 1
            print("[INFO] Recursive 용 again_k 값을 정하기 위한 대상 값은 {0}".format(df_temp.shape[0]))
            print("[INFO] again_k value for Recursive {0}".format(again_k))
        else :
            # find elbow K
            inertias = []
            tempK = range(1, 11)
            for j in tempK:
                print("[INFO] Start to verifying for {0} k".format(j))
                diff = []
                kmeansModel2 = KMeans(n_clusters=j, random_state=21, init='k-means++').fit(df_temp[['tolatitude', 'tolongitude']])
                inertias.append(kmeansModel2.inertia_)
                elbowKrange = range(0,len(inertias))
                print(len(inertias))
                for y in elbowKrange :
                    print("[INFO] Current y value in elbowKrange is {0}".format(y))
                    if y == 0 :
                        diff.append(0)
                    else :
                        diff.append(inertias[y-1] - inertias[y])
                print(inertias)
                print(diff)
            tmp = max(diff)
            if diff.index(tmp) == 0 :
                continue
            again_k = diff.index(tmp) + 3
            print("[INFO] Derived elbow_k is {0}".format(again_k))

        # recursive clustering
        print("!!!!! [WARNING] Recursive Clustering execution !!!!!!!!!!")
        clustering(again_k, df_temp, 'recursive')

# Clustering & find the number of data points in each cluster & judgement
def clustering (k, target, term) :
    global checkPointer
    checkPointer = checkPointer +1

    print('[INFO] global checkpointer is {0}'.format(checkPointer))
    KmeansModel = KMeans(n_clusters=k, random_state=21, init='k-means++').fit(target[['tolatitude', 'tolongitude']])

    # find the number of data points in each cluster
    clusterCount = []
    dataPoints = KmeansModel.predict(target[['tolatitude', 'tolongitude']])
    clusterCount = np.bincount(dataPoints)
    print("[INFO] datapoints with in the clusters for k {0} is {1}".format(k, clusterCount))
    invalidCount = 0

    for z in range(len(clusterCount)):
        if clusterCount[z] > 35:
            invalidCount = invalidCount + 1
    print("[INFO] invalidCount is {0}".format(invalidCount))

    #judgement
    if invalidCount >= clusterCount.size/2 :
        print("[INFO] case1) over the half : merge into the result and do recursive again")
        print(invalidCount)
        print(clusterCount.size / 2)
        KmeansModel.labels_
        target['result'] = KmeansModel.labels_
        applyToResult(df_result, target, 'id', 'result')

        if checkPointer != 1 :
        # 하위 리컬시브 대상 추출 해서 그냥 recur K 를 35개 기준으로 나눠서 최적화 후 재호출 함수 호출
            findRecursiveTarget(k, target, 'recursive')
        else :
        # 하위 리컬시브 대상 추출 해서 elbow K 찾고 재호출 함수 호출
            findRecursiveTarget(k, target, 'first')

    elif invalidCount < clusterCount.size/2 and invalidCount > 1 :
        print("[INFO] case2) over the 2 : k + 1 and do recursive again")
        print(invalidCount)
        print(clusterCount.size / 2)
        k = k + 1
        if checkPointer != 1 :
            # 하위 리컬시브 대상 추출 해서 그냥 recur K 를 35개 기준으로 나눠서 최적화 하는 재호출 함수 호출
            findRecursiveTarget(k, target, 'recursive')
        else :
            # 하위 리컬시브 대상 추출 해서 elbow K 찾고 재호출 함수 호출
            findRecursiveTarget(k, target, 'first')

    else :
        print("[INFO] case3) under the 1 : merge into the result and termination")
        print(invalidCount)
        print(clusterCount.size / 2)
        KmeansModel.labels_
        target['result'] = KmeansModel.labels_
        applyToResult(df_result, target, 'id', 'result')


###########################################################
################### Section 3. Main ######################
###########################################################
# Initial K means clustering
clustering(10, df_receiptxy[['id', 'tolatitude', 'tolongitude']], 'first')


###########################################################
#### Section 4. Summary with descriptive statistics ######
###########################################################
# print final clustering result
print('[INFO] Summary of the clustering result')
print(df_result.info())
print(df_result)
FinalScatter = sns.lmplot(x = 'tolatitude', y='tolongitude', data=df_result,
                          fit_reg=False, scatter_kws={'s':2}, hue='result', height=6)
FinalScatter.fig.subplots_adjust(top=.90)
FinalScatter.ax.set_title("(K-means) Final Clustered scatter chart")
plt.show()

print(df_result['result'].unique())
print(len(df_result['result'].unique()))

# Analyize by Silhoutte coeff and descriptive statistics
# METRIC 1) Silhoutte coefficient by each TOTAL data point
df_result_silhoutte = df_result
df_resultFeature = df_result_silhoutte[['tolatitude', 'tolongitude']]
print("[SUMMARY] K-means METRIC 1) -- Silhoutte coefficient by each TOTAL data point")
score_samples = silhouette_samples(df_resultFeature, df_result_silhoutte['result'], metric='euclidean')
print(score_samples)
df_result_silhoutte['silhouette_coeff'] = score_samples

# METRIC 2) Mean of Silhoutte coefficients by each TOTAL data point
print("[SUMMARY] K-means METRIC 2) -- Mean of Silhoutte coefficients by each TOTAL data point")
average_score = silhouette_score(df_resultFeature, df_result_silhoutte['result'])
print(average_score)

# METRIC 3) Mean of Silhoutte coefficients by each CLUSTER`s data point
print("[SUMMARY] K-means METRIC 3) -- Mean of Silhoutte coefficients by each CLUSTER`s data point")
print(df_result_silhoutte.groupby('result')['silhouette_coeff'].mean())

# METRIC 4,5,6,7) descriptive statistics for the number of data point distribution by clusters
df_resultGroupby = df_result.groupby(by=['result'], as_index=False).count()
df_resultGroupby = df_resultGroupby[['result', 'id']]
print("[SUMMARY] K-means METRIC 4) -- The number of Clusters : {0}".format(df_resultGroupby.shape[0]))
print("[SUMMARY] K-means METRIC 5) -- Average quota by drivers : {0}".format(df_resultGroupby['id'].mean(axis=0)))
print("[SUMMARY] K-means METRIC 6) -- Variance quota by drivers : {0}".format(df_resultGroupby['id'].var(axis=0)))
print("[SUMMARY] K-means METRIC 7) -- Standard Deviation by drivers : {0}".format(df_resultGroupby['id'].std(axis=0)))
sns.boxplot(x="id",data=df_resultGroupby)
plt.title('(K-means) The Quota Distribution boxplot chart')
plt.show()

df_groupBy_distribution = df_resultGroupby[['id']]
plt.plot(df_groupBy_distribution)
plt.title('(K-means) The Quota Distribution with threshold')
plt.axhline(35, 0, 200, ls='--', color='r')
plt.show()

# export to data sheet
df_result.to_excel('C:/newPython/projects/Kmeans/KmeansCode/result_Kmeans.xlsx', sheet_name='new_name')
df_resultGroupby.to_excel('C:/newPython/projects/Kmeans/KmeansCode/result_Kmeans_analize.xlsx', sheet_name='new_name')
print('[INFO] Summary process completed successfully !!')
print('[INFO] Finished Program Successfully !!')
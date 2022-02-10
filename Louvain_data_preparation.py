import math

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
#import hdbscan
from sklearn.metrics import silhouette_samples, silhouette_score
import numpy as np

###########################################################
############# Section 1. Data set load ####################l
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
InitialScatter.ax.set_title("(DBSCAN) Initial Destination scatter chart")
plt.show()

df_result = df_receipt[['id', 'tolatitude', 'tolongitude' ]]
df1 = df_result.copy()
df_result['result'] = ""

df_nodes = df_receipt[['id','tolatitude', 'tolongitude']]
df_CalculatedEdges = pd.DataFrame(columns=['source', 'sourceIdx', 'target', 'targetIdx', 'weight'])


#df_nodes.to_csv('C:/myPython/Kmeans/KmeansCode/df_node_all.csv')
df_node_1925 = pd.read_csv('df_node_1925.csv')
df_CalculatedEdges1925 = pd.DataFrame(columns=['source', 'sourceIdx', 'target', 'targetIdx', 'weight'])

sourceRange = range(0,len(df_node_1925.index))
targetRange = range(0,len(df_nodes.index))
# node list 와 node list 간 distance 전체 연산
for i in sourceRange:
    print('### 1925/{0} 번째 노드 처리 시작'.format(i))
    for j in targetRange:
        source = df_node_1925['id'][i]
        target = df_nodes['id'][j]
        sourceX = df_node_1925['tolatitude'][i]
        sourceY = df_node_1925['tolongitude'][i]
        targetX = df_nodes['tolatitude'][j]
        targetY = df_nodes['tolongitude'][j]
        distance = math.sqrt(math.pow(sourceX - targetX, 2) + math.pow(sourceY - targetY, 2))
        df_CalculatedEdges1925 = df_CalculatedEdges1925.append(pd.DataFrame([[source, i, target, j, distance]], columns=['source', 'sourceIdx', 'target', 'targetIdx', 'distance']), ignore_index=True)
        print ('소스 {0}에서 소스 {1}으로 가는 Euclidean distance : {2}'.format(i,j,distance))

df_CalculatedEdges1925.to_csv('C:/newPython/projects/Kmeans/KmeansCode/df_CalculatedEdges_1925.csv')
print('대상 노드 리스트는 {0}'.format(df_node_1925))
print('거리 연산 결과 리스트는 {0}'.format(df_CalculatedEdges1925))



print('[INFO] Data load process completed successfully !!')


# ###########################################################
# ########## Section 2. function definition #################
# ###########################################################
# # merging data points to the result data frame
# def applyToResult (df_result, df_target, id, result, param):
# for i in df_target.index:
# id = df_target['id'][i]
# val = df_target['result'][i]
# if param == 'noise' :
# df_result.loc[df_result.id == id, result] = str(val) + 'N'
# else :
# df_result.loc[df_result.id == id, result] = str(val)
#
# # processing clustering
# def clustering (target) :
# # Searching Elbow for eps hyper-param
# neigh = NearestNeighbors(n_neighbors=5)
# neigh.fit(target[['tolatitude', 'tolongitude']].values)
# distances, indices = neigh.kneighbors(target[['tolatitude', 'tolongitude']].values)
#
# plt.figure(figsize=(12, 6))
# plt.plot(np.sort(distances[:, 4]))
# #plt.axvline(188, 0, 2, ls='--'), plt.axhline(18, 0, 200, ls='--')
# print(np.sort(distances[:, 4])[188])
# plt.title("(DBSCAN) Initial Destination elbow graph")
# plt.show()
#
# # DBSCAN clustering
# dbscanModel = DBSCAN(eps = 0.006, min_samples = 25)
# dbscanModel.fit(target[['tolatitude', 'tolongitude']].values)
# n_clusters_ = len(set(dbscanModel.labels_)) - (1 if -1 in dbscanModel.labels_ else 0)
# n_noise_ = list(dbscanModel.labels_).count(-1)
# print("[INFO] The number of clusters {0}".format(n_clusters_)) #17 : 0~16
# print("[INFO] The number of Noises {0}".format(n_noise_))
#
# # merging into result dataframe
# target['result'] = dbscanModel.labels_
# applyToResult(df_result, target, 'id', 'result', 'first')
#
# clusterCount = []
# clusterRange = range(0, n_clusters_)
# for z in clusterRange:
# clusterCount.append(list(dbscanModel.labels_).count(z))
# print("[INFO] The number of Data points by each cluster {0}".format(clusterCount))
#
#
# ###########################################################
# ################### Section 3. Main ######################
# ###########################################################
# # Initial DBSCAN clustering
# clustering(df_receiptxy[['id', 'tolatitude', 'tolongitude']])
#
# # Noise clustering by seperating outliers
# df_noise = df_result.loc[df_result['result'] == '-1']
# df_noise['result'] = ""
# print(df_noise)
#
# NoiseScatter = sns.lmplot(x = 'tolatitude', y='tolongitude', data=df_noise,
# fit_reg=False, scatter_kws={"s": 2}, height=6)
# NoiseScatter.fig.subplots_adjust(top=.90)
# NoiseScatter.ax.set_title("(DBSCAN) Noise Destination scatter chart")
# plt.show()
#
#
# # Searching Elbow for eps hyper-param
# neigh = NearestNeighbors(n_neighbors=5)
# neigh.fit(df_noise[['tolatitude', 'tolongitude']].values)
# distances, indices = neigh.kneighbors(df_noise[['tolatitude', 'tolongitude']].values)
#
# plt.figure(figsize=(12, 6))
# plt.plot(np.sort(distances[:, 4]))
# #plt.axvline(176, 0, 2, ls='--'), #plt.axhline(18, 0, 200, ls='--')
# print(np.sort(distances[:, 4])[176])
# plt.title("(DBSCAN) Noise Destination elbow graph")
# plt.show()
#
# ### Noise DBSCAN clustering
# dbscanModel = DBSCAN(eps = 0.01, min_samples = 5) # 7 - 10개 조합
# dbscanModel.fit(df_noise[['tolatitude', 'tolongitude']].values)
# n_clusters_ = len(set(dbscanModel.labels_)) - (1 if -1 in dbscanModel.labels_ else 0)
# n_noise_ = list(dbscanModel.labels_).count(-1)
# print("[INFO] The number of clusters for noise {0}".format(n_clusters_))
# print("[INFO] The number of Noises for noise {0}".format(n_noise_))
#
# clusterCount = []
# clusterRange = range(0, n_clusters_)
# for z in clusterRange:
# clusterCount.append(list(dbscanModel.labels_).count(z))
# print("[INFO] The number of Data points by each cluster for noise {0}".format(clusterCount))
#
# ### merging into result dataframe
# df_noise['result'] = dbscanModel.labels_
# applyToResult(df_result, df_noise, 'id', 'result', 'noise')
# NoiseResultScatter = sns.lmplot(x = 'tolatitude', y='tolongitude', data=df_noise,
# fit_reg=False, scatter_kws={"s": 2}, hue='result', height=6)
# NoiseResultScatter.fig.subplots_adjust(top=.90)
# NoiseResultScatter.ax.set_title("(DBSCAN) only Noise Clustered scatter chart")
# plt.show()
# print('[INFO] Clustering process completed successfully !!')
#
#
# ###########################################################
# #### Section 4. Summary with descriptive statistics ######
# ###########################################################
# # print final clustering result
# print('[INFO] Summary of the clustering result')
# print(df_result.info())
# print(df_result)
# FinalScatter = sns.lmplot(x = 'tolatitude', y='tolongitude', data=df_result,
# fit_reg=False, scatter_kws={"s": 2}, hue='result', height=6)
# FinalScatter.fig.subplots_adjust(top=.90)
# FinalScatter.ax.set_title("(DBSCAN) Final Clustered scatter chart")
# plt.show()
#
# print(df_result['result'].unique())
# print(len(df_result['result'].unique()))
#
# # Analyize by Silhoutte coeff and descriptive statistics
# # METRIC 1) Silhoutte coefficient by each TOTAL data point
# df_result_silhoutte = df_result
# df_resultFeature = df_result_silhoutte[['tolatitude', 'tolongitude']]
# print("[SUMMARY] DBSCAN METRIC 1) -- Silhoutte coefficient by each TOTAL data point")
# score_samples = silhouette_samples(df_resultFeature, df_result_silhoutte['result'], metric='euclidean')
# print(score_samples)
# df_result_silhoutte['silhouette_coeff'] = score_samples
#
# # METRIC 2) Mean of Silhoutte coefficients by each TOTAL data point
# print("[SUMMARY] DBSCAN METRIC 2) -- Mean of Silhoutte coefficients by each TOTAL data point")
# average_score = silhouette_score(df_resultFeature, df_result_silhoutte['result'])
# print(average_score)
#
# # METRIC 3) Mean of Silhoutte coefficients by each CLUSTER`s data point
# print("[SUMMARY] DBSCAN METRIC 3) -- Mean of Silhoutte coefficients by each CLUSTER`s data point")
# print(df_result_silhoutte.groupby('result')['silhouette_coeff'].mean())
#
# # METRIC 4,5,6,7) descriptive statistics for the number of data point distribution by clusters
# df_groupBy = df_result.groupby(by=['result'], as_index=False).count()
# df_groupBy = df_groupBy[['result', 'id']]
# print("[SUMMARY] DBSCAN METRIC 4) -- The number of Clusters : {0}".format(df_groupBy.shape[0]))
# print("[SUMMARY] DBSCAN METRIC 5) -- Average quota by drivers : {0}".format(df_groupBy['id'].mean(axis=0)))
# print("[SUMMARY] DBSCAN METRIC 6) -- Variance quota by drivers : {0}".format(df_groupBy['id'].var(axis=0)))
# print("[SUMMARY] DBSCAN METRIC 7) -- Standard Deviation by drivers : {0}".format(df_groupBy['id'].std(axis=0)))
# sns.boxplot(x="id",data=df_groupBy)
# plt.title('(DBSCAN) The Quota Distribution boxplot chart')
# plt.show()
#
# df_groupBy_distribution = df_groupBy[['id']]
# plt.plot(df_groupBy_distribution)
# plt.title('(DBSCAN) The Quota Distribution with threshold')
# plt.axhline(35, 0, 200, ls='--', color='r')
# plt.show()
#
# # export to data sheet
# df_result.to_excel('C:/myPython/Kmeans/KmeansCode/dbscan_result.xlsx', sheet_name='new_name')
# print('[INFO] Summary process completed successfully !!')
# print('[INFO] Finished Program Successfully !!')
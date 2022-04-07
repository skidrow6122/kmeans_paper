import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

from sklearn.metrics import silhouette_samples, silhouette_score
import numpy as np

import networkx as nx
from networkx.algorithms import community
import community as community_louvain
import pydot
from networkx.drawing.nx_pydot import graphviz_layout





###########################################################
############# Section 1. Data set load ####################
###########################################################
pd.set_option('display.max_columns', 5000)
pd.set_option('mode.chained_assignment', None)

# 소스 로딩
ds_receipt = "boxbee_nothern_seoul_receipt_v3.xlsx"
df_receipt = pd.read_excel(ds_receipt)
df_receipt = df_receipt[['id', 'tozipcode', 'tolatitude', 'tolongitude', 'ordertype']]
df_receiptxy = df_receipt[['id','tolatitude', 'tolongitude']]
print(df_receiptxy.info())

InitialScatter = sns.lmplot(x = 'tolatitude', y='tolongitude', data=df_receiptxy,
                            fit_reg=False, scatter_kws={"s": 2}, height=6)
InitialScatter.fig.subplots_adjust(top=.90)
InitialScatter.ax.set_title("(K-means) Initial Destination scatter chart")
#plt.show()

df_result = df_receipt[['id', 'tolatitude', 'tolongitude' ]]
df_result['result'] = ""
checkPointer = 0
print('[INFO] Data load process completed successfully !!')




#
df_edges_1 = pd.read_csv("done_df_CalculatedEdges_600.csv")
#print(df_edges_1)
# df_edges_2 = pd.read_csv("done_df_CalculatedEdges_600.csv")
# df_edges_3 = pd.read_csv("done_df_CalculatedEdges_800.csv")
# df_edges_4 = pd.read_csv("done_df_CalculatedEdges_1200.csv")
# df_edges_5 = pd.read_csv("done_df_CalculatedEdges_1400.csv")
# df_edges_6 = pd.read_csv("done_df_CalculatedEdges_1600.csv")
# df_edges_7 = pd.read_csv("done_df_CalculatedEdges_1925.csv")
#
df_edge_total = df_edges_1[['sourceIdx', 'targetIdx', 'distance']]
print(df_edge_total)

#
df_receipt = df_receipt[['id', 'tozipcode', 'tolatitude', 'tolongitude', 'ordertype']]
df_receiptxy = df_receipt[['id','tolatitude', 'tolongitude']]
#print(df_receiptxy.info())



edges = df_edge_total[['sourceIdx', 'targetIdx']].values.tolist()
print(edges)
weights = [float(l) for l in df_edge_total['distance']]
print(weights)

G = nx.Graph(directed=True)
G.add_edges_from(edges)
print(weights[:1])

for cnt, a in enumerate(G.edges(data=True)):
    G.edges[(a[0],a[1])]['weight'] = weights[cnt]















# InitialScatter = sns.lmplot(x = 'tolatitude', y='tolongitude', data=df_receiptxy,
#                             fit_reg=False, scatter_kws={"s": 2}, height=6)
# InitialScatter.fig.subplots_adjust(top=.90)
# InitialScatter.ax.set_title("(DBSCAN) Initial Destination scatter chart")
# plt.show()
#
# df_result = df_receipt[['id', 'tolatitude', 'tolongitude' ]]
# df_result['result'] = ""
# print('[INFO] Data load process completed successfully !!')
#
#
###########################################################
########## Section 2. function definition #################
###########################################################
# merging data points to the result data frame
def simple_Louvain (G):
     print("simple louvain 진입")
     partition = community_louvain.best_partition(G)
     print("best partition 종료")
     #pos = graphviz_layout(G)
     pos = nx.spring_layout(G)
     print("pos 뽑기 종료")

     cmap = cm.get_cmap('viridis', max(partition.values()) +1)
     nx.draw_networkx_nodes(G, pos, partition.keys(), node_size=70, cmap=cmap, node_color =list(partition.values()))
     nx.draw_networkx_edges(G, pos, alpha=0.5)
     plt.show()



"""
     max_k_w = []
     for com in set(partition.values()):
         print("{0} 번째 처리중".format(com))
         list_nodes = [nodes for nodes in partition.keys()
                       if partition[nodes] == com]
         max_k_w = max_k_w + [list_nodes]

     node_mapping = {}
     map_v = 0
     for node in G.nodes():
         node_mapping[node] = map_v
         print("{0} 번째 노드 처리중".format(com))
         map_v += 1

     community_num_group = len(max_k_w)
     color_list_community = [[] for i in range(len(G.nodes()))]

     # color
     for i in G.nodes():
         for j in range(community_num_group):
             if i in max_k_w[j]:
                 color_list_community[node_mapping[i]] = j

     return G, pos, color_list_community, community_num_group, max_k_w
"""


###########################################################
################### Section 3. Main ######################
###########################################################
# Initial DBSCAN clustering
#clustering(df_receiptxy[['id', 'tolatitude', 'tolongitude']])

#G, pos, color_list_community, community_num_group, max_k_w = simple_Louvain(G)

simple_Louvain(G)

"""
edges = G.edges()
Feature_color_sub = color_list_community
node_size = 70

fig = plt.figure(figsize=(20,10))
im = nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=Feature_color_sub, cmap='jet', vmin=0, vmax=community_num_group, with_labels=False)
nx.draw_networkx_edges(G, pos)
nx.draw_networkx_labels(G, pos, font_size=12, font_color="black")
plt.xticks([])
plt.yticks([])
plt.colorbar(im)
plt.show(block=False)
"""

# # Noise clustering by seperating outliers
# df_noise = df_result.loc[df_result['result'] == '-1']
# df_noise['result'] = ""
# print(df_noise)
#
# NoiseScatter = sns.lmplot(x = 'tolatitude', y='tolongitude', data=df_noise,
#                             fit_reg=False, scatter_kws={"s": 2}, height=6)
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
#     clusterCount.append(list(dbscanModel.labels_).count(z))
# print("[INFO] The number of Data points by each cluster for noise {0}".format(clusterCount))
#
# ### merging into result dataframe
# df_noise['result'] = dbscanModel.labels_
# applyToResult(df_result, df_noise, 'id', 'result', 'noise')
# NoiseResultScatter = sns.lmplot(x = 'tolatitude', y='tolongitude', data=df_noise,
#                             fit_reg=False, scatter_kws={"s": 2}, hue='result', height=6)
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
#                             fit_reg=False, scatter_kws={"s": 2}, hue='result', height=6)
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
# df_resultGroupby = df_result.groupby(by=['result'], as_index=False).count()
# df_resultGroupby = df_resultGroupby[['result', 'id']]
# print("[SUMMARY] DBSCAN METRIC 4) -- The number of Clusters : {0}".format(df_resultGroupby.shape[0]))
# print("[SUMMARY] DBSCAN METRIC 5) -- Average quota by drivers : {0}".format(df_resultGroupby['id'].mean(axis=0)))
# print("[SUMMARY] DBSCAN METRIC 6) -- Variance quota by drivers : {0}".format(df_resultGroupby['id'].var(axis=0)))
# print("[SUMMARY] DBSCAN METRIC 7) -- Standard Deviation by drivers : {0}".format(df_resultGroupby['id'].std(axis=0)))
# sns.boxplot(x="id",data=df_resultGroupby)
# plt.title('(DBSCAN) The Quota Distribution boxplot chart')
# plt.show()
#
# df_groupBy_distribution = df_resultGroupby[['id']]
# plt.plot(df_groupBy_distribution)
# plt.title('(DBSCAN) The Quota Distribution with threshold')
# plt.axhline(35, 0, 200, ls='--', color='r')
# plt.show()
#
# # export to data sheet
# df_result.to_excel('C:/newPython/projects/Kmeans/KmeansCode/result_dbscan.xlsx', sheet_name='new_name')
# df_resultGroupby.to_excel('C:/newPython/projects/Kmeans/KmeansCode/result_dbscan_analize.xlsx', sheet_name='new_name')
# print('[INFO] Summary process completed successfully !!')
# print('[INFO] Finished Program Successfully !!')
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_samples, silhouette_score


###########################################################
############# Section 1. Data set load ####################
###########################################################
pd.set_option('display.max_columns', 5000)
pd.set_option('mode.chained_assignment', None)
ds_result = "zipcode_result.csv"
df_result = pd.read_csv(ds_result)
df_result = df_result[['ORDERID', 'TOLATITUDE', 'TOLONGITUDE', 'DRIVER']]

print('[INFO] Data load process completed successfully !!')


###########################################################
#### Section 4. Summary with descriptive statistics ######
###########################################################
# print final clustering result
print('[INFO] Summary of the clustering result')
print(df_result.info())
print(df_result)
finalScatter = sns.lmplot(x = 'TOLATITUDE', y='TOLONGITUDE', data=df_result,
fit_reg=False, scatter_kws={"s": 2}, hue='DRIVER' ,height=6)
finalScatter.fig.subplots_adjust(top=.90)
finalScatter.ax.set_title("(HEURISTIC) Final Assigned scatter chart")
plt.show()

print(df_result['DRIVER'].unique())
print(len(df_result['DRIVER'].unique()))

# Analyize by Silhoutte coeff and descriptive statistics
# METRIC 1) Silhoutte coefficient by each TOTAL data point
df_result_silhoutte = df_result
df_resultFeature = df_result_silhoutte[['TOLATITUDE', 'TOLONGITUDE']]
print("[SUMMARY] HEURISTIC METRIC 1) -- Silhoutte coefficient by each TOTAL data point")
score_samples = silhouette_samples(df_resultFeature, df_result_silhoutte['DRIVER'], metric='euclidean')
print(score_samples)
df_result_silhoutte['silhouette_coeff'] = score_samples

# METRIC 2) Mean of Silhoutte coefficients by each TOTAL data point
print("[SUMMARY] HEURISTIC METRIC 2) -- Mean of Silhoutte coefficients by each TOTAL data point")
average_score = silhouette_score(df_resultFeature, df_result_silhoutte['DRIVER'])
print(average_score)

# METRIC 3) Mean of Silhoutte coefficients by each CLUSTER`s data point
print("[SUMMARY] HEURISTIC METRIC 3) -- Mean of Silhoutte coefficients by each CLUSTER`s data point")
print(df_result_silhoutte.groupby('DRIVER')['silhouette_coeff'].mean())

# METRIC 4,5,6,7) descriptive statistics for the number of data point distribution by clusters
df_groupBy = df_result.groupby(by=['DRIVER'], as_index=False).count()
df_groupBy = df_groupBy[['DRIVER', 'ORDERID']]
print("[SUMMARY] HEURISTIC METRIC 4) -- The number of Clusters : {0}".format(df_groupBy.shape[0]))
print("[SUMMARY] HEURISTIC METRIC 5) -- Average quota by drivers : {0}".format(df_groupBy['ORDERID'].mean(axis=0)))
print("[SUMMARY] HEURISTIC METRIC 6) -- Variance quota by drivers : {0}".format(df_groupBy['ORDERID'].var(axis=0)))
print("[SUMMARY] HEURISTIC METRIC 7) -- Standard Deviation by drivers : {0}".format(df_groupBy['ORDERID'].std(axis=0)))
sns.boxplot(x="ORDERID",data=df_groupBy)
plt.title('(HEURISTIC) The Quota Distribution boxplot chart')
plt.show()

df_groupBy_distribution = df_groupBy[['ORDERID']]
plt.plot(df_groupBy_distribution)
plt.title('(HEURISTIC) The Quota Distribution with threshold')
plt.axhline(35, 0, 200, ls='--', color='r')
plt.show()

print('[INFO] Summary process completed successfully !!')
print('[INFO] Finished Program Successfully !!')
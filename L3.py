import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans,AgglomerativeClustering
from sklearn import preprocessing
from scipy.cluster.hierarchy import dendrogram, ward

# 数据加载
data=pd.read_csv('car_data.csv',encoding='gbk')
tran_x=data[["人均GDP","城镇人口比重", "交通工具消费价格指数",'百户拥有汽车量']]
print(tran_x)

# 规范化到 [0,1] 空间
min_max_scaler=preprocessing.MinMaxScaler()
tran_x=min_max_scaler.fit_transform(tran_x)
pd.DataFrame(tran_x).to_csv('temp.csv', index=False)
print(tran_x)

# KMeans聚类
n=3
kmeans=KMeans(n_clusters=n)
kmeans.fit(tran_x)
predict_y = kmeans.predict(tran_x)
K_result = pd.concat((data,pd.DataFrame(predict_y)),axis=1)
K_result.rename({0:u'聚类结果'},axis=1,inplace=True)
print(K_result)
K_result.to_csv('KMeans Reault.csv')

# 聚类结果可视化
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.figure()
plt.subplots_adjust(hspace=0.4, wspace=0.4)  # 设置小多图之间的间隙

plt.subplot(2, 2, 1)
for i in range(0, n):
        x = K_result[K_result['聚类结果'] == i]['人均GDP']
        y = K_result[K_result['聚类结果'] == i]['百户拥有汽车量']
        plt.scatter(x, y, alpha=0.4, label=i)
        plt.xlabel('人均GDP')
        plt.ylabel('百户拥有汽车量')

plt.subplot(2, 2, 2)
for i in range(0, n):
        x = K_result[K_result['聚类结果'] == i]['交通工具消费价格指数']
        y = K_result[K_result['聚类结果'] == i]['百户拥有汽车量']
        plt.scatter(x, y, alpha=0.4, label=i)
        plt.xlabel('交通工具消费价格指数')
        plt.ylabel('百户拥有汽车量')

plt.subplot(2, 2, 3)
for i in range(0, n):
        x = K_result[K_result['聚类结果'] == i]['人均GDP']
        y = K_result[K_result['聚类结果'] == i]['城镇人口比重']
        plt.scatter(x, y, alpha=0.4, label=i)
        plt.xlabel('人均GDP')
        plt.ylabel('城镇人口比重')

plt.subplot(2, 2, 4)
for i in range(0, n):
        x = K_result[K_result['聚类结果'] == i]['百户拥有汽车量']
        y = K_result[K_result['聚类结果'] == i]['城镇人口比重']
        plt.scatter(x, y, alpha=0.4, label=i)
        plt.xlabel('百户拥有汽车量')
        plt.ylabel('城镇人口比重')

plt.legend(loc='lower right', fontsize=6, frameon=True, fancybox=True, framealpha=0.2, borderpad=0.3,
               ncol=1, markerfirst=True, markerscale=1, numpoints=1, handlelength=3.5)
plt.savefig('Kmeans_result.png')
plt.show()

# K-Means 手肘法
sse = []
for k in range(1, 11):
	# kmeans算法
	kmeans = KMeans(n_clusters=k)
	kmeans.fit(tran_x)
	# 计算inertia簇内误差平方和
	sse.append(kmeans.inertia_)
x = range(1, 11)
plt.xlabel('K')
plt.ylabel('SSE')
plt.plot(x, sse, 'o-')
plt.savefig('sse.png')
plt.show()

# 输出每一类的城市
dic={K_result['地区'][i] : K_result['聚类结果'][i] for i in range(len(K_result['地区']))}
#print(dic)
for k in range(n):
    list = []
    for i,j in dic.items():
           if j==k:
                list.append(i)
    print(f'第{k}类为:'+str(list))

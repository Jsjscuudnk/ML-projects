import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('/Users/bunny2020/Downloads/Mall_Customers.csv')

X = data.iloc[:,[3,4]].values

import scipy.cluster.hierarchy as sch

dendogram = sch.dendrogram(sch.linkage(X , method='ward'))

plt.title('Dendograms')
plt.xlabel('Customers')
plt.ylabel('Eculidean Distance')
plt.show()

from sklearn.cluster import KMeans

wcss = []

for i in range(1 , 11):
    kmeans = KMeans(n_clusters=i , init='k-means++' , random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1,11) , wcss)
plt.xlabel('Number of ckusters')
plt.ylabel('wcss')
plt.show()

from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=5 , affinity='euclidean' , linkage='ward')
y_hc = hc.fit_predict(X)
y_hc

plt.scatter(X[y_hc == 0,0], X[y_hc == 0,1] , s = 100 , c = 'red' , label = 'cluster 1')
plt.scatter(X[y_hc == 1,0], X[y_hc == 1,1] , s = 100 , c = 'blue' , label = 'cluster 2')
plt.scatter(X[y_hc == 2,0], X[y_hc == 2,1] , s = 100 , c = 'magenta' , label = 'cluster 3')
plt.scatter(X[y_hc == 3,0], X[y_hc == 3,1] , s = 100 , c = 'cyan' , label = 'cluster 4')
plt.scatter(X[y_hc == 4,0], X[y_hc == 4,1] , s = 100 , c = 'green' , label = 'cluster 5')
plt.title('clusters of customers')
plt.xlabel('Annual income')
plt.ylabel('Spending score')
plt.legend()
plt.show()

kmeans = KMeans(n_clusters=5 , init='k-means++' , random_state=0)
y_pred = kmeans.fit_predict(X)
y_pred

plt.scatter(X[y_pred == 0,0], X[y_pred == 0,1] , s = 100 , c = 'red' ,label = 'cluster1' )
plt.scatter(X[y_pred == 1,0], X[y_pred == 1,1] , s = 100 , c = 'blue' ,label = 'cluster1' )
plt.scatter(X[y_pred == 2,0], X[y_pred == 2,1] , s = 100 , c = 'green' ,label = 'cluster1' )
plt.scatter(X[y_pred == 3,0], X[y_pred == 3,1] , s = 100 , c = 'grey' ,label = 'cluster1' )
plt.scatter(X[y_pred == 4,0], X[y_pred == 4,1] , s = 100 , c = 'cyan' ,label = 'cluster1' )
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1] , s = 100 , c = 'yellow' , label = 'centroid')

plt.title('clusters of customers')
plt.xlabel('Annual income')
plt.ylabel('Spending score')
plt.legend()
plt.show()




data['kmeans_prediction'] = kmeans.labels_
data['hirerachical_prediction'] = hc.labels_


data.to_csv('/Users/bunny2020/Downloads/Clustered_Customers.csv', index=False)






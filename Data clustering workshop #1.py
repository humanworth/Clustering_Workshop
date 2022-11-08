# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 16:00:19 2022

@author: aliab
"""

import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# create dataset
X, y = make_blobs(
   n_samples=150, n_features=2,
   centers=3, cluster_std=0.5,
   shuffle=True, random_state=0
)

# plot
plt.scatter(
   X[:, 0], X[:, 1],
   c='white', marker='o',
   edgecolor='black', s=50
)
plt.show()



from sklearn.cluster import KMeans

km = KMeans(
    n_clusters=3, init='random',
    n_init=10, max_iter=300, 
    tol=1e-04, random_state=0
)
y_km = km.fit_predict(X)

plt.scatter(
   X[:, 0], X[:, 1],
   c=y_km, marker='o',
   edgecolor='black', s=50
)
plt.show()


plt.scatter(
    X[y_km == 0, 0], X[y_km == 0, 1],
    s=50, c='lightgreen',
    marker='s', edgecolor='black',
    label='cluster 1'
)

plt.scatter(
    X[y_km == 1, 0], X[y_km == 1, 1],
    s=50, c='orange',
    marker='o', edgecolor='black',
    label='cluster 2'
)

plt.scatter(
    X[y_km == 2, 0], X[y_km == 2, 1],
    s=50, c='lightblue',
    marker='v', edgecolor='black',
    label='cluster 3'
)

# plot the centroids
plt.scatter(
    km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
    s=250, marker='*',
    c='red', edgecolor='black',
    label='centroids'
)
plt.legend(scatterpoints=1)
plt.grid()
plt.show()



##################################### Hierac=rchical clustering
from sklearn.cluster import AgglomerativeClustering
import numpy as np
clustering = AgglomerativeClustering(n_clusters=3).fit(X)

y_hierarchical=clustering.labels_
plt.scatter(
   X[:, 0], X[:, 1],
   c=y_hierarchical, marker='o',
   edgecolor='black', s=50
)
plt.show()



################################## compute metrics
plt.scatter(
   X[:, 0], X[:, 1],
   c=y, marker='o',
   edgecolor='black', s=50
)
plt.show()

from sklearn.metrics import silhouette_score
km_sil_score = silhouette_score(X, y_km, metric='euclidean')
ag_sil_score = silhouette_score(X, y_hierarchical, metric='euclidean')

print("Silhouette_score kmeans++: ",km_sil_score)
print("Silhouette_score agglomerrative: ",ag_sil_score)

from sklearn.metrics.cluster import adjusted_rand_score
km_ARI=adjusted_rand_score(y, y_km)
ag_ARI=adjusted_rand_score(y, y_hierarchical)

print("ARI kmeans++: ",km_ARI)
print("ARI agglomerative: ",ag_ARI)


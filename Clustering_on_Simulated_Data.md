import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
%matplotlib inline

plt.style.use('seaborn')

pd.set_option('precision',4)
pd.set_option('display.float_format','{:20,.4f}'.format)


```python
np.random.seed(42)

x = np.random.normal(size=50*2).reshape(50,2)
x[0:25,0] +=3
x[25:50,1] -=4
# first 25 observations have a mean shift relative to next 25 observations
```


```python
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=2, random_state=42, n_init=20)
kmeans.fit(x)
```




    KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
           n_clusters=2, n_init=20, n_jobs=None, precompute_distances='auto',
           random_state=42, tol=0.0001, verbose=0)




```python
# the cluster assigment of the 50 observations
kmeans.labels_
```




    array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0], dtype=int32)




```python
plt.scatter(x[:,0], x[:,1], c=kmeans.labels_, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=200)
plt.title('K-means Clustering with K=2', fontdict= {'size':17, 'color':'brown'})
```




    Text(0.5, 1.0, 'K-means Clustering with K=2')




![png](Clustering_on_Simulated_Data_files/Clustering_on_Simulated_Data_4_1.png)



```python
np.random.seed(4)
kmeans3 = KMeans(n_clusters=3,n_init=20)
kmeans3.fit(x)
```




    KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
           n_clusters=3, n_init=20, n_jobs=None, precompute_distances='auto',
           random_state=None, tol=0.0001, verbose=0)




```python
# clustering vector
kmeans3.labels_
```




    array([1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1,
           1, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0], dtype=int32)




```python
# cluster means
kmeans3.cluster_centers_
```




    array([[-0.09155989, -3.87287837],
           [ 2.60450418,  0.24696837],
           [ 3.27858059, -1.37217166]])




```python
#cluster size i.e. number of observations in each cluster
# With k=3, 50 observationsa are divided as following 
pd.Series(kmeans3.labels_).value_counts()
```




    0    25
    1    17
    2     8
    dtype: int64




```python
# Sum of squared distances of samples to their closest cluster center
kmeans3.inertia_
```




    62.737378097355716




```python
plt.scatter(x[:,0],x[:,1],c=kmeans3.labels_,cmap='viridis')
plt.scatter(kmeans3.cluster_centers_[:,0],kmeans3.cluster_centers_[:,1],marker='+',s=200)
plt.title('K-means Clustering with K=3',fontdict= {'size':17, 'color':'brown'})
```




    Text(0.5, 1.0, 'K-means Clustering with K=3')




![png](Clustering_on_Simulated_Data_files/Clustering_on_Simulated_Data_10_1.png)


**n_init** parameter defines number of times K-means clustering will be perfomred using multiple random assignments. And the kmeans() function will only report the best result.

Attribute **intertia_** is the total within-cluster sum of squares which we seek to minimize by performing k-means clustering. The individual within-cluster sum of squares is available in R but didn't find it in python.

Running kmeans() function with **intertia_** 1 vs 20 gives different results. Since our goal is to minimize the total within-cluster sum of squares, init=20 is a better option. 


```python
np.random.seed(3)

kmeans_1 = KMeans(n_clusters=3,random_state=42,n_init=1)
kmeans_1.fit(x)

kmeans_20 = KMeans(n_clusters=3,random_state=42,n_init=20)
kmeans_20.fit(x)
```




    KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
           n_clusters=3, n_init=20, n_jobs=None, precompute_distances='auto',
           random_state=42, tol=0.0001, verbose=0)




```python
print('intertia with n_init=1  ',kmeans_1.inertia_)
print('intertia with n_init=20 ',kmeans_20.inertia_)
```

    intertia with n_init=1   64.71464506342775
    intertia with n_init=20  62.737378097355716



```python
N_init = []
for i in range(1,15):
    kmeans = KMeans(n_clusters=3,random_state=42,n_init=i)
    kmeans.fit(x)
    N_init.append(kmeans.inertia_)
    print('inertia with n_init = {} and kmeans.inertia {}'.format(i,kmeans.inertia_))
```

    inertia with n_init = 1 and kmeans.inertia 64.71464506342775
    inertia with n_init = 2 and kmeans.inertia 64.22208350611646
    inertia with n_init = 3 and kmeans.inertia 64.22208350611646
    inertia with n_init = 4 and kmeans.inertia 64.22208350611646
    inertia with n_init = 5 and kmeans.inertia 64.22208350611646
    inertia with n_init = 6 and kmeans.inertia 64.22208350611646
    inertia with n_init = 7 and kmeans.inertia 64.22208350611646
    inertia with n_init = 8 and kmeans.inertia 63.107468591004995
    inertia with n_init = 9 and kmeans.inertia 63.107468591004995
    inertia with n_init = 10 and kmeans.inertia 63.107468591004995
    inertia with n_init = 11 and kmeans.inertia 62.737378097355716
    inertia with n_init = 12 and kmeans.inertia 62.737378097355716
    inertia with n_init = 13 and kmeans.inertia 62.737378097355716
    inertia with n_init = 14 and kmeans.inertia 62.737378097355716


Dendrogram() function is used to plot a dendrogram. The first argument in the function is to provide linkage matrix, which is done by using linkage() function, where the linkage type is passed i.e. complete, average, or single. Also, metric=euclidean as we want Euclidean distance as dissimilarity measure.


```python
from scipy.cluster import hierarchy
#from scipy.cluster.hierarchy import linkage,dendrogram
#hc_complete = linkage(x,method='complete')
#hc_average = linkage(x,method='average')
#hc_single = linkage(x,method='single')

f, axes = plt.subplots(3,1, sharex=False, sharey=False,figsize=(16,32))

# dendrogram with 'complete' linkage
dendrogram(linkage(x,method='complete'),truncate_mode='level', labels=x, leaf_font_size=5, leaf_rotation=90, ax=axes[0])

# dendrogram with 'average' linkage
dendrogram(linkage(x,method='average'), truncate_mode='level', labels=x, leaf_font_size=5, leaf_rotation=90, ax=axes[1])

# dendrogram with 'single' linkage
dendrogram(linkage(x,method='single'), truncate_mode='level', labels=x, leaf_font_size=5, leaf_rotation=90, ax=axes[2])

axes[0].set_title('Complete Linkage', fontdict= {'size':17, 'color':'brown'})
axes[1].set_title('Average Linkage', fontdict= {'size':17, 'color':'brown'})
axes[2].set_title('Single Linkage', fontdict= {'size':17, 'color':'brown'})
```

    /Users/Life_Is_Beautiful/opt/anaconda3/lib/python3.7/site-packages/matplotlib/text.py:1150: FutureWarning: elementwise comparison failed; returning scalar instead, but in the future will perform elementwise comparison
      if s != self._text:





    Text(0.5, 1.0, 'Single Linkage')




![png](Clustering_on_Simulated_Data_files/Clustering_on_Simulated_Data_16_2.png)


Since Euclidean distance was the dissimilarity measure in the linkage() function. Using Sklearn, metrics.pairwsie library as shown below, the inter-observation Euclidean distance matrix can be calcuated. In this case since x has 50 observatios, the shape for Euclidean distance matrix will be 50x50. 


```python
from sklearn.metrics.pairwise import euclidean_distances
euclidean_distances(x).shape
```




    (50, 50)




```python
from scipy.cluster.hierarchy import cut_tree

# ravel shows the output in a single 1-D array
print('Cluster label for each observation associated with a given cut of the dendrogram, where linkage is complete')
print(pd.Series(cut_tree(hc_complete,2).ravel()).value_counts())

print('\nCluster label for each observation associated with a given cut of the dendrogram, where linkage is average')
print(pd.Series(cut_tree(hc_average,2).ravel()).value_counts())

print('\nCluster label for each observation associated with a given cut of the dendrogram, where linkage is single')
print(pd.Series(cut_tree(hc_single,2).ravel()).value_counts())
```

    Cluster label for each observation associated with a given cut of the dendrogram, where linkage is complete
    1    25
    0    25
    dtype: int64
    
    Cluster label for each observation associated with a given cut of the dendrogram, where linkage is average
    1    25
    0    25
    dtype: int64
    
    Cluster label for each observation associated with a given cut of the dendrogram, where linkage is single
    0    49
    1     1
    dtype: int64


In case of linkage as average & complete the observations are correctly labeled. But in case of 'single' one point as belonging to it's own cluster. A more sensible answer is obtained when four clusters are selected, although there are still two singletons [1,3]. 


```python
print('Cluster label for each observation associated with a given cut of the dendrogram, where linkage is single')
pd.Series(cut_tree(hc_single,4).ravel()).value_counts()
```

    Cluster label for each observation associated with a given cut of the dendrogram, where linkage is single





    2    24
    0    24
    3     1
    1     1
    dtype: int64




```python
fig, ax1 = plt.subplots(1,1,figsize=(18,8))

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

dendrogram(linkage(x_scaled,method='complete'), 
           truncate_mode='level', labels=x_scaled, leaf_rotation=90, leaf_font_size=6, ax=ax1 )

plt.title('Hirerchial Cluster with Scaled Features',fontdict= {'size':17, 'color':'brown'})
```

    /Users/Life_Is_Beautiful/opt/anaconda3/lib/python3.7/site-packages/matplotlib/text.py:1150: FutureWarning: elementwise comparison failed; returning scalar instead, but in the future will perform elementwise comparison
      if s != self._text:





    Text(0.5, 1.0, 'Hirerchial Cluster with Scaled Features')




![png](Clustering_on_Simulated_Data_files/Clustering_on_Simulated_Data_22_2.png)


If you notice in this particular example, scaling features doesn't bring in any change to earlier results i.e. without scaled.


```python
print('Cluster label for each observation associated with a given cut of the dendrogram, where linkage is complete')
print(pd.Series(cut_tree(linkage(x_scaled,method='complete'),2).ravel()).value_counts())
```

    Cluster label for each observation associated with a given cut of the dendrogram, where linkage is complete
    1    25
    0    25
    dtype: int64


The following is to demonstrate metrics= correlation based distance in the linkage() function and not 'Euclidean'. Now since absolute correlation between any two observations with mesurements on two features is always 1, it makes sense to have data with at least 3 features. So 30 examples with 3 features data is generated


```python
fig, ax2 = plt.subplots(1,1,figsize=(18,10))

x = np.random.normal(size=30*3).reshape(30,3)

dendrogram(linkage(x,method='complete',metric='correlation'), labels = x, leaf_font_size=8, leaf_rotation=90, ax=ax2)

plt.title('Complete Linkage with Correlation-Based Distance',fontdict= {'size':17, 'color':'brown'})
```




    Text(0.5, 1.0, 'Complete Linkage with Correlation-Based Distance')




![png](Clustering_on_Simulated_Data_files/Clustering_on_Simulated_Data_26_1.png)


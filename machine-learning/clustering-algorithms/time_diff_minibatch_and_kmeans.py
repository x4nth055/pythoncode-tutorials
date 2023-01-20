import numpy as np
from sklearn.datasets import make_classification
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans
from matplotlib import pyplot
import timeit

X, y = make_classification(n_samples=1000, n_features=2, n_informative=2,
                           n_redundant=0, n_clusters_per_class=1, random_state=10)
# start timer for Mini Batch K-Means
t1_mkm = timeit.default_timer() 
m = MiniBatchKMeans(n_clusters=2)
m.fit(X)
p = m.predict(X)
# stop timer for Mini Batch K-Means
t2_mkm = timeit.default_timer()
# start timer for K-Means
t1_km = timeit.default_timer()
m = KMeans(n_clusters=2)
m.fit(X)
p = m.predict(X)
# stop timer for K-Means
t2_km = timeit.default_timer()
# print time difference
print("Time difference between Mini Batch K-Means and K-Means = ",
      (t2_km-t1_km)-(t2_mkm-t1_mkm))

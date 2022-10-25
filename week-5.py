import pandas as pd
from minisom import MiniSom
from sklearn.cluster import KMeans
import numpy as np

veri = pd.read_csv("datasets/airline-safety.csv")
X = veri.drop(["airline","avail_seat_km_per_week"],axis=1)

net = sps.somNet(20,20,X.values, PBC=true)

net.train(0.01,10000)

hrt = np.array((net.project(X.values)))
kmeans = KMeans(n_clusters=3, max_iter=300, random_state=0)

y_kmeans=kmeans.fit_predict(hrt)

veri["kümeler"] = kmeans.labels_
print(veri[veri["kümeler"]==0].head(5))
# K - Means Clustering
from numpy import random, array
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale


# Creating fake income/age clusters for N people in k clusters
def createClusteredData(N, k):
    random.seed(10)
    pointsPerCluster = float(N)/k
    X = []
    for i in range(k):
        incomeCentroid = random.uniform(20000, 200000)
        ageCentroid = random.uniform(20,70)
        for j in range(int(pointsPerCluster)):
            X.append([random.normal(incomeCentroid, 10000), random.normal(ageCentroid, 2.0)])
    X = array(X)
    return X

data = createClusteredData(100, 5)

model = KMeans(n_clusters=5)
# We need to scale data for normalizing
model = model.fit(scale(data))

print(model.labels_)
plt.figure(figsize=(8, 6))
plt.scatter(data[:,0], data[:,1], c=model.labels_.astype(float))
plt.show()

# Implementing SVC using linear kernel
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm


# Create fake income/age clusters for N people in k clusters
def createClusteredData(N, k):
    np.random.seed(5)
    pointsPerCluster = float(N)/k
    X = []
    y = []
    for i in range(k):
        incomeCentroid = np.random.uniform(20000,200000)
        ageCentroid = np.random.uniform(20,70)
        for j in range(int(pointsPerCluster)):
            X.append([np.random.normal(incomeCentroid, 10000.0), np.random.normal(ageCentroid, 2)])
            y.append(i)
    X = np.array(X)
    y = np.array(y)
    return X, y

(X, y) = createClusteredData(100, 5)

# Visualising clustered training data
plt.figure(figsize=(8,6))
plt.scatter(X[:,0],X[:,1],c=y.astype(np.float))
plt.show()

# Using Linear SVC to partition our graph into clusters
C = 1.0 # error penality term
svc = svm.SVC(kernel='linear', C=C).fit(X, y)

# Visualize the classification
# By setting up a dense mesh of points in the grid and classifying all of them,
# we can render the regions of each cluster as distinct colors
def plotPredictions(clf):
    xx, yy = np.meshgrid(np.arange(0,250000, 10), np.arange(10, 70, 0.5))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    plt.figure(figsize=(8, 6))
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
    plt.scatter(X[:,0], X[:,1], c=y.astype(np.float))
    plt.show()

plotPredictions(svc)

# We can use predict for a given point
print(svc.predict([[200000, 40]]))
print(svc.predict([[50000, 65]]))

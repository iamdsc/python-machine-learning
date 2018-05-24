# Implemeting Principal Component Analysis
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from pylab import *
from itertools import cycle
import matplotlib.pyplot as plt


iris = load_iris()
numSamples, numFeatures = iris.data.shape
#print(numSamples)
#print(numFeatures)
#print(list(iris.target_names))

# Distill 4D dataset down to 2D, by projecting it down to two orthogonal 4D
# vectors that make up the basis of our new 2D projection
X = iris.data # 4D data
pca = PCA(n_components=2, whiten=True).fit(X) # whiten is for normalizing data
X_pca = pca.transform(X) # 2D data

# finding the 4D vectors
#print(pca.components_)

# to analyse preserved variance
print(pca.explained_variance_ratio_)
print(sum(pca.explained_variance_ratio_))

#plotting 2D representation of our data
colors = cycle('rgb')
target_ids = range(len(iris.target_names))
figure()
for i, c, label in zip(target_ids, colors, iris.target_names):
    scatter(X_pca[iris.target == i, 0], X_pca[iris.target == i, 1], c=c, label=label)
legend()
show()

# Implementing KNN 
import pandas as pd
import numpy as np
from scipy import spatial
import operator


r_cols = ['user_id', 'movie_id', 'rating']
ratings = pd.read_csv('C:/Users/Dilpreet Singh/Desktop/Machine Learning/DataScience-Python3/ml-100k/u.data', sep='\t', names=r_cols, usecols = range(3))
#print(ratings.head())

# compute total number of ratings and average rating for every movie
movieProperties = ratings.groupby('movie_id').agg({'rating':[np.size, np.mean]})
#print(movieProperties.head())

# normalizing number of ratings in the scale from 0 - 1
movieNumRatings = pd.DataFrame(movieProperties['rating']['size'])
movieNormalizedNumRatings = movieNumRatings.apply(lambda x:(x-np.min(x))/(np.max(x)-np.min(x)))
#print(movieNormalizedNumRatings.head())

# get the genre info from u.item file
movieDict = {}
with open(r'C:/Users/Dilpreet Singh/Desktop/Machine Learning/DataScience-Python3/ml-100k/u.item') as f:
    temp = ''
    for line in f:
        fields = line.rstrip('\n').split('|')
        movieID = int(fields[0])
        name = fields[1]
        genres = fields[5:25]
        genres = map(int, genres)
        movieDict[movieID] = (name, np.array(list(genres)), movieNormalizedNumRatings.loc[movieID].get('size'), movieProperties.loc[movieID].rating.get('mean'))

# compute dist btw two movies based on how similar their genres are
# and how similar their popularity is
def ComputeDistance(a, b):
    genresA = a[1]
    genresB = b[1]
    genreDistance = spatial.distance.cosine(genresA, genresB) # cosine distance btw vectors
    popularityA = a[2]
    popularityB = b[2]
    popularityDistance = abs(popularityA - popularityB)
    return genreDistance + popularityDistance

# getting 10 nearest neighbours based on distances
def getNeighbours(movieID, k):
    distances = []
    for movie in movieDict:
        if (movie != movieID):
            dist = ComputeDistance(movieDict[movieID], movieDict[movie])
            distances.append((movie, dist))
    distances.sort(key=operator.itemgetter(1))
    neighbours = []
    for x in range(k):
        neighbours.append(distances[x][0])
    return neighbours

# predict average rating of movieID 1
k = 10
avgRating = 0
neighbours = getNeighbours(1, k)
for neighbour in neighbours:
    avgRating += movieDict[neighbour][3]
    print(movieDict[neighbour][0] + ' ' + str(movieDict[neighbour][3]))

print('Predicted Average Rating', avgRating/k)
print('Actual Average Rating', movieDict[1][3])

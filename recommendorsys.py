# Implementing Item Based CF
import pandas as pd

r_cols = ['user_id', 'movie_id', 'rating']
ratings = pd.read_csv('C:/Users/Dilpreet Singh/Desktop/Machine Learning/DataScience-Python3/ml-100k/u.data', sep='\t', names=r_cols, usecols=range(3), encoding='ISO-8859-1')
m_cols = ['movie_id', 'title']
movies = pd.read_csv('C:/Users/Dilpreet Singh/Desktop/Machine Learning/DataScience-Python3/ml-100k/u.item', sep='|', names=m_cols, usecols=range(2), encoding='ISO-8859-1')
ratings = pd.merge(movies, ratings)
userRatings = ratings.pivot_table(index=['user_id'], columns=['title'], values='rating')

# we use corr() method to compute a correlation score for every column pair in matrix
# where at least one user rated both movies-otherwise NaN will show up
# to avoid spurious results we use min_periods argument
# to throw out results where fewer than 100 users rated a given movie pair
corrmatrix = userRatings.corr(method='pearson', min_periods=100)

# eg: user with id=0, likes Star Wars and The Empire Strikes Back, but hated Gone with the Wind
myRatings = userRatings.loc[0].dropna()

# building up a list of possible recommendations based on the movies similar to the ones I rated
simCandidates = pd.Series()
for i in range(0, len(myRatings.index)):
    sims = corrmatrix[myRatings.index[i]].dropna()
    sims = sims.map(lambda x:x * myRatings[i])
    simCandidates = simCandidates.append(sims)

simCandidates.sort_values(inplace=True, ascending=False)

# use groupby() to add together the scores from movies that show up more than once,
# so they'll count more
simCandidates = simCandidates.groupby(simCandidates.index).sum()
simCandidates.sort_values(inplace=True, ascending=False)

# filtering out movies I rated
filteredSims = simCandidates.drop(myRatings.index)
print(filteredSims.head())

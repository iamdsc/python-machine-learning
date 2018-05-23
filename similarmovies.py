# finding similar movies to Star Wars for recommendation system
# Working on MovieLens dataset
import pandas as pd
import numpy as np


r_cols = ['user_id', 'movie_id', 'rating']
ratings = pd.read_csv('C:/Users/Dilpreet Singh/Desktop/Machine Learning/DataScience-Python3/ml-100k/u.data', sep='\t', names=r_cols, usecols=range(3), encoding='ISO-8859-1')
#print(ratings.head())

m_cols = ['movie_id', 'title']
movies = pd.read_csv('C:/Users/Dilpreet Singh/Desktop/Machine Learning/DataScience-Python3/ml-100k/u.item', sep='|', names=m_cols, usecols=range(2), encoding='ISO-8859-1')
#print(movies.head())

# combine the two dfs to a new ratings df
ratings = pd.merge(movies, ratings)
#print(ratings.head())

# we will construct user/movie rating matrix using pivot_table func on df
# NaN indicates missing data - movies that specific users didn't rate
movieRatings = ratings.pivot_table(index=['user_id'], columns=['title'], values='rating')
#print(movieRatings.head())
# we can use this df for user-based cf using rows and item-based cf using columns

# extract a Series of users who rated Star Wars
starWarsRatings = movieRatings['Star Wars (1977)']
#print(starWarsRatings.head())

# we use pandas' corrwith func to compute the pairwise correlation
# of Star War's vector of user rating with every other movie
similarMovies = movieRatings.corrwith(starWarsRatings)
# drop the results having no data
similarMovies = similarMovies.dropna()
# contruct a new df of movies and their correlation score to Star Wars
#df = pd.DataFrame(similarMovies)
#print(df.head(10))

# sort the results by similarity score
#print(similarMovies.sort_values(ascending=False))
# we need to get rid of movies that were only watched by a few people
# that are producing spurious results.

# we construct new df that counts up how many ratings exist for each movie and average rating
movieStats = ratings.groupby('title').agg({'rating':[np.size, np.mean]})
#print(movieStats.head())

# removing movies watched by fewer than 100 people
popularMovies = movieStats['rating']['size'] >=200
#print(movieStats[popularMovies].sort_values([('rating', 'mean')], ascending=False)[:15])

# joining this data to set of similar movies to Star Wars
df = movieStats[popularMovies].join(pd.DataFrame(similarMovies, columns=['similarity']))
#print(df.head())
# sort with similarity scores
print(df.sort_values(['similarity'], ascending=False)[:15])

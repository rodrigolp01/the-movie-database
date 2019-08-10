import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel

def convert_int(x):
    try:
        return int(x)
    except:
        return np.nan

movies_df = pd.read_csv('movies_metadata.csv')
links_small = pd.read_csv('links_small.csv')
links_small = links_small[links_small['tmdbId'].notnull()]['tmdbId'].astype('int')

movies_df['id'] = movies_df['id'].apply(convert_int)
movies_df = movies_df.drop([19730, 29503, 35587])
movies_df['id'] = movies_df['id'].astype('int')

moviesSubset = movies_df[movies_df['id'].isin(links_small)]

moviesSubset['tagline'] = moviesSubset['tagline'].fillna('')
moviesSubset['description'] = moviesSubset['overview'] + moviesSubset['tagline']
moviesSubset['description'] = moviesSubset['description'].fillna('')

tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
tfidf_matrix = tf.fit_transform(moviesSubset['description'])

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

moviesSubset = moviesSubset.reset_index()
titles = moviesSubset['title']
indices = pd.Series(moviesSubset.index, index=moviesSubset['title'])

def get_recommendations(title):
  idx = indices[title]
  if isinstance(idx, pd.Series):
  	idx = idx.iloc[0]
  sim_scores = list(enumerate(cosine_sim[idx]))
  sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
  sim_scores = sim_scores[1:31]
  movie_indices = [i[0] for i in sim_scores]
  return titles.iloc[movie_indices]

print('Recommendation by Similarity with: Titanic')
print(get_recommendations('Titanic').head(10))

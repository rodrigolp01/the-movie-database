import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from ast import literal_eval


movies_df = pd.read_csv('movies_metadata.csv')
movies_df['genres'] = movies_df['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])

movies_df['year'] = pd.to_datetime(movies_df['release_date'], errors='coerce').apply(lambda x: str(x).split('-')[0] if x != np.nan else np.nan)

s = movies_df.apply(lambda x: pd.Series(x['genres']),axis=1).stack().reset_index(level=1, drop=True)
s.name = 'genre'
gen_movies_df = movies_df.drop('genres', axis=1).join(s)

def recommender_by_genre(genre, perc=0.95):
  df = gen_movies_df[gen_movies_df['genre'] == genre]
  vote_counts = df[df['vote_count'].notnull()]['vote_count'].astype('int')
  vote_averages = df[df['vote_average'].notnull()]['vote_average'].astype('int')
  va_mean = vote_averages.mean()
  vc_quantile = vote_counts.quantile(perc)
    
  qualified = df[(df['vote_count'] >= vc_quantile) & (df['vote_count'].notnull()) & (df['vote_average'].notnull())][['title', 'year', 'vote_count', 'vote_average', 'popularity']]
  qualified['vote_count'] = qualified['vote_count'].astype('int')
  qualified['vote_average'] = qualified['vote_average'].astype('int')
  
  qualified = qualified.sort_values('vote_count', ascending=False).head(250)
    
  return qualified

print('recommendation for genre: Romance')
print(recommender_by_genre('Romance').head(15))

#Code used to forecasting movies revenue and classification movie success

import datetime
import ast
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.model_selection import train_test_split
from wordcloud import WordCloud, STOPWORDS

# Reading CSV's used for prediction and classification
movies_df = pd.read_csv('movies_metadata.csv')
credits_df = pd.read_csv('credits.csv')


#Creating methods and variables to help feature's preprocessing
month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
day_order = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

def get_month(x):
    try:
        return month_order[int(str(x).split('-')[1]) - 1]
    except:
        return np.nan

def get_day(x):
    try:
        year, month, day = (int(i) for i in x.split('-'))    
        answer = datetime.date(year, month, day).weekday()
        return day_order[answer]
    except:
        return np.nan

def clean_numeric(x):
    try:
        return float(x)
    except:
        return np.nan

def convert_int(x):
    try:
        return int(x)
    except:
        return np.nan


# Drop features that are not important at first!
movies_df = movies_df.drop(['imdb_id'], axis=1) 
movies_df = movies_df.drop('original_title', axis=1)

movies_df['adult'].value_counts() #The adult feature is not of much use because 99.98% of the movies are not adult.
movies_df = movies_df.drop('adult', axis=1)

movies_df['id'] = movies_df['id'].apply(convert_int) #To permit joins with other dataframes
movies_df = movies_df.drop([19730, 29503, 35587]) #NaN movie id's
movies_df['id'] = movies_df['id'].astype('int') #TypeCast to int to permit Joins

movies_df['revenue'] = movies_df['revenue'].replace(0, np.nan) #Convert 0 revenue to NaN for future removal
movies_df['vote_average'] = movies_df['vote_average'].replace(0, np.nan) #Convert 0 vote_average to NaN for future removal

movies_df['budget'] = pd.to_numeric(movies_df['budget'], errors='coerce') #Convert to numeric because of unclear values
movies_df['budget'] = movies_df['budget'].replace(0, np.nan) #Convert 0 budget to NaN for future removal

movies_df['return'] = movies_df['revenue'] / movies_df['budget'] #Create return feature for classifier (class1: return > 1 | class2: return <=1)

#Pre-process date feature format and nan values. Create new relevant features: year, day, month
movies_df['year'] = pd.to_datetime(movies_df['release_date'], errors='coerce').apply(lambda x: str(x).split('-')[0] if x != np.nan else np.nan)
movies_df['year'] = movies_df['year'].replace('NaT', np.nan)
movies_df['year'] = movies_df['year'].apply(clean_numeric)
movies_df['day'] = movies_df['release_date'].apply(get_day)
movies_df['month'] = movies_df['release_date'].apply(get_month)


#Type Cast to string for guarantee (use in wordcloud)
movies_df['title'] = movies_df['title'].astype('str')
movies_df['overview'] = movies_df['overview'].astype('str')

#WordCloud Exploration Analysis: 
title_corpus = ' '.join(movies_df['title'])
overview_corpus = ' '.join(movies_df['overview'])
title_wordcloud = WordCloud(stopwords=STOPWORDS, background_color='white', height=2000, width=4000).generate(title_corpus)
#print(title_wordcloud.words_)
plt.figure(figsize=(16,8))
plt.imshow(title_wordcloud)
plt.axis('off')
plt.savefig('titleWordCloud.png')

overview_wordcloud = WordCloud(stopwords=STOPWORDS, background_color='white', height=2000, width=4000).generate(overview_corpus)
#print(overview_wordcloud.words_)
plt.figure(figsize=(16,8))
plt.imshow(overview_wordcloud)
plt.axis('off')
plt.savefig('OverviewWordCloud.png')


#Convert to List object
movies_df['production_countries'] = movies_df['production_countries'].fillna('[]').apply(ast.literal_eval)
movies_df['production_countries'] = movies_df['production_countries'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])

movies_df['production_companies'] = movies_df['production_companies'].fillna('[]').apply(ast.literal_eval)
movies_df['production_companies'] = movies_df['production_companies'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])


#TypeCast to Float and check statistical values (deviation to mean value)
movies_df['popularity'] = movies_df['popularity'].apply(clean_numeric).astype('float')
movies_df['vote_count'] = movies_df['vote_count'].apply(clean_numeric).astype('float')
movies_df['vote_average'] = movies_df['vote_average'].apply(clean_numeric).astype('float')
movies_df['runtime'] = movies_df['runtime'].astype('float')
movies_df['runtime'] = movies_df['runtime'].fillna(movies_df['runtime'].mean())

movies_df['popularity'].describe()
movies_df['vote_count'].describe()
movies_df['vote_average'].describe()
movies_df['runtime'].describe()

#Dropping status feature because 99.2% of all types are released
movies_df['status'].value_counts()
movies_df = movies_df.drop('status', axis=1)


# Count the number of spoken languages per film
movies_df['spoken_languages'] = movies_df['spoken_languages'].fillna('[]').apply(ast.literal_eval).apply(lambda x: len(x) if isinstance(x, list) else np.nan)
#movies_df['spoken_languages'].value_counts()


#Check Budget and Revenue Correlation
movies_df['budget'].describe()
movies_df['revenue'].describe()
movies_df[['budget', 'revenue']].corr() #correlation with revenue: ~0.73
movies_df.corr()['budget'][:] 
plt.figure(figsize=(16,8))
sns.jointplot(x='budget',y='revenue',data=movies_df[movies_df['return'].notnull()])
plt.savefig('budgetAndRevenueCorr.png')


#Pre-process genre feature to get list of genre names for each movie
movies_df['genres'] = movies_df['genres'].fillna('[]').apply(ast.literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])


#Merging Movie and credits CSV
movies_df = movies_df.merge(credits_df, on='id')


#Preprocess credits features for regression and classify
movies_df['cast'] = movies_df['cast'].apply(ast.literal_eval) # convert string list to list
movies_df['crew'] = movies_df['crew'].apply(ast.literal_eval)

movies_df['cast_size'] = movies_df['cast'].apply(lambda x: len(x)) # New features: count cast and crew
movies_df['crew_size'] = movies_df['crew'].apply(lambda x: len(x))


#Removing remaing unused features for better model trainning (curse of dimensionality)
movies_df = movies_df.drop(['id', 'overview', 'poster_path', 'release_date', 'tagline', 'video'], axis=1)

#Regression: Predicting Movie Revenues
rgf = movies_df[movies_df['return'].notnull()] #Filter not Null revenue/budget to predict

rgf = rgf.drop(['return', 'crew'], axis=1)

rgf['belongs_to_collection'] = rgf['belongs_to_collection'].apply(lambda x: 0 if x == np.nan else 1)

#Get list of avaliable genres
s = rgf.apply(lambda x: pd.Series(x['genres']),axis=1).stack().reset_index(level=1, drop=True)
s.name = 'genre'
gen_rgf = rgf.drop('genres', axis=1).join(s)
genres_train = gen_rgf['genre'].drop_duplicates()

#Method for regression features pre-processing. Create boolean feature column for each genre!
def regression_engineering(df):
  for genre in genres_train:
    df['is_' + str(genre)] = df['genres'].apply(lambda x: 1 if genre in x else 0)
  df['genres'] = df['genres'].apply(lambda x: len(x))
  df['homepage'] = df['homepage'].apply(lambda x: 0 if x == np.nan else 1)
  df['is_english'] = df['original_language'].apply(lambda x: 1 if x=='en' else 0)
  df = df.drop('original_language', axis=1)
  df['production_companies'] = df['production_companies'].apply(lambda x: len(x))
  df['production_countries'] = df['production_countries'].apply(lambda x: len(x))
  df['is_Friday'] = df['day'].apply(lambda x: 1 if x=='Fri' else 0)
  df = df.drop('day', axis=1)
  df['is_Holiday'] = df['month'].apply(lambda x: 1 if x in ['Apr', 'May', 'Jun', 'Nov'] else 0)
  df = df.drop('month', axis=1)
  df = df.drop(['title', 'cast'], axis=1)
  df = pd.get_dummies(df, prefix='is') #Quantify all is_ columns!!
  df['vote_average'] = df['vote_average'].fillna(df['vote_average'].mean())
  return df

X, Y = rgf.drop('revenue', axis=1), rgf['revenue']
X = regression_engineering(X)
train_X, test_X, train_Y, test_Y = train_test_split(X, Y, train_size=0.75, test_size=0.25) #randomly separating training and test set
reg = GradientBoostingRegressor() 
reg.fit(train_X, train_Y) #Train regressor model
print('Regressor Score: ', reg.score(test_X, test_Y))

#Compare with dummy regressor!!
dummy = DummyRegressor()
dummy.fit(train_X, train_Y)
print('Dummy Regressor Score: ', dummy.score(test_X, test_Y))

sns.set_style('whitegrid')
plt.figure(figsize=(12,14))
sns.barplot(x=reg.feature_importances_, y=X.columns)
plt.savefig('regressor.png')



#Classification: Predicting Movie Sucess
cls = movies_df[movies_df['return'].notnull()]

cls = cls.drop(['revenue'], axis=1)
cls['return'] = cls['return'].apply(lambda x: 1 if x >=1 else 0) #create binary output for classification
cls['return'].value_counts() #balanced classes

cls['belongs_to_collection'] = cls['belongs_to_collection'].fillna('').apply(lambda x: 0 if x == '' else 1)
cls['homepage'] = cls['homepage'].fillna('').apply(lambda x: 0 if x == '' else 1)

def classification_engineering(df):
  for genre in genres_train:
    df['is_' + str(genre)] = df['genres'].apply(lambda x: 1 if genre in x else 0)
  df['genres'] = df['genres'].apply(lambda x: len(x))
  df = df.drop('homepage', axis=1)
  df['is_english'] = df['original_language'].apply(lambda x: 1 if x=='en' else 0)
  df = df.drop('original_language', axis=1)
  df['production_companies'] = df['production_companies'].apply(lambda x: len(x))
  df['production_countries'] = df['production_countries'].apply(lambda x: len(x))
  df['is_Friday'] = df['day'].apply(lambda x: 1 if x=='Fri' else 0)
  df = df.drop('day', axis=1)
  df['is_Holiday'] = df['month'].apply(lambda x: 1 if x in ['Apr', 'May', 'Jun', 'Nov'] else 0)
  df = df.drop('month', axis=1)
  df = df.drop(['title', 'cast'], axis=1)
  #df = pd.get_dummies(df, prefix='is')
  df['vote_average'] = df['vote_average'].fillna(df['vote_average'].mean())
  df = df.drop('crew', axis=1)
  return df

cls = classification_engineering(cls)
X, Y = cls.drop('return', axis=1), cls['return']
train_X, test_X, train_Y, test_Y = train_test_split(X, Y, train_size=0.75, test_size=0.25, stratify=Y)
clf = GradientBoostingClassifier() #Gradient Tree Boosting or Gradient Boosted Regression Trees (GBRT)
clf.fit(train_X, train_Y)
print('Classification Score: ', clf.score(test_X, test_Y))
dummy = DummyClassifier(strategy='most_frequent')
dummy.fit(train_X, train_Y)
dummy.score(test_X, test_Y)
plt.figure(figsize=(12,14))
sns.barplot(x=clf.feature_importances_, y=X.columns)
plt.savefig('classification.png')

#Most relevant Features for Revenue prediction: vote_count and Budget
#Most relevant Features for success binary classification: vote_count, Budget, year and belongs_to_collection
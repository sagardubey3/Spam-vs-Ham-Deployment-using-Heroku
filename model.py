# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 23:12:25 2019

@author: sagar
"""

import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import nltk
import re

df=pd.read_csv('spam.csv',encoding="latin-1")
df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
df.head()
from nltk.tokenize.toktok import ToktokTokenizer
tokenizer=ToktokTokenizer()
stopwords_list=set(nltk.corpus.stopwords.words("english"))
def remove_stopwords(text):
  tokens=tokenizer.tokenize(text)
  tokens=[token.strip() for token in tokens]
  filtered_tokens=[token for token in tokens if token.lower() not in stopwords_list]
  filtered_tokens=" ".join(filtered_tokens)
  return filtered_tokens
def remove_special_char(text):
  pattern=r'[^a-zA-Z\s]'
  text=re.sub(pattern,'',text)
  return text
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
def stemmer(text):
        text=' '.join([ps.stem(word) for word in text.split()])
        return text
def normalize_corpus(corpus):
    
    normalized_corpus = []
    # normalize each document in the corpus
    for doc in corpus:
        # lowercase the text    
        doc = doc.lower()
        # remove extra newlines
        doc = re.sub(r'[\r|\n|\r\n]+', ' ',doc)
        # lemmatize text
        doc =stemmer(doc)
        # remove special characters and\or digits    
        special_char_pattern = re.compile(r'([{.(-)!}])')
        doc = special_char_pattern.sub(" \\1 ", doc)
        doc = remove_special_char(doc)  
        # remove extra whitespace
        doc = re.sub(' +', ' ', doc)
        # remove stopwords
        doc = remove_stopwords(doc)
        normalized_corpus.append(doc)
        
    return normalized_corpus
df["Processed"]=normalize_corpus(df["v2"])
norm_corpus=list(df["Processed"])
#showing sample
df.iloc[1][["Processed","v2"]].to_dict()
df['v1'] = df['v1'].map({'ham': 0, 'spam': 1})
y = df['v1']

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['Processed'])
model=MultinomialNB()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=2019)
model = MultinomialNB()
model.fit(X_train,y_train)
model.score(X_test,y_test)

 
from sklearn.externals import joblib
filename = 'finalized_model.sav'
joblib.dump(model, filename)
with open('vectorizer.pickle', 'wb') as handle:
    pickle.dump(vectorizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

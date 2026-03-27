import pandas as pd
import numpy as np

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

import pickle

nltk.download('stopwords')

data = pd.read_csv("spam.csv", encoding="latin-1")
data = data[['v1', 'v2']]
data.columns = ['label', 'message']
print("Data head:")
print(data.head())

ps = PorterStemmer()
corpus = []

for msg in data['message']:
    msg = re.sub('[^a-zA-Z]', ' ', msg)
    msg = msg.lower()
    msg = msg.split()
    msg = [ps.stem(word) for word in msg if word not in stopwords.words('english')]
    msg = ' '.join(msg)
    corpus.append(msg)

tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(corpus).toarray()

y = data['label'].map({'ham': 0, 'spam': 1})

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

pickle.dump(model, open("spam_model.pkl", "wb"))
pickle.dump(tfidf, open("tfidf.pkl", "wb"))
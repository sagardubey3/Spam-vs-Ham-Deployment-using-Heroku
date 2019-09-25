import pandas as pd
from sklearn.externals import joblib
from flask import Flask, request,render_template
from flask_cors import CORS
import flask
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import os
from sklearn.model_selection import train_test_split

app = Flask(__name__)
CORS(app)

app=flask.Flask(__name__,template_folder='templates')
clf=MultinomialNB()
cv = CountVectorizer()
global tokenizer
global model
global labels

df = pd.read_csv('spam.csv', encoding="latin-1")
df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
df['v1'] = df['v1'].map({'ham': 0, 'spam': 1})
X = df['v2']
y = df['v1']
cv = CountVectorizer()
X = cv.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
clf = MultinomialNB()
clf.fit(X_train,y_train)


@app.route('/')
def main():
    return render_template('main.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
    temp=request.get_data(as_text=True)
    new=[temp]
    vect = cv.transform(new).toarray()
    pred = clf.predict(vect)
    if pred==1:
        pred="SPAM"
    else:
        pred="HAM"
    return str(pred)
    
if __name__=="__main__":
    port=int(os.environ.get('PORT',5000))
    app.run(port=port,debug=True,use_reloader=False)
    
    
import numpy as np
from flask import Flask, request,render_template
from flask_cors import CORS
import os
from sklearn.externals import joblib
import pickle
import flask
import os
app = Flask(__name__)
CORS(app)

app=flask.Flask(__name__,template_folder='templates')

filename = 'finalized_model.sav'
model = joblib.load(filename)
with open('vectorizer.pickle', 'rb') as handle:
	vectorizer = pickle.load(handle)

@app.route('/')
def main():
    return render_template('main.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
    temp=request.get_data(as_text=True)
    new=[temp]
    message=vectorizer.transform(new)
    pred = model.predict(message)
    if pred == 1:
        return "SPAM"
    else:
        return "HAM"
    return str(pred)
    
if __name__=="__main__":
    port=int(os.environ.get('PORT',5000))
    app.run(port=port,debug=True,use_reloader=False)
    
    
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 13:22:13 2023

@author: AbdullahNaveed
"""

from flask import Flask, request
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)

filename = "saved_models/my_model.pickle"
classifier = pickle.load(open(filename, "rb"))



@app.route('/')
def welcome():
    return "Welcome!"



@app.route('/predict')
def predict_note_authentication():
    
    variance = request.args.get('variance')
    skewness = request.args.get('skewness')
    curtosis = request.args.get('curtosis')
    entropy = request.args.get('entropy')

    prediction = classifier.predict([[variance,skewness,curtosis,entropy]])
    
    if(prediction >= 0.5):
        return "The Bank Note is Authentic!"
    else:
        return "The Bank note is Fake!"
    
    
    
    
@app.route('/predict_file', methods = ['GET', 'POST' ])
def predict_note_authentication_by_file():
    
    df_test = pd.read_csv(request.files.get("file"))
    
    predictions = classifier.predict(df_test)
     
    return "The predicted values are " + str(list(predictions))




if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
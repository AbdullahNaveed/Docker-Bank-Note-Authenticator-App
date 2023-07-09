# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 16:10:34 2023

@author: AbdullahNaveed
"""

import numpy as np
import pickle
import pandas as pd
import streamlit as st 

from PIL import Image


filename = "saved_models/my_model.pickle"
classifier = pickle.load(open(filename, "rb"))

def welcome():
    return "Welcome"

def predict_note_authentication(variance,skewness,curtosis,entropy):
   
    prediction=classifier.predict([[variance,skewness,curtosis,entropy]])
    
    if(prediction >= 0.5):
        return "The Bank Note is Authentic!"
    else:
        return "The Bank note is Fake!"



def main():
    
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Bank Note Authenticator</h2>
    </div>
    """
    
    st.markdown(html_temp,unsafe_allow_html=True)
    
    variance = st.text_input("Variance","")
    skewness = st.text_input("Skewness","")
    curtosis = st.text_input("Curtosis","")
    entropy = st.text_input("Entropy","")
    
    result=""
    if st.button("Predict"):
        result=predict_note_authentication(variance,skewness,curtosis,entropy)
        
    st.success('{}'.format(result))
    
    if st.button("About Developer"):
        st.text("Made by Abdullah Naveed")
        st.write("Github: [Link](https://github.com/AbdullahNaveed)")
        
        
if __name__=='__main__':
    main()
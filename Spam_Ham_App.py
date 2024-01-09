import numpy as np
import pandas as pd
import pickle
import streamlit as st

def spam_ham_prediction(input_message):
    
    with open('Spam_Ham_Model.pkl', 'rb') as model_file:
        lr_model = pickle.load(model_file)
        tfidf = pickle.load(model_file)

    # Transform the input using the same vectorizer used during training
    input_message_vectorized = tfidf.transform([input_message])

    lr_prediction = lr_model.predict(input_message_vectorized.reshape(1,-1))

    return lr_prediction

def main():
    # giving a title
    st.title('Spam or Ham Prediction App')
    
    # Getting the input data from the user 
    Message = st.text_input('Enter the Message here...')

    # creating a button for Prediction
    if st.button('Predict'):
        lr_prediction = spam_ham_prediction(Message)

        st.subheader('Prediction:')

        if lr_prediction[0] == 0:
            st.success("The Message is Spam!")
        else:
            st.success("The Message is Ham.")

if __name__ == '__main__':
    main()

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

word_index = imdb.get_word_index()
reversed_word_index = {value:key for key,value in word_index.items()}

model = load_model('simple_rnn_imdb1.h5')

def decoded_review(review):
    return ' '.join([reversed_word_index.get(i-3, '?') for i in review])

def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word , 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

import streamlit as st

st.title('IMDB Movie Review Sentiment Analysis')
st.write('Please give movie review so it can classify it as positive or negative sentiment')

user_input = st.text_area('Movie Review')

if st.button('Classify'):
    preprocess_input = preprocess_text(user_input)

    prediction = model.predict(preprocess_input)
    sentiment = 'positive' if prediction[0][0] > 0.5 else 'Negative'

    st.write(f'Sentiment: {sentiment}')
    st.write(f'Score: {prediction[0][0]}')

else:
    st.write('Please give movie review so it can classify it as positive or negative sentiment')

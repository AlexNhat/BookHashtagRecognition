import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import pickle
import nltk
from nltk.corpus import stopwords
from flask import Flask, request, jsonify

stop_words = set(stopwords.words('english'))

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = text.lower().split()  # Chuyển tất cả các từ thành chữ thường
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)

import emoji

def remove_emoji(text):
    return emoji.demojize(text)

import re

def remove_dates(text):
    return re.sub(r'\b(?:\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}|\d{1,2}[/\-]\d{1,2})\b', '', text)

def remove_numbers_and_special_characters(text):
    return re.sub(r'[^a-zA-Z\s]', '', text)

def preprocess_data_real_time(text):
    text = remove_stopwords(text)
    text = remove_emoji(text)
    text = remove_dates(text)
    text = remove_numbers_and_special_characters(text)
    text = remove_stopwords(text)
    tokenize = load_tokenize_data()
    sequences = tokenize.texts_to_sequences([text])
    padded = pad_sequences(sequences, maxlen=100, padding='post')
    return padded
# Create Flask app
app = Flask(__name__)

# Load TensorFlow model
try:
    model = load_model('mlp_model_with_custom_loss1.h5')
except OSError:
    with open('mlp_model.pkl', 'rb') as file:
        model = pickle.load(file)

# Load second model using pickle
with open('mlp_model_hashtag2.pkl', 'rb') as file:
    model2 = pickle.load(file)

def load_tokenize_data():
  with open('tokenizer.pkl', 'rb') as file:
    loaded_tokenizer = pickle.load(file)
    return loaded_tokenizer

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        # Receive input data
        data = request.json
        book_name = data.get('Title')
        book_description = data.get('Description')

        # Preprocess data and make predictions
        padd = preprocess_data_real_time(book_description)
        pred_1 = model.predict(padd)
        pred_1 = np.round(pred_1)
        value = pred_1[0, 0]
        pred_2 = model2.predict(padd)

        # Get top 3 predictions
        top_indices = np.argsort(pred_2[0])[::-1][:3]
        top_values = pred_2[0][top_indices]

        result = {
            'pred_1': float(value),
            'top_indices': top_indices.tolist(),
            'top_values': top_values.tolist()
        }

        return jsonify(result)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5501, debug=True)
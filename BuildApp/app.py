from flask import Flask, request, render_template
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the trained model
with open('song_popularity_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Dummy columns based on the features used in the model
# Replace with actual columns used during training
expected_columns = ['artist_name_length', 'genre_count', 'dummy_feature1', 'dummy_feature2']

def preprocess_input(artist_name, genres):
    # Example preprocessing, adjust as necessary
    features = {}
    features['artist_name_length'] = len(artist_name)
    features['genre_count'] = len(genres.split(','))
    
    # Add other necessary preprocessing steps here
    # For example, handle genres and other features used during training
    
    # Convert the features into a DataFrame with the same columns as during training
    features_df = pd.DataFrame([features], columns=expected_columns).fillna(0)
    return features_df

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        artist_name = request.form['artist_name']
        genres = request.form['genres']
        
        # Preprocess the input and make prediction
        features = preprocess_input(artist_name, genres)
        prediction = model.predict(features)
        
        return render_template('index.html', prediction=prediction[0])

    return render_template('index.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True)

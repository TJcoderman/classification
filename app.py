from flask import Flask, render_template, url_for
import pandas as pd
import json
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

app = Flask(__name__, static_folder='static')

@app.route('/')
def index():
    # Read the dataset
    dataset = pd.read_csv('C:/Users/Tejus Kapoor/Desktop/classification/templates/Social_Network_Ads.csv')
    
    # Prepare the data
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    
    # Scale the features
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    # Create and train the KNN classifier
    classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
    classifier.fit(X_train, y_train)
    
    # Make predictions and calculate accuracy
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Convert dataset to JSON for easy rendering in the template
    data_json = dataset.to_json(orient='records')
    
    # Get list of algorithm files
    algorithm_files = [f for f in os.listdir('codes') if f.endswith('.ipynb') or f.endswith('.py')]
    
    # Final algorithm information
    final_algorithm = "K-Nearest Neighbors (KNN)"
    
    return render_template('index.html', data=data_json, algorithm_files=algorithm_files, 
                           final_algorithm=final_algorithm, accuracy=accuracy)

if __name__ == '__main__':
    app.run(debug=True)
from flask import Flask, render_template, request, send_from_directory
import random, os
from werkzeug.utils import secure_filename
import os
import joblib
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.sentiment import SentimentIntensityAnalyzer


app = Flask(__name__)  
random.seed(0)
app.config['SECRET_KEY'] = os.urandom(24)

UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'

dir_path = os.path.dirname(os.path.realpath(__file__))

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/result', methods=['GET', 'POST'])
def result():
    return render_template('result.html')

@app.route('/algo', methods=['GET', 'POST'])
def algo():
    return render_template('algo.html')

import pickle
import numpy as np
# Replace 'model.pkl' with the actual filename of your model
model = pickle.load(open('models/RF_model.pkl', 'rb'))

@app.route("/predict", methods=["GET", "POST"])
def pred():
    if request.method == "POST":
        try:
            # Extracting form data
            age = int(request.form['age'])
            gender = request.form['gender']
            stream = int(request.form['stream'])
            internships = int(request.form['internships'])
            cgpa = float(request.form['cgpa'])
            certification = request.form['certification']
            backlogs = request.form['backlogs']
            
            # Encoding categorical data as required
            gender_encoded = 1 if gender == 'male' else 0
            certification_encoded = 1 if certification == 'yes' else 0
            backlogs_encoded = 1 if backlogs == 'yes' else 0

            # Prepare input array for prediction
            input_data = np.array([[age, gender_encoded, stream, internships, cgpa, certification_encoded, backlogs_encoded]])
            
            # Prediction
            prediction = model.predict(input_data)
            prediction_proba = model.predict_proba(input_data)  # If you want confidence score

            # Map prediction to a label
            if prediction[0] == 1:
                result = "You will be Placed"
            else:
                suggestions = []
                if cgpa < 7:  # Assuming 7 as a threshold
                    suggestions.append("Try to increase CGPA")
                if internships == 0:
                    suggestions.append("Try to do some internships")
                if certification == 0:
                    suggestions.append("Do some global certifications")
                if backlogs=='yes':
                    suggestions.append("Try to Clear Backlogs")
                
                suggestion_text = " | ".join(suggestions) if suggestions else "Work on improving your profile"
                result = "You may not be Placed! So,  "
                result=result+suggestion_text

            # Pass the result to the results page
            return render_template('r.html', result=result, confidence=prediction_proba[0][prediction[0]])

        except Exception as e:
            return render_template('r.html', error=f"An error occurred: {e}")
    
    # Render the input form for GET request
    return render_template('reviewpred.html')

if __name__ == '__main__':
    app.run(debug=True)
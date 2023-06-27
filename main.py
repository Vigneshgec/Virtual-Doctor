import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from flask import Flask, render_template, request, jsonify, redirect, url_for
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import cv2
import pytesseract
import spacy

app = Flask(__name__)
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

# Step 1: Data Collection and Preprocessing
data = pd.read_csv("C:\\Users\\vigne\\OneDrive\\Desktop\\vignesh\\vd\\data1.csv")
symptoms = data['Symptoms']
diagnoses = data['Diagnoses']
treatment_recommendations = data['Recommendation']
severity_scores = data['severity']  # Added column for severity scores
nlp = spacy.load('en_core_web_md')
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(symptoms)
y = np.array(diagnoses)

# Step 2: Model Training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 3: Treatment Recommendations
diagnosis_to_recommendation = dict(zip(diagnoses, treatment_recommendations))

# Define severity scores for each diagnosis. The scores are on a scale of 1 to 10, with 10 being the most severe.
diagnosis_to_severity = dict(zip(diagnoses, severity_scores))

# Assign a health score to each symptom based on the severity of its predicted diagnosis
def compute_health_percentage(severity):
    health_percentage = 100 - severity
    return max(0, health_percentage)

# Perform text preprocessing on user symptoms
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    # Tokenize the text
    tokens = word_tokenize(text)

    # Remove stopwords and punctuation, and lemmatize the tokens
    preprocessed_tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens if token.lower() not in stop_words and token.isalpha()]

    # Join the preprocessed tokens back into a single string
    preprocessed_text = ' '.join(preprocessed_tokens)

    return preprocessed_text

# Extract text from an image using OCR (Optical Character Recognition)
def extract_text_from_image(image_path):
    try:
        # Read the image using OpenCV
        image = cv2.imread(image_path)

        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply image preprocessing (if required) using OpenCV (e.g., thresholding, noise removal)

        # Perform text extraction using PyTesseract
        extracted_text = pytesseract.image_to_string(gray, lang='eng')

        return extracted_text.strip()
    except Exception as e:
        print("Error during text extraction:", str(e))
        return None

# Analyze the extracted text and identify relevant symptoms
def analyze_extracted_text(text):
    # Perform NLP processing on the extracted text
    preprocessed_text = preprocess_text(text)

    matched_symptom = None
    max_match_count = 0

    # Match the preprocessed text with symptoms in the database
    for symptom in symptoms:
        preprocessed_symptom = preprocess_text(symptom)
        match_count = sum(1 for word in preprocessed_symptom.split() if word in preprocessed_text)

        if match_count > max_match_count:
            max_match_count = match_count
            matched_symptom = symptom

    return matched_symptom

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/diagnose', methods=['POST'])
def diagnose():
    user_symptoms = request.form['symptoms']
    user_symptoms = [s.strip() for s in user_symptoms.split(',') if s.strip()]
    user_file = request.files.get('file')

    results = []
    total_severity = 0
    symptom_count = 0

    if user_symptoms:
        for symptom in user_symptoms:
            preprocessed_symptom = preprocess_text(symptom)
            symptom_vector = vectorizer.transform([preprocessed_symptom])
            if symptom_vector.sum() == 0:
                results.append({
                    'symptom': symptom,
                    'diagnosis': "Unknown symptom",
                    'recommendation': "Consult a medical professional",
                })
                total_severity += 10
                symptom_count += 1
            else:
                prediction = model.predict(symptom_vector)
                treatment_recommendation = diagnosis_to_recommendation.get(prediction[0], "No treatment recommendation found")
                severity = diagnosis_to_severity.get(prediction[0], 1)
                total_severity += severity
                symptom_count += 1
                results.append({
                    'symptom': symptom,
                    'diagnosis': prediction[0],
                    'recommendation': treatment_recommendation,
                })

    extracted_data = ""
    if user_file:
        # Save the uploaded image temporarily
        image_path = "C:\\Users\\vigne\\OneDrive\\Desktop\\vignesh\\vd\\templates\\IMG1.png"
        user_file.save(image_path)

        # Extract text from the uploaded image
        extracted_text = extract_text_from_image(image_path)

        if extracted_text:
            # Analyze the extracted text to identify relevant symptoms
            matched_symptom = analyze_extracted_text(extracted_text)

            if matched_symptom is not None:
                preprocessed_symptom = preprocess_text(matched_symptom)
                symptom_vector = vectorizer.transform([preprocessed_symptom])
                if symptom_vector.sum() == 0:
                    results.append({
                        'symptom': matched_symptom,
                        'diagnosis': "Unknown symptom",
                        'recommendation': "Consult a medical professional",
                    })
                    total_severity += 10
                    symptom_count += 1
                else:
                    prediction = model.predict(symptom_vector)
                    treatment_recommendation = diagnosis_to_recommendation.get(prediction[0], "No treatment recommendation found")
                    severity = diagnosis_to_severity.get(prediction[0], 1)
                    total_severity += severity
                    symptom_count += 1
                    results.append({
                        'symptom': matched_symptom,
                        'diagnosis': prediction[0],
                        'recommendation': treatment_recommendation,
                    })
            extracted_data = extracted_text

    if symptom_count == 0:
        response = {
            'diagnosis': 'Not Found',
            'recommendation': 'No diagnosis or recommendation found for the entered symptoms.',
            'health_percentage': 100,
            'extracted_data': extracted_data
        }
    else:
        average_severity = total_severity / symptom_count
        health_percentage = compute_health_percentage(average_severity)

        if health_percentage < 20:
            appointment_link = url_for('make_appointment')
            response = {
                'results': results,
                'health_percentage': health_percentage,
                'extracted_data': extracted_data,
                'appointment_link': appointment_link
            }
        else:
            response = {
                'results': results,
                'health_percentage': health_percentage,
                'extracted_data': extracted_data
            }
    return jsonify(response)

@app.route('/make_appointment')
def make_appointment():
    # Render the appointment page here
    return render_template('diagnosis.html')

if __name__ == '__main__':
    app.run(debug=True)

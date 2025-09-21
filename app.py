from flask import Flask, request, jsonify, render_template
import joblib
import logging
import os

# إعداد Flask
app = Flask(__name__)

# إعداد logging
logging.basicConfig(filename='app.log', level=logging.INFO, 
                    format='%(asctime)s %(levelname)s %(message)s')

# تحميل الموديل والـ vectorizer
try:
    lr_model = joblib.load("models/logistic_final.joblib")
    vectorizer = joblib.load("models/vectorizer.joblib")
    logging.info("Model and vectorizer loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model/vectorizer: {e}")
    lr_model, vectorizer = None, None

# صفحة البداية
@app.route('/')
def home():
    return render_template('index.html')

# RESTful endpoint للتنبؤ
@app.route('/predict', methods=['POST'])
def predict():
    if lr_model is None or vectorizer is None:
        return jsonify({'error': 'Model or vectorizer not loaded'}), 500

    try:
        data = request.json
        if 'features' not in data or len(data['features']) == 0:
            return jsonify({'error': 'Missing comment in request'}), 400

        comment = data['features'][0]  # نص التعليق
        features_array = vectorizer.transform([comment])  # تحويل النص لأرقام

        prediction = lr_model.predict(features_array)

        sentiment = "Positive" if prediction[0] == 1 else "Negative"

        return jsonify({'prediction': sentiment})

    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 400

# تشغيل التطبيق
if __name__ == '__main__':
    app.run(debug=True)

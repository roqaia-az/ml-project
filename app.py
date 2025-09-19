from flask import Flask, request, jsonify, render_template
import pickle
import os

app = Flask(__name__)

# --------- التحميل المؤقت للموديل ----------
# حاليًا القيمة ثابتة، لاحقًا استبدلي ملفات الموديل هنا
MODEL_PATH = "model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"

model = None
vectorizer = None

if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
    model = pickle.load(open(MODEL_PATH, "rb"))
    vectorizer = pickle.load(open(VECTORIZER_PATH, "rb"))
    print("✅ Model and vectorizer loaded successfully.")
else:
    print("⚠️ Model or vectorizer not found. Using dummy prediction.")

# --------- الصفحة الرئيسية ----------
@app.route("/")
def home():
    return render_template("index.html")

# --------- API Prediction ----------
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    comment = data.get("comment", "").strip()

    if not comment:
        return jsonify({"prediction": "⚠️ Please enter a comment"})

    if model and vectorizer:
        # prediction حقيقي
        X = vectorizer.transform([comment])
        prediction = model.predict(X)[0]
    else:
        # prediction مؤقت
        prediction = "Positive"

    return jsonify({"prediction": prediction})

# --------- تشغيل السيرفر ----------
if __name__ == "__main__":
    app.run(debug=True)

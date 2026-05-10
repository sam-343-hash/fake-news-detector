"""
Inference module for fake news detection.
"""
import joblib

class FakeNewsDetector:
    def __init__(self, model_path='models/logistic_regression.pkl', vectorizer_path='models/vectorizer.pkl'):
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)

    def predict(self, text):
        vec = self.vectorizer.transform([text])
        label = self.model.predict(vec)[0]
        proba = self.model.predict_proba(vec)[0]
        return {
            'label': 'REAL' if label == 0 else 'FAKE',
            'confidence': round(float(max(proba)), 4),
            'real_probability': round(float(proba[0]), 4),
            'fake_probability': round(float(proba[1]), 4),
        }

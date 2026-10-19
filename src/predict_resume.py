
from pathlib import Path
import joblib

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIR / "models" / "resume_classifier.pkl"

# Load the trained model
model = joblib.load(MODEL_PATH)

# Sample resume text for testing
resume_text = """
Python developer with experience in Python, SQL, Pandas,
NumPy, machine learning, data analysis, and data visualization.
Built projects using scikit-learn and Flask.
"""

# Predict the resume category
predicted_category = model.predict([resume_text])[0]

print("\n===== SMART HIRE PREDICTION =====")
print("Predicted job category:", predicted_category)

# Show the top 3 category probabilities
if hasattr(model, "predict_proba"):
    probabilities = model.predict_proba([resume_text])[0]
    categories = model.classes_

    ranked = sorted(
        zip(categories, probabilities),
        key=lambda item: item[1],
        reverse=True
    )

    print("\nTop 3 predicted categories:")
    for category, probability in ranked[:3]:
        print(f"{category}: {probability:.2%}")

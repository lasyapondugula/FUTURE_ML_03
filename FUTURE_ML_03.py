# RESUME UPLOAD & JOB ROLE PREDICTION SYSTEM

import pandas as pd
import numpy as np
import re
from PyPDF2 import PdfReader

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

# 1. Load Training Dataset
df = pd.read_csv("candidate_job_role_dataset.csv")  # Your dataset file

# Combine features
df["combined_text"] = (
    df["skills"] + " " +
    df["qualification"] + " " +
    df["experience_level"]
)

# 2. Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["combined_text"])

# Encode target
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df["job_role"])

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

print("Model trained successfully!\n")

# 3. FUNCTION TO EXTRACT TEXT FROM PDF
def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# 4. PREDICT JOB ROLE FROM RESUME FILE
def predict_from_resume(file_path):
    
    # Check file type
    if file_path.endswith(".pdf"):
        resume_text = extract_text_from_pdf(file_path)
    elif file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as file:
            resume_text = file.read()
    else:
        print("Unsupported file format! Use PDF or TXT.")
        return
    
    # Clean text (basic cleaning)
    resume_text = resume_text.lower()
    resume_text = re.sub(r'[^a-zA-Z ]', ' ', resume_text)
    
    # Convert to vector
    vector = vectorizer.transform([resume_text])
    
    # Predict
    prediction = model.predict(vector)
    job_role = label_encoder.inverse_transform(prediction)
    
    print("Predicted Job Role:", job_role[0])

# 5. UPLOAD RESUME HERE

# Example:
# predict_from_resume("sample_resume.pdf")

file_path = input("Enter resume file path (PDF or TXT): ")
predict_from_resume(file_path)
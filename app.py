import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------
# Load Dataset & Train Model
# -------------------------
data = pd.read_csv("diabetes.csv")

X = data[['Glucose', 'BloodPressure', 'BMI', 'Age']]
y = data['Outcome']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestClassifier(random_state=42)
model.fit(X_scaled, y)

# -------------------------
# Medicine Recommendation
# -------------------------
medicine_recommendation = {
    "Diabetes": [
        "Metformin",
        "Insulin (if prescribed by doctor)",
        "Low sugar diet",
        "Regular exercise"
    ],
    "No Diabetes": [
        "Maintain healthy diet",
        "Regular physical activity",
        "Routine health checkups"
    ]
}

# -------------------------
# Content-Based Recommendation (TF-IDF)
# -------------------------
health_content = pd.DataFrame({
    "content_id": [1, 2, 3, 4, 5],
    "text": [
        "Diabetes insulin treatment and sugar control",
        "Low sugar diet and regular exercise for diabetes",
        "Blood pressure and heart health management",
        "Healthy lifestyle and balanced diet",
        "Regular physical activity and weight management"
    ]
})

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(health_content["text"])
cosine_sim = cosine_similarity(tfidf_matrix)

def recommend_content(content_id, top_n=3):
    idx = health_content.index[health_content["content_id"] == content_id][0]
    similarity_scores = list(enumerate(cosine_sim[idx]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    similarity_scores = similarity_scores[1:top_n+1]
    return [health_content.iloc[i[0]]["text"] for i in similarity_scores]

# -------------------------
# STREAMLIT UI
# -------------------------
st.title("🩺 Personalized Healthcare Recommendation System")

st.sidebar.header("Enter Patient Details")

glucose = st.sidebar.number_input("Glucose Level", 0, 300, 150)
bp = st.sidebar.number_input("Blood Pressure", 0, 200, 85)
bmi = st.sidebar.number_input("BMI", 0.0, 60.0, 32.0)
age = st.sidebar.number_input("Age", 1, 100, 45)

if st.sidebar.button("Get Recommendation"):
    sample = pd.DataFrame(
        [[glucose, bp, bmi, age]],
        columns=['Glucose', 'BloodPressure', 'BMI', 'Age']
    )

    sample_scaled = scaler.transform(sample)
    prediction = model.predict(sample_scaled)

    disease = "Diabetes" if prediction[0] == 1 else "No Diabetes"

    st.subheader("🧾 Prediction Result")
    st.write("**Predicted Disease:**", disease)

    st.subheader("💊 Recommended Medicines / Advice")
    for item in medicine_recommendation[disease]:
        st.write("-", item)

    st.subheader("📚 Recommended Health Tips")
    for rec in recommend_content(1):
        st.write("-", rec)

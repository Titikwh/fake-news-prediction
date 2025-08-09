import streamlit as st
import joblib

# Judul aplikasi
st.title("📰 Fake News Prediction App")
st.write("Masukkan teks berita untuk memprediksi apakah berita tersebut **FAKE** atau **REAL**.")

# Load model & vectorizer
@st.cache_resource
def load_model():
    model = joblib.load("PA_model.pkl")
    vectorizer = joblib.load("file_vectorizer_baru.pkl")
    return model, vectorizer

model, vectorizer = load_model()

# Input teks dari user
input_text = st.text_area("Masukkan teks berita di sini", height=200)

if st.button("Prediksi"):
    if input_text.strip() == "":
        st.warning("Silakan masukkan teks berita terlebih dahulu.")
    else:
        # Transform teks menjadi vektor
        input_vector = vectorizer.transform([input_text])

        # Prediksi
        prediction = model.predict(input_vector)[0]

        # Tampilkan hasil
        if prediction == 1:
            st.success("✅ Berita ini **REAL**.")
        else:
            st.error("🚨 Berita ini **FAKE**.")

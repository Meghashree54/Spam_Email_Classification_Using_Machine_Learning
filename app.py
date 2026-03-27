import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')

model = pickle.load(open("spam_model.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))
ps = PorterStemmer()

st.title("📧 Spam Email Classification")
st.write("Enter an email or message to check if it is Spam or Not")

input_msg = st.text_area("Email / Message Text")

def preprocess(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.split()
    text = [ps.stem(word) for word in text if word not in stopwords.words('english')]
    return ' '.join(text)

if st.button("Check"):
    processed_msg = preprocess(input_msg)
    vectorized_msg = tfidf.transform([processed_msg])
    prediction = model.predict(vectorized_msg)

    if prediction[0] == 1:
        st.error("🚨 This message is SPAM")
    else:
        st.success("✅ This message is NOT SPAM")
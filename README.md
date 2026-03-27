# Spam Email Classification Using Machine Learning

A complete end-to-end project for detecting spam email or SMS messages using natural language processing (NLP) and machine learning.

## 🚀 Project Overview

This repository implements a pipeline that:
- Loads an SMS spam dataset
- Cleans and preprocesses text data
- Vectorizes text using TF-IDF
- Trains a Naive Bayes classifier to detect spam vs ham
- Saves the trained model and vectorizer for inference
- Deploys a simple Streamlit web app (`app.py`) for live predictions

## 📁 Repository Contents

- `spam.csv` - dataset containing labeled messages (`spam` or `ham`)
- `train.py` - training script (preprocess + fit model + save artifacts)
- `app.py` - Streamlit application for prediction
- `spam_model.pkl` - serialized trained model
- `tfidf.pkl` - serialized TF-IDF transformer
- `requirements.txt` - Python dependencies
- `README.md` - project documentation
- `Spam_Email_Classification.ipynb` / `Spam_Email_Classification_Project.ipynb` - exploratory notebooks

## 🛠️ Prerequisites

- Python 3.8+
- Git

## ⚙️ Setup

1. Clone the repository:

```bash
git clone https://github.com/Meghashree54/Spam_Email_Classification_Using_Machine_Learning.git
cd Spam_Email_Classification_Using_Machine_Learning
```

2. (Optional) Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate     # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## 🧠 Train the Model

```bash
python train.py
```

This will generate `spam_model.pkl` and `tfidf.pkl` in the project folder.

## 🌐 Run the Streamlit App

```bash
streamlit run app.py
```

Open the local URL shown in terminal to test sample messages.

## 🧾 File descriptions

- `train.py` handles dataset loading, preprocessing, TF-IDF transformation, model training, evaluation, and artifact persistence.
- `app.py` loads persisted artifacts and provides a web UI for entering text and receiving predictions.

## 📌 Notes

- You can fine-tune model performance by adjusting preprocessing/feature settings and trying different models (Logistic Regression, SVM, etc.).
- If dataset is missing, re-run `train.py` after downloading a fresh SMS Spam Collection dataset.
- This project is ideal for learning text classification and deploying a lightweight interactive app.

## 🤝 License

MIT License


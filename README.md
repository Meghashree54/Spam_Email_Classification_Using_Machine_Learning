# Spam Email Classification Project

Python NLP + ML project to classify email/messages as spam or ham.

## Files
- `app.py`: Streamlit app for live prediction
- `train.py`: Train/load model + save `spam_model.pkl` and `tfidf.pkl`
- `spam.csv`: dataset (SMS Spam Collection  dataset)
- `spam_model.pkl`: trained Naive Bayes model
- `tfidf.pkl`: TF-IDF vectorizer pipeline
- `requirements.txt`: Python dependencies

## Usage
1. Install dependencies: `pip install -r requirements.txt`
2. Train model: `python train.py`
3. Run app: `streamlit run app.py`

## GitHub
To publish to your repo:

```bash
# install git if missing
# configure user
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/Meghashree54/Spam_Email_Classification_Project.git
git push -u origin main
```

If you use GitHub CLI:

```bash
gh auth login
gh repo clone Meghashree54/Spam_Email_Classification_Project
```

# 💬 SENTIMENT-ANALYSIS

**Company: CODTECH IT SOLUTIONS PVT. LTD.**

**Name: SALILA PUNNESHETTY**

**Intern ID: CT04DH2206**

**Domain: DATA ANALYSIS**

**Duration: 4 Weeks**

**Mentor: NEELA SANTOSH**

---

## 🔍 Overview

This project is submitted as part of the **CODTECH Internship**, focused on **Sentiment Analysis using Natural Language Processing (NLP)**. The main objective is to analyze sentiment (positive/negative/neutral) from textual data—specifically, tweets from the [Kaggle Twitter US Airline Sentiment Dataset](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment).

---

## 📁 Folder Structure

SentimentAnalysis_Task4/

│
├── data/

│ └── Tweets.csv

│
├── models/

│ └── sentiment_model.pkl # Trained ML model
│
├── outputs/

│ ├── wordcloud_positive.png

│ └── wordcloud_negative.png

│
├── notebooks/

│ └── SentimentAnalysis.ipynb # Main notebook

│
├── requirements.txt

└── README.md


---

## 📌 Instructions

- Perform **sentiment analysis** on text data using NLP.
- Include in your notebook:
  - 📑 Data preprocessing
  - 🤖 Model implementation
  - 📈 Insights & visualizations

---

## ⚙️ Project Workflow

### 1. Import Libraries
Essential libraries for:
- Data manipulation (`pandas`, `numpy`)
- Text processing (`re`, `nltk`)
- Visualization (`matplotlib`, `seaborn`, `wordcloud`)
- Machine Learning (`sklearn`)
- Model saving (`pickle`)

---

### 2. Load Dataset
- Dataset: `Tweets.csv` (from Kaggle)
- Source: [Twitter US Airline Sentiment Dataset](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment)

---

### 3. Data Preprocessing
- Convert all text to lowercase
- Remove URLs, special characters, and stop words
- Tokenize and clean the tweets
- Store in a new column: `clean_text`

---

### 4. Visualizations
Generate **word clouds** for each sentiment category:

```python
for sentiment in ['positive', 'negative']:
    text = " ".join(df[df['label'] == sentiment]['clean_text'])
    wordcloud = WordCloud(width=600, height=400, background_color='white').generate(text)
    wordcloud.to_file(f'../outputs/wordcloud_{sentiment}.png')
```
### 5. Feature Extraction
```
    Use TfidfVectorizer with max_features=3000:
    tfidf = TfidfVectorizer(max_features=3000, stop_words='english')
```

### Why 3000 features?
To reduce overfitting, speed up training, and focus on the most important words while avoiding sparse matrices.

### 6. Model Training & Evaluation
Split data into training and testing sets

Train a Logistic Regression model

Evaluate using:

Accuracy Score

Classification Report

Confusion Matrix

### 7. Model Saving
Save the trained model using:
```
import pickle
with open('../models/sentiment_model.pkl', 'wb') as f:
    pickle.dump(model, f)
```
## 🛠 How to Use

Place the dataset in data/Tweets.csv

Open and run the notebook in notebooks/SentimentAnalysis.ipynb

## Outputs:

Trained model in models/sentiment_model.pkl & Performance metrics in the notebook cell outputs


<img width="1920" height="1020" alt="Image" src="https://github.com/user-attachments/assets/2bbc4f05-ae55-422f-bfaa-2c668a7a9654" />
Word clouds in outputs

<img width="1920" height="1020" alt="Image" src="https://github.com/user-attachments/assets/21a4a339-0ec4-4563-a571-216df068206b" />

<img width="1920" height="1020" alt="Image" src="https://github.com/user-attachments/assets/d3442f7e-6826-45a7-9f50-b94db78b34cf" />

## Conclusion:
This project provided hands-on experience with Natural Language Processing, enabling effective sentiment classification from real-world Twitter data. It strengthened my understanding of text preprocessing, feature extraction, and model evaluation using Python.

#### *Links & References*

📁 Dataset: Kaggle - Twitter US Airline Sentiment

🛠 WordCloud Docs: https://github.com/amueller/word_cloud

📖 Sklearn TF-IDF Vectorizer[Docs] :https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html



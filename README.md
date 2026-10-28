# 📩 SMS Spam Classifier

A machine learning application that classifies SMS messages as Spam or Not Spam using Natural Language Processing (NLP).

## 🚀 Features

* Classifies messages as Spam or Not Spam.
* Uses TF-IDF for text vectorization.
* Uses Multinomial Naive Bayes for classification.
* Displays the model's predicted class and estimated confidence.
* Provides an interactive web interface using Streamlit.

## 🛠️ Technologies Used

* Python
* Pandas
* Scikit-learn
* TF-IDF
* Multinomial Naive Bayes
* Streamlit
* Joblib

## 📂 Project Structure

```text
SMS-Spam-Classifier/
├── data/
│   └── SMSSpamCollection
├── train.py
├── app.py
├── spam_model.pkl
├── requirements.txt
└── README.md
```

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd SMS-Spam-Classifier
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare the dataset

Download the SMS Spam Collection dataset from the UCI Machine Learning Repository:

https://archive.ics.uci.edu/dataset/228/sms+spam+collection

Extract the archive and place the `SMSSpamCollection` file inside the `data` directory.

### 4. Train the model

```bash
python train.py
```

This trains the classifier, evaluates its performance and saves the model as `spam_model.pkl`.

### 5. Run the application

```bash
streamlit run app.py
```

## 🧠 How It Works

1. The user enters an SMS message.
2. TF-IDF converts the message into numerical features.
3. Multinomial Naive Bayes predicts whether the message is spam.
4. The application displays the predicted class and estimated model confidence.

## 📊 Model Evaluation

The training script reports accuracy, precision, recall and F1-score on a held-out test set.

Add your actual evaluation results here after training.

## ⚠️ Limitations

* Predictions are not guaranteed to be correct.
* The model may struggle with unfamiliar wording or messages that differ from its training data.
* Predicted confidence is not a guarantee of correctness.

## 👨‍💻 Developed By

Ayush

# Text Classification
# Text classification involves categorizing text into different groups. Think about it as sorting emails into spam and non-spam folders or classifying news articles into sports, politics, or entertainment sections.

# These types of models use Python to classify text into predefined categories using a Naive Bayes classifier. A Naive Bayes classifier is a simple and powerful tool in machine learning. It's based on a basic probability rule called Bayes' Theorem and assumes that all features (like words in a text) are independent of each other.

# Naive Bayes works well for tasks like identifying spam emails, analyzing sentiment, and classifying documents. For example, if you want to sort emails into "spam" or "not spam," Naive Bayes can learn from examples and predict the category of a new email based on word patterns

#For this we use the scikit-learn library. This library provides the tools needed for text vectorization, model training and evaluation

## Classes & Functions
# There are classes and functions that are crucial for text classification:

# CountVectorizer: This class converts text data into a numerical format that the machine-learning model can understand. It counts how many times each word appears in the text, turning words into a matrix of counts.

# MultinomialNB: This is a Naive Bayes classifier, which is used to train our model on the numerical text data.

# train_test_split: This function helps split our dataset into training and testing sets. It is commonly used in predictive machine learning. The training set is used to train the model, while the testing set is used to evaluate its performance. Learn more here:

# accuracy_score: This function provides a way to measure the accuracy of our model by comparing the predicted labels with the actual labels in the test set. A higher accuracy score indicates better performance, a score of 1.0 = great predictions.

# These classes and functions are essential for building a text classification model. Let's dive into the code and see how we can classify text data using a Naive Bayes classifier!

import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df=pd.read_csv("A:\Codedex\Generative_AI_Codedex\emails.csv")
print(df.head())

y=df['Prediction']
X_df=df.drop(columns=['Prediction'])
X_text=df['Email No.']
vectorizer = CountVectorizer()
X=vectorizer.fit_transform(X_text)
x_train, x_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=42)

model=MultinomialNB()
model.fit(x_train, y_train)

y_pred_train=model.predict(x_train)
y_pred_test=model.predict(x_test)
print(f"First 5 predictions: {y_pred_test[:100]}")

train_score=accuracy_score(y_train, y_pred_train)
test_score=accuracy_score(y_test, y_pred_test)
print("Train accuracy score:", train_score*100,"%")
print("Test accuracy score:", test_score*100,"%")

#take the content of a spam mail and classify the percentage of it being spam
# Create a test spam email
test_spam_email = """
URGENT! Congratulations! You've WON $1,000,000 in our EXCLUSIVE lottery! 
Click HERE NOW to claim your prize before it expires! 
FREE money waiting for you! Act FAST! Limited time offer!
Call 1-800-GET-RICH immediately! Don't miss this AMAZING opportunity!
100% guaranteed! No purchase necessary! WINNER WINNER!
"""

# Create a test legitimate email
test_legitimate_email = """
Dear Students,

 

This is to inform you that Dean Student Affairs will be interacting with 2nd Year Students instead of 1st Year Students today.

 

B.Tech / iMBA / BBA – 2nd Year: Monday, August 11, 2025 | 6:00 – 6:30 PM
B.Tech / iMBA / iMSc / BBA – 3rd & 4th Year: Thursday, August 14, 2025 | 6:00 – 6:30 PM
 

Date of interaction with first year students will be informed later.

 

Office, Dean Academic Affairs

NIIT University, Neemrana
"""

def predict_email_spam(mail, vectorizer, model):
    converted_mail=vectorizer.transform([mail])
    prediction=model.predict(converted_mail)
    probability=model.predict_proba(converted_mail)
    # prob_class_0 = float(probability[0, 1])
    # prob_class_1 = float(probability[1, 1])
    # return prediction, prob_class_0, prob_class_1
    return prediction, probability


prediction,probability=predict_email_spam(test_spam_email, vectorizer, model)
if prediction==0:
    print("Mail tested is Spam")
else:
    print("Mail tested is not Spam")
print("Probability of being spam",probability[0][0])
print("Probability of not being spam",probability[0][1])

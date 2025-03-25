import pandas as pd
import numpy as np 
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

#load data
file_path= r"C:\Users\manas\OneDrive\Desktop\OASIS\spam.csv"
df=pd.read_csv(file_path, encoding='latin-1')

#drop unnecessary columns and renaming
df=df.iloc[:,:2] #keeping first two columns
df.columns=['label', 'message']

#convert labels to binary
df['label']= df['label'].map({'ham':0, 'spam':1})

#text preprocessing 
def clean_text(text):
    text=text.lower()
    text=re.sub(f"[{string.punctuation}]","",text) #removes punctuation
    text=re.sub(r'\d+','', text) #remove numbers
    text=text.strip()
    return text

#apply preprocessing
df['message'] = df['message'].apply(clean_text)

#split data into training and testing sets
x_train, x_test, y_train, y_test= train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

#convert text to numerical features using TF-IDF

vectorizer= TfidfVectorizer()
x_train_tfidf= vectorizer.fit_transform(x_train)
x_test_tfidf= vectorizer.transform(x_test)

#train a naive bayes classifier

model= MultinomialNB()
model.fit(x_train_tfidf,y_train)

#make prediction
y_pred= model.predict(x_test_tfidf)

#evaluate model

accuracy= accuracy_score(y_test, y_pred)
print(f'Accuracy:, {accuracy:.2f}')
print(classification_report(y_test,y_pred))

#function to predict new emails

def predict_email(email_text):
    email_text= clean_text(email_text)
    email_tfidf= vectorizer.transform([email_text])
    prediction=model.predict(email_tfidf)
    return "Spam" if prediction[0]== 1 else "Not Spam"

#example
print(predict_email("Congratulations! you won an iphone!!!"))

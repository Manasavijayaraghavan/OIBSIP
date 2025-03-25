import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#loading the dataset
file_path=r"C:\Users\manas\OneDrive\Desktop\OASIS\Iris.csv"
data=pd.read_csv(file_path)

#ID coloumn is irrelevant to the task provided, hence removing it

data= data.drop(columns=["Id"])

#exploring dataset
print(data.head())
print(data.info())
print(data.describe())

#visualise the data

sns.pairplot(data, hue="Species")
plt.show()

#split data set into features and trgt variable

x= data.drop(columns=["Species"])
y= data["Species"]

#train test split
x_train, x_test, y_train, y_test= train_test_split(x,y, test_size=0.2, random_state=42)

#model training
model= RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train, y_train)

#predictions
y_pred=model.predict(x_test)

#model evaluation
accuracy= accuracy_score(y_test,y_pred)
classfrep= classification_report(y_test, y_pred)
confusionmat= confusion_matrix(y_test, y_pred)

#print 

print("Accuracy:", accuracy)
print("Classfication report:\n", classfrep)
print("Confusion matrix:\n", confusionmat)

#feature importance

feature_importances=pd.Series(model.feature_importances_, index=x.columns)
feature_importances.sort_values(ascending=False).plot(kind='bar', title='Feature Importance')
plt.show()


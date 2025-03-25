import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#load data
file_path=r"C:\Users\manas\OneDrive\Desktop\OASIS\Advertising.csv"
df=pd.read_csv(file_path)

#display first few rows
print(df.head())

#Check for missing values
print(df.isnull().sum())

#select features and target variables

x= df[['TV', 'Radio', 'Newspaper']] #independent
y= df['Sales'] #dependent

#split data 

x_train, x_test, y_train, y_test= train_test_split(x,y, test_size=0.2, random_state=42)

#training a linear reg model

model= LinearRegression()
model.fit(x_train, y_train)

#prediction

y_pred= model.predict(x_test)

#evaluate model

mae= mean_absolute_error(y_test, y_pred)
mse= mean_squared_error (y_test, y_pred)
rmse= np.sqrt(mse)
r2= r2_score(y_test, y_pred)

print(f'Mean Absolute Error: {mae:.2f}')
print(f'Mean Squared Error: {mse:.2f}')
print(f'RMSE: {rmse:.2f}')
print(f'R2 Score: {r2:.2f}')

#Function to predict sales based on advertising spend

def predict_sales(Tv, Radio, Newspaper):
    input_data=np.array([[Tv,Radio,Newspaper]])
    prediction= model.predict(input_data)
    return prediction[0]

#example

print(predict_sales(230, 37, 69))







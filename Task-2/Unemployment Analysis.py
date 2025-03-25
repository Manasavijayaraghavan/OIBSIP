import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Loaing the data set
file_path= r"C:\Users\manas\OneDrive\Desktop\OASIS\Unemployment_Rate_upto_11_2020.csv"
data=pd.read_csv(file_path)

#explore data set
print(data.head())
print(data.info())

data.columns=data.columns.str.strip()

#Converting Date column to date time format

data['Date']= pd.to_datetime(data['Date'], errors='coerce')

#check for missing values

print("Missing vals:")
print(data.isnull().sum())

#Summary statistics

print("Summary Statistics:")
print(data.describe())

#Line plot of unemployment rate

plt.figure(figsize=(12,6))
sns.lineplot(x='Date', y='Estimated Unemployment Rate (%)', data=data, marker='o', color='r')
plt.title("UNEMPLOYMENT RATE OVER TIME")
plt.xlabel("Date")
plt.ylabel("Unemployment rate %")
plt.xticks(rotation=45)
plt.show()

#Regional Unemployment rate analysis

plt.figure(figsize=(14,7))
sns.boxplot(x='Region', y= 'Estimated Unemployment Rate (%)', data=data)
plt.xticks(rotation=90)
plt.title("UNEMPLOYMENT RATE BY REGION")
plt.show()

#Heatmap 
plt.figure(figsize=(10,5))
sns.heatmap(data.corr(numeric_only=True), annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Correleation HeatMap")
plt.show()

print("Analysis Complete!")

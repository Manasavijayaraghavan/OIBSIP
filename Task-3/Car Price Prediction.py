import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

#Loading dataset
file_path= r"C:\Users\manas\OneDrive\Desktop\OASIS\car data.csv"
df=pd.read_csv(file_path)

#feature engineering
df["Car_Age"]=2025- df["Year"] #assuming year as 2025
df.drop(["Year", "Car_Name"], axis=1, inplace=True) #redudant columns dropped

#Categorical features

encoder=OneHotEncoder(drop='first', sparse_output=False)
categorical_cols=["Fuel_Type", "Selling_type", "Transmission"]
encoded_data=encoder.fit_transform(df[categorical_cols])
encoded_df= pd.DataFrame(encoded_data, columns= encoder.get_feature_names_out(categorical_cols))

df=pd.concat([df.drop(categorical_cols, axis=1), encoded_df], axis=1)

#splitting data 

x= df.drop("Selling_Price", axis=1)
y=df["Selling_Price"]

x_train, x_test, y_train, y_test= train_test_split(x,y, test_size=0.2, random_state=42)

#Model Training

model= RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(x_train, y_train)

#Prediction

y_pred= model.predict(x_test)

#Evaluation
r2= r2_score(y_test,y_pred)
rmse=np.sqrt(mean_squared_error(y_test, y_pred))

print(f"R^2 Score: {r2:4f}")
print(f"RMSE: {rmse:4f}")

#Feature importance
feature_importances=pd.Series(model.feature_importances_, index=x.columns).sort_values(ascending=False)

plt.figure(figsize=(10,5))
sns.barplot(x=feature_importances, y= feature_importances.index)
plt.xlabel("Feature Importance Score")
plt.ylabel("Features")
plt.title("Feature Importance")

plt.show()
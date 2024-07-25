import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle


data = pd.read_csv('train.csv', low_memory=False)

data["Date"]  = pd.to_datetime(data['Date'])

data['Year']  = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
data['Day']   = data['Date'].dt.day

features = ['Store', 'DayOfWeek', 'Promo', 'Year', 'Month', 'Day', 'SchoolHoliday']

X = data[features]
y = data['Sales']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)

model.fit(X_train[:2000], y_train[:2000])

y_pred = model.predict(X_test)

with open('random_forest_model.pkl','wb') as file:
    pickle.dump(model, file)
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error, mean_absolute_error 
from sklearn import preprocessing 
import joblib
from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

import warnings
warnings.filterwarnings("ignore")


data = pd.read_csv('Diwali Sales Data.csv', encoding='unicode_escape')


data = data.drop(columns=['Status','unnamed1'])
data.dropna(inplace=True)
data = data.reset_index(drop=True)

maps={}

#Gives labels Categorical data
def load_data(data):
    label = LabelEncoder()
    cols = ['Gender','Age Group','Marital_Status','State','Zone','Occupation','Product_Category']
    for col in cols:
        data[col] = label.fit_transform(data[col])
        # mapping = dict(zip(label.classes_, range(1, len(label.classes_)+1)))
        # maps[col] = mapping
    return data

def split(df):
    y = df['Amount']
    y = y.apply(np.int64) 
    x = df.drop(columns=['User_ID','Cust_name','Product_ID','Amount','Orders','Age'])
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=0)
    return x_train, x_test, y_train, y_test

n_estimators = 100
max_depth = 20

df = load_data(data)
x_train, x_test, y_train,y_test = split(df)
x_train = x_train.drop(columns=['level_0','index'])
x_test = x_test.drop(columns=['level_0','index'])
model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, n_jobs=-1, criterion='mse')
# model = LinearRegression()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
model_file_name = "final.joblib"
joblib.dump(model, model_file_name)

# print(maps)
mse = mean_squared_error(y_test, y_pred)

print("Mean Squared Error:", mse)
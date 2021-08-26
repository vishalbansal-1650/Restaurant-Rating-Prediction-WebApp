import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor

import pickle
import warnings
warnings.filterwarnings(action='ignore')


df = pd.read_csv('data/data_zomato.csv')
df.drop('Unnamed: 0',axis=1,inplace=True)
print(df.head())

x = df.drop('rate',axis=1)
y = df['rate']

x_train, x_test, y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=101)

ET_model = ExtraTreesRegressor(n_estimators=150,random_state=101)
ET_model.fit(x_train,y_train)

y_test_pred = ET_model.predict(x_test)

pickle.dump(ET_model,open('model/model.pkl','wb'))

model = pickle.load(open('model/model.pkl','rb'))
print(y_test_pred)




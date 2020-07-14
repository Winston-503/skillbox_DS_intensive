# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 20:35:00 2020
First day of free Data Science intensive by Skillbox

@author: DimiD
"""

intensive = "Data Science"
print(f"You look intensive on {intensive}")

import pandas as pd

trips = pd.read_excel("trips_data.xlsx", index_col=0)

trips.describe()  # Mean, std, min and max

trips.head()  # First 5 rows
trips.tail()  # Last 5 rows

trips.age  # Get age colomn

trips.age.hist()  # Histograms
trips.salary.hist()

trips.vacation_preference.value_counts()  # Unique values

# Histograms for unique values
trips.vacation_preference.value_counts.plot(kind="bar")
trips.city.value_counts.plot(kind="bar")

# Income for each family member
(trips.salary / (trips.family_members + 1)).hist()

# Transformation city column to lot of boolean columns
pd.get_dummies(trips, columns=['city'])

trips_no_strings = pd.get_dummies(trips, \
    columns=['city', 'vacation_preference', 'transport_preference'])

X = trips_no_strings.drop('target', axis=1)  # Data on base
y = trips_no_strings.targer  # Result, that we try to predict

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()  # Creating a model

model.fit(X, y)  # Learning a model

# Data, that we want to predict
myData = {'salary': [50000],
 'age': [20],
 'family_members': [1],
 'city_Екатеринбург': [0],
 'city_Киев': [0],
 'city_Краснодар': [0],
 'city_Минск': [0],
 'city_Москва': [0],
 'city_Новосибирск': [0],
 'city_Омск': [0],
 'city_Петербург': [1],
 'city_Томск': [0],
 'city_Хабаровск': [0],
 'city_Ярославль': [0],
 'vacation_preference_Архитектура': [0],
 'vacation_preference_Ночные клубы': [0],
 'vacation_preference_Пляжный отдых': [1],
 'vacation_preference_Шоппинг': [0],
 'transport_preference_Автомобиль': [0],
 'transport_preference_Космический корабль': [0],
 'transport_preference_Морской транспорт': [0],
 'transport_preference_Поезд': [0],
 'transport_preference_Самолет': [1]}

# Creating DF object based on myData
myDF = pd.DataFrame(myData, columns=X.columns)

model.predict([myDF])  # Prediction

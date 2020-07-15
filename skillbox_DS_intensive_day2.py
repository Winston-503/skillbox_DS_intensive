# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 19:19:06 2020
Second day of free Data Science intensive by Skillbox

@author: DimiD
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error

# Function for checking model by plots and mean_absolute_error
def check_model(model=None, name="Model", day=1):
    print(f"Day {day} {name}")
    # iloc -> to list, not DF
    prediction = model.predict([X_test.iloc[day-1]])[0]
    real = Y_test.iloc[day-1]  
    mae = mean_absolute_error(prediction, real)
    print(f"mae = {mae}")
    plt.plot(prediction, label="Prediction")
    plt.plot(real, label="Real")
    plt.legend()
    plt.show()


# Dollar rate 01.01.2018 - 01.07.2020
usd_rates = pd.read_excel("usd_rates.xlsx")

usd_rates.curs.hist()
usd_rates.curs.plot()

future = 7  # Prediction for "future" days
past = 28  # Prediction is based on "past" days

money = usd_rates.curs

start = past  # Day, when we begin to collect training sets
end = money.size - future  # Day, when we finish
print(f"{start} ... {end} = {end - start}")

training_past = []
training_future = []
for day in range(start, end):
    past_data = money[(day-past):day]
    future_data = money[day:(day+future)]
    training_past.append(list(past_data))
    training_future.append(list(future_data))

# Genarating names for DF columns
future_columns = []
for i in range(future):
    future_columns.append(f"future_{i}")
past_columns = []
for i in range(past):
    past_columns.append(f"past_{i}")

# Data based on
past_df = pd.DataFrame(training_past, columns=past_columns)
# Data, that we try to predict
future_df = pd.DataFrame(training_future, columns=future_columns)  

# Training set
X_train = past_df[:-10]
Y_train = future_df[:-10]
# Test set
X_test = past_df[-10:]
Y_test = future_df[-10:]

forest = RandomForestRegressor()
forest.fit(X_train, Y_train)
check_model(model=forest, name="RandomForestRegressor", day=1)

forest = RandomForestRegressor(n_estimators=1000)
forest.fit(X_train, Y_train)
check_model(model=forest, name="RandomForestRegressor with n_estimators=1000", day=1)

regression = LinearRegression()
regression.fit(X_train, Y_train)
check_model(model=regression, name="LinearRegression", day=1)

regression = LinearRegression(normalize=True)
regression.fit(X_train, Y_train)
check_model(model=regression, name="LinearRegression with normalize=True", day=1)

mlp = MLPRegressor()
mlp.fit(X_train, Y_train)
check_model(model=mlp, name="MLPRegressor", day=1)

mlp = MLPRegressor(max_iter=2000)
mlp.fit(X_train, Y_train)
check_model(model=mlp, name="MLPRegressor with max_iter=2000", day=1)

mlp = MLPRegressor(max_iter=2000, hidden_layer_sizes=(100,100), random_state=42)
mlp.fit(X_train, Y_train)
check_model(model=mlp, name="MLPRegressor with max_iter=2000 and ...", day=1)

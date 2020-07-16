# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 20:35:00 2020
Third day of free Data Science intensive by Skillbox

@author: DimiD
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.neural_network import MLPRegressor

# Function for checking model by plots and mean_absolute_error
def check_model(model=None, name="Model", day=1):
    # iloc -> to list, not DF
    prediction = model.predict([X_test.iloc[day-1]])[0]
    real = Y_test.iloc[day-1]  
    mae = mean_absolute_error(prediction, real)
    print(f"For {name} on {day} day mean abs. error = {mae}")
    # Two plots on one axis
    plt.plot(prediction, label="Предсказание")
    plt.plot(real, label="Реальные данные")
    plt.legend()
    plt.show()

# Data preparation to load it on model
# Dollar rate 01.01.2018 - 01.07.2020
usd_rates = pd.read_excel("usd_rates.xlsx")

future = 7  # Prediction for "future" days
past = 28  # Prediction is based on "past" days

money = usd_rates.curs
start = past  # Day, when we begin to collect training sets
end = money.size - future  # Day, when we finish

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


from sklearn.neural_network import MLPRegressor

mlp = MLPRegressor()
mlp.fit(X_train, Y_train)
check_model(mlp, "MLPRegressor")

# GridSearchCV helps to select good parameters for model
#
# For example, we want to check out
# max_iter = 100, 500, 1000, 2000
# learning_rate_init = 0.001, 1, 0.1, 0.01
#
# CV - cross validation
# fit model several times on differents parts of data
#
# * - train, Х - test
# [XXXX*****************]
# [****XXXX*************]
# [********XXXX*********]
# [*************XXXX****]
# [*****************XXXX]

from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor

KNR = KNeighborsRegressor(p=2)
param_grid = {
    "n_neighbors": [1, 2, 4, 8],
    "weights": ['uniform', 'distance']
}

# Parameters of GridSearchCV: model, grid of parameters 
# quality criterion (error function) and count of CV (partitions)
gs = GridSearchCV(KNR, param_grid, "neg_mean_absolute_error", cv=4)
gs.fit(X_train, Y_train)
gs.best_score_  # best error
gs.best_params_  # best parameters
best_model = gs.best_estimator_  # best model

KNR_model = KNeighborsRegressor(p=2)
KNR_model.fit(X_train, Y_train)

check_model(KNR_model, "KNeighborsRegressor without GridSearchCV", 4)
check_model(best_model, "KNeighborsRegressor with GridSearchCV", 4)

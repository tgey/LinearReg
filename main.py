#!/usr/bin/python3

import sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def main(argv, prediction_days=30):
    df = pd.read_csv(
    'dataset.csv')
    df['date'] = pd.to_datetime(df['Timestamp'], unit='s').dt.date
    group = df.groupby('date')
    Real_Price = group['Weighted_Price'].mean()

    if len(argv) >= 2:
        try:
            prediction_days = int(argv[1])
        except ValueError:
            prediction_days = 30
    df_train = Real_Price[:len(Real_Price)-prediction_days]
    df_test = Real_Price[len(Real_Price)-prediction_days:]

    training_set = df_train.values
    training_set = training_set.reshape(-1, 1)
    sc = MinMaxScaler()
    training_set = sc.fit_transform(training_set)
    X_train = training_set[0:len(training_set)-1]
    y_train = training_set[1:len(training_set)]

    test_set = df_test.values
    test_set = test_set.reshape(-1, 1)
    sc = MinMaxScaler()
    test_set = sc.fit_transform(test_set)
    X_test = test_set[0:len(test_set)-1]
    y_test = test_set[1:len(test_set)]

    # Create linear regression object
    regr = LinearRegression()

    # Train the model using the training sets
    regr.fit(X_train, y_train)

    # Make predictions using the testing set
    y_pred = regr.predict(X_test)

    # The coefficients
    print('Coefficients: \n', regr.coef_)
    # The mean squared error
    print("Mean squared error: %.5f"
          % mean_squared_error(y_test, y_pred))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.5f' % r2_score(y_test, y_pred))

    # Plot outputs
    plt.title(str(prediction_days) + ' days predicted')
    plt.scatter(X_test, y_test,  color='black')
    plt.plot(X_test, y_pred, color='blue', linewidth=3)

    plt.xticks(())
    plt.yticks(())

    plt.show()



if __name__ == "__main__":
    main(sys.argv)

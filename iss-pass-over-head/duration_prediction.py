"""Module for predicting the duration of ISS over Chandigarh."""

import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as seabornInstance
import sys

from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
# from sklearn import metrics %matplotlib inline

class DurationPrediction:

  def __init__(self):
    cwd = os.getcwd()

    # Read the data from CSV. Mark the datetime fields.
    self.dataframe = pd.read_csv(
      os.path.join(cwd, 'data/combined_csv.csv'),
      parse_dates=['Start Time', 'End Time']).sort_values(by='Start Time')
    rows, cols = self.dataframe.shape
    logging.info(f'Data size:  Rows = {rows}, Coumns = {cols}')
    # logging.info('*' * 20)
    # logging.info(self.dataframe.describe())
    # logging.info('*' * 20)
    # logging.info(self.dataframe.info())
    logging.info('*' * 20)
    logging.info(self.dataframe.head())

  def plot_data(self):
    self.dataframe.plot(x='Start Time', y='Duration', style='o')
    plt.title('Start Time vs Duration')
    plt.xlabel('Start Time')
    plt.ylabel('Duration')
    plt.show()
    # plot_average_duration(plt, self.dataframe)

  def plot_average_duration(self, plt, dataframe):
    plt.figure(figsize=(15,10))
    plt.tight_layout()
    seabornInstance.distplot(dataframe['Duration'])

  def mark_attributes_label(self, attr='Start Time', label='Duration'):
    # Mark Start Time as attribute.
    X = self.dataframe[attr].values.reshape(-1, 1)
    # Mark Duration as attribute.
    y = self.dataframe[label].values.reshape(-1, 1)
    return X, y
  
  def split_data(self):
    """ Split data in training and test sets.
    
    Split 80% of the data to the training set while 20% of the data to test
    set using below code.
    """
    X, y = self.mark_attributes_label()
    X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.2, random_state=0)
    
    regressor = LinearRegression()
    regressor.fit(X_train, y_train) #training the algorithm
    #To retrieve the intercept:
    print(regressor.intercept_)
    #For retrieving the slope:
    print(regressor.coef_)
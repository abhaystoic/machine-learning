"""Module for predicting the duration of ISS over Chandigarh."""

import datetime
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as seabornInstance
import sys

from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics

class DurationPrediction:

  def __init__(self):
    cwd = os.getcwd()

    # Read the data from CSV. Mark the datetime fields.
    self.dataframe = pd.read_csv(
      os.path.join(cwd, 'data/combined_csv.csv'),
      parse_dates=['Start Time', 'End Time'],
      index_col='Start Time').sort_values(by='Start Time')
    self.clean_dates()
  
  def clean_dates(self):
    """ Convert dates to ordinal.

    Linear regression doesn't work on date data. Therefore we need to
    convert it into numerical value.
    """
    # self.dataframe['Start Time'] = pd.to_datetime(
    #   self.dataframe['Start Time'], infer_datetime_format=True)
    # # self.dataframe['Start Time'] = self.dataframe['Start Time'].map(
    # #   datetime.datetime.toordinal)
    # self.dataframe = self.dataframe.set_index('Start Time')
    # self.dataframe['End Time'] = pd.to_datetime(
    #   self.dataframe['End Time'], infer_datetime_format=True)
    # # self.dataframe['End Time'] = self.dataframe['End Time'].map(
    # #   datetime.datetime.toordinal)

    rows, cols = self.dataframe.shape
    logging.info(f'Data size:  Rows = {rows}, Coumns = {cols}')
    # logging.info('*' * 20)
    # logging.info(self.dataframe.describe())
    # logging.info('*' * 20)
    # logging.info(self.dataframe.info())
    logging.info('*' * 20)
    logging.info(self.dataframe.head())
    # self.plot_df(title='ISS Pass Overhead. Location: Chandiarh, India.')
    self.plot_monthly_distribution()

  
  def plot_df(self, title='', xlabel='Date', ylabel='Value', dpi=100):
    x = self.dataframe.index
    y = self.dataframe.Duration
    plt.figure(figsize=(16,5), dpi=dpi)
    plt.plot(x, y, color='tab:red')
    plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
    plt.show()
  
  def plot_monthly_distribution(self):
    df = self.dataframe
    df.reset_index(inplace=True)

    # Prepare data
    df['year'] = [d.year for d in df['Start Time']]
    df['month'] = [d.strftime('%b') for d in df['Start Time']]
    years = df['year'].unique()

    # Draw Plot
    fig, axes = plt.subplots(1, 2, figsize=(20,7), dpi= 80)
    seabornInstance.boxplot(x='year', y='Duration', data=df, ax=axes[0])
    seabornInstance.boxplot(x='month',
                y='Duration',
                data=df.loc[~df.year.isin([1991, 2020]), :])

    # Set Title
    axes[0].set_title('Year-wise Box Plot\n(The Trend)', fontsize=18)
    axes[1].set_title('Month-wise Box Plot\n(The Seasonality)', fontsize=18)
    plt.show()

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
    # Split the data.
    X, y = self.mark_attributes_label()
    X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.2, random_state=0, shuffle=False)
    
    self.predict(X_train, X_test, y_train, y_test)
  
  def predict(self, X_train, X_test, y_train, y_test):
    # Training the algorithm.
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    # To retrieve the intercept:
    logging.info('Intercept = {}'.format(regressor.intercept_))
    # For retrieving the slope:
    logging.info('Slope = {}'.format(regressor.coef_))

    y_pred = regressor.predict(X_test)
    # df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
    
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

    self.plot_predictions(X_train, X_test, y_train, y_test, y_pred)

  def plot_predictions(self, X_train, X_test, y_train, y_test, y_pred):
    plt.scatter(X_test, y_test,  color='gray')
    plt.plot(X_test, y_pred, color='red', linewidth=2)
    plt.show()

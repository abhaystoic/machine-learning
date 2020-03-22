""""""
from config_logger import configure_logging
from duration_prediction import DurationPrediction


if __name__ == '__main__':
  configure_logging()
  dp = DurationPrediction()
  dp.split_data()
  dp.plot_data()

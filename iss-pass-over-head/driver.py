"""Driver module for predictions."""

from config_logger import configure_logging
from duration_prediction import DurationPrediction


if __name__ == '__main__':
  configure_logging()
  dp = DurationPrediction()
  dp.plot_data()
  dp.split_predict()

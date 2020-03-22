"""Module for configure the logger."""

import logging


def configure_logging():
  """Method to cofigure logger to outpur on STDERR."""
  # Create logger.
  logger = logging.getLogger()
  logger.setLevel(logging.DEBUG)
  ch = logging.StreamHandler()
  ch.setLevel(logging.DEBUG)
  # Add ch to logger.
  logger.addHandler(ch)

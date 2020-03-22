"""Script to merge CSV files."""

import glob
import os
import pandas as pd

def merge():
  os.chdir('../data')
  all_file_names = set()
  for file in glob.glob('*.csv'):
    all_file_names.add(file)
  
  combined_csv = pd.concat([pd.read_csv(f) for f in all_file_names], sort=True)
  combined_csv.to_csv("combined_csv.csv", index=False, encoding='utf-8-sig')

if __name__ == '__main__':
  merge()
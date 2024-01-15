import json
import numpy as np

def generate_permutation(size):
  perm = np.random.permutation(size)
  return perm

def read_json():
  with open("config.json", "r") as config_file:
    config_data = json.load(config_file)
    return config_data
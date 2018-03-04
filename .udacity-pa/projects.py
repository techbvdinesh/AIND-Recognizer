import argparse
import shutil
import os
from udacity_pa import udacity

nanodegree = 'nd889'
projects = ['asl_recognizer']

def submit(args):
  filenames = ['asl_recognizer.ipynb', 'asl_recognizer.html','my_model_selectors.py','my_recognizer.py']

  udacity.submit(nanodegree, projects[0], filenames, 
                 environment = args.environment,
                 jwt_path = args.jwt_path)

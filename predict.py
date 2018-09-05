#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Sample the RNN poetry language model
'''

from pickle import load
from collections import Counter
from tensorflow import logging

from constants import *
from models import PoetryRNNModel, Word2VecModel

logging.set_verbosity(logging.ERROR)

__author__ = "James Dorfman"
__copyright__ = "Copyright 2018, James Dorfman"
__license__ = "GNU"

try:
    with open(RNN_PICKLE_FILE, 'rb') as pickle_input:
        rnn = load(pickle_input)
except Exception as e:
    exit("\nPoetryRNNModel storage file not found. Please run `train.py` first. \n\nExiting...")

rnn.init_model()

print('\nSAMPLE FROM LANGUAGE MODEL')
print('-' * 50)
print(rnn.sample())

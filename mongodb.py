# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 04:02:07 2020

@author: Derek
"""

# Load Libraries & Dependencies
from nltk.corpus import stopwords
from nltk import ngrams, FreqDist
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

import pandas as pd
import numpy as np
import string
import nltk
import re

import pymongo

password = input('Enter password: ')

client = pymongo.MongoClient("mongodb+srv://aidly: " + password + "@aidly-testing-data-yi9tx.mongodb.net/test?retryWrites=true&w=majority")
db = client.test





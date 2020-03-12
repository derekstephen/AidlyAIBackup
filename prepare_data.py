# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 17:37:37 2020

@author: Derek
"""

# Load Libraries & Dependencies
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
import pandas as pd
import numpy as np
import string
import nltk
import re



# First time download stop words
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Load Stop Words
stop_words = stopwords.words('english')

# Import data
df = pd.read_csv("./Data/MISSION.csv")


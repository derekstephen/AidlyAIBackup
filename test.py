# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 05:48:32 2020

@author: Derek
"""

# Load Libraries & Dependencies
from nltk.corpus import stopwords
from nltk import ngrams, FreqDist
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer

import pandas as pd
import numpy as np
import string
import nltk
import re

# First time download stop words
#nltk.download('stopwords')
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')

# Load Stop Words
stop_words = stopwords.words('english')

# Import data
df = pd.read_csv("./Data/MISSION.csv")

split_mission = df.iloc[8]["F9_03_PZ_MISSION"].lower().split()

#print(split_mission)

token_mission = nltk.sent_tokenize(df.iloc[8]["F9_03_PZ_MISSION"].lower())

#print(token_mission)

token_word_mission = [nltk.word_tokenize(sent) for sent in token_mission]

#print(token_word_mission)


token_remove_stop = [word for word in [sent for sent in nltk.sent_tokenize(df.iloc[8]["F9_03_PZ_MISSION"].lower())]]

#print(token_remove_stop)

#for sent in token_word_mission:
#    for word in sent:
#        if word not in stop_words:
#            pass

test = [item for item in [nltk.word_tokenize(word) for word in [sent for sent in nltk.sent_tokenize(df.iloc[8]["F9_03_PZ_MISSION"].lower())]] if item not in stop_words]

print(test)
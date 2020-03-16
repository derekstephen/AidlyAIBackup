#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 02:55:49 2020
@author: ddetommaso12
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

# PREPARE DATA CODE


def prep_text(mission):
    sentences = nltk.sent_tokenize(mission)
    sentences = [nltk.word_tokenize(sent) for sent in sentences]
    return sentences


# First time download stop words
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Load Stop Words
stop_words = stopwords.words('english')

# Import data
df = pd.read_csv("./Data/MISSION.csv")

# Split Mission to Sentences and then Words
df_missions = df["F9_03_PZ_MISSION"].apply(lambda x: prep_text(str(x).lower()))
df["MISSION"] = df_missions

# Flatten Separated Words to one List
df["WORDS"] = df["MISSION"].apply(lambda column: [y for x in column for y in x])

# Remove Stop Words
df["WORDS"] = df["WORDS"].apply(lambda x: [item for item in x if item not in stop_words])

# END PREPARE DATA CODE

# Start Step One Code

# End Step One Code

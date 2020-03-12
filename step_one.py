#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 02:55:49 2020

@author: ddetommaso12
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

# Remove unnecessary columns and rename mission column
df = df[['EIN', 'NAME', 'F9_03_PZ_MISSION']]
df = df.rename(columns={'F9_03_PZ_MISSION': 'MISSION'})

# Grab Mission statement to test
text = df.iloc[7]

text["MISSION"] = text["MISSION"].lower()

sentences = nltk.sent_tokenize(text["MISSION"])
sentences = [nltk.word_tokenize(sent) for sent in sentences]
sentences = [nltk.pos_tag(sent) for sent in sentences]

print(sentences)





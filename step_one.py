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

print("\n\n", nltk.word_tokenize(nltk.sent_tokenize(str(df.iloc[7]["MISSION"]).lower())), "\n\n")

# Remove Stop Words
df_missions = df["MISSION"].apply(lambda x: [item for item in nltk.word_tokenize(nltk.sent_tokenize(str(x).lower())) if item not in stop_words])
df["MISSION"] = df_missions




# Grab Mission statement to test
text = df.iloc[7]


# END PREPARE DATA CODE

# Tokenize and add POS Tags
#sentences = nltk.sent_tokenize(text["MISSION"])
sentences = [nltk.word_tokenize(sent) for sent in text["MISSION"]]
sentences = [nltk.pos_tag(sent) for sent in sentences]

print(sentences)




# Word Frequency
fdist = FreqDist()
for word in word_tokenize(text["MISSION"]):
    fdist[word.lower()] += 1
    
print(fdist.most_common(4))

# TF-IDF
tfidf = TfidfVectorizer()
















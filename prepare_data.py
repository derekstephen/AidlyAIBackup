# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 17:37:37 2020

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
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Load Stop Words
stop_words = stopwords.words('english')

# Import data
df = pd.read_csv("./Data/MISSION.csv")

"""

text = df.iloc[0]

mission = text["F9_03_PZ_MISSION"].lower()

sent_tokenized = nltk.sent_tokenize(mission)

word_tokenized = [nltk.word_tokenize(sent) for sent in sent_tokenized]

print(word_tokenized)


"""


# Make mission lowercase & Remove Stop Words
df_missions = df["F9_03_PZ_MISSION"].apply(lambda x: item for item in (nltk.word_tokenize(sent) for sent in nltk.sent_tokenize(str(x).lower())) if item not in stop_words)
df["F9_03_PZ_MISSION"] = df_missions

# Print Example Mission Statement
print(df.iloc[0])

pos_tags = [nltk.pos_tag(sent) for sent in df.iloc[0]["F9_03_PZ_MISSION"]]







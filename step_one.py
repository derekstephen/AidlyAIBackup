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

df["OLD_MISSION"] = df["F9_03_PZ_MISSION"]
df = df.apply(lambda x: x.astype(str).str.lower())

# Make mission lowercase & Remove Stop Words
df_missions = df["F9_03_PZ_MISSION"].apply(lambda x: [item for item in nltk.sent_tokenize(str(x).lower()) if item not in stop_words])
df["F9_03_PZ_MISSION"] = df_missions

# Remove unnecessary columns and rename mission column
df = df[['EIN', 'NAME', 'F9_03_PZ_MISSION', 'OLD_MISSION']]
df = df.rename(columns={'F9_03_PZ_MISSION': 'MISSION'})

# Grab Mission statement to test
<<<<<<< HEAD
<<<<<<< HEAD
text = df.iloc[0]


# END PREPARE DATA CODE

# Tokenize and add POS Tags
#sentences = nltk.sent_tokenize(text["MISSION"])
sentences = [nltk.word_tokenize(sent) for sent in text["MISSION"]]
=======
text = df.iloc[7]

text["MISSION"] = text["MISSION"].lower()

=======
text = df.iloc[7]

text["MISSION"] = text["MISSION"].lower()

>>>>>>> parent of fb68bbd... Update step_one.py
sentences = nltk.sent_tokenize(text["MISSION"])
sentences = [nltk.word_tokenize(sent) for sent in sentences]
>>>>>>> parent of fb68bbd... Update step_one.py
sentences = [nltk.pos_tag(sent) for sent in sentences]

print(sentences)




<<<<<<< HEAD
<<<<<<< HEAD
# Word Frequency
fdist = FreqDist()
for word in word_tokenize(" ".join(df.iloc[0]["MISSION"])):
    fdist[word.lower()] += 1
    
print(fdist.most_common(4))
=======
>>>>>>> parent of fb68bbd... Update step_one.py
=======
>>>>>>> parent of fb68bbd... Update step_one.py

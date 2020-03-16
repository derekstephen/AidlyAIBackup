# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 02:55:49 2020
@author: ddetommaso12
"""

# Load Libraries & Dependencies
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

from sklearn.feature_extraction.text import TfidfVectorizer

import pandas as pd
import nltk

# PREPARE DATA CODE


def prep_text(mission):
    """Preps mission statement by tokenizing sentences and words."""
    sentences = nltk.sent_tokenize(mission)
    sentences = [nltk.word_tokenize(sent) for sent in sentences]
    return sentences


# First time download stop words
nltk.download('stopwords')

# Load Stop Words
stop_words = stopwords.words('english')

# Import data
df = pd.read_csv("./Data/MISSION.csv")

# Split Mission to Sentences and then Words
df["MISSION"] = df["F9_03_PZ_MISSION"].apply(lambda x: prep_text(str(x).lower()))

# Flatten Separated Words to one List
df["WORDS"] = df["MISSION"].apply(lambda column: [y for x in column for y in x])

# Remove Stop Words
df["WORDS"] = df["WORDS"].apply(lambda x: [item for item in x if item not in stop_words])

# END PREPARE DATA CODE

# START STEP ONE CODE

# Create Porter Stemmer
stemmer = SnowballStemmer("english")

# Stem mission statements
df["STEMMER"] = df["WORDS"].apply(lambda x: [stemmer.stem(word) for word in x])



# END STEP ONE CODE

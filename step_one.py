# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 02:55:49 2020
@author: ddetommaso12
"""

# Load Libraries & Dependencies
from nltk.corpus import stopwords
from nltk.stem import snowball, WordNetLemmatizer

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

# Separate Mission to Sentences
df["MISSION"] = df["F9_03_PZ_MISSION"].apply(lambda x: nltk.sent_tokenize(str(x).lower()))

# Split Mission to Sentences and then Words
df["WORD"] = df["F9_03_PZ_MISSION"].apply(lambda x: prep_text(str(x).lower()))

# Flatten Separated Words to one List
df["WORD"] = df["WORD"].apply(lambda column: [y for x in column for y in x])

# Remove Stop Words
df["WORD"] = df["WORD"].apply(lambda x: [item for item in x if item not in stop_words])

# END PREPARE DATA CODE

# START STEP ONE CODE

# First time download wordnet
nltk.download('wordnet')

# Create Porter Stemmer
stemmer = snowball.SnowballStemmer('english')

# Stem mission statements
df["STEMMER"] = df["WORD"].apply(lambda x: [stemmer.stem(word) for word in x])

# Get POS for each word to use in Lemmatizer

# Create WordNet Lemmatization
wordnet_lemmatizer = WordNetLemmatizer()

# Lemmatization of Words
df["LEMMATIZATION"] = df["WORD"].apply(lambda x: [wordnet_lemmatizer.lemmatize(word, pos="v") for word in x])

# END STEP ONE CODE

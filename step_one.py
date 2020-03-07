#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 02:55:49 2020

@author: ddetommaso12
"""

# Load Libraries & Dependencies
from nltk.corpus import stopwords
import numpy as np
import pandas as pd

# First time download stop words
import nltk
nltk.download('stopwords')

# Import data
df = pd.read_csv("./Data/MISSION.csv")

# Convert to all lowercase
df = df.apply(lambda x: x.astype(str).str.lower())

# Remove unnecessary Columns
df = df[['EIN', 'NAME', 'F9_03_PZ_MISSION' ]]

# Load Stop Words
stop_words = stopwords.words('english')

# Filter Stop Words from Mission

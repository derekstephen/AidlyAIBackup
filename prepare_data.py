# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 17:37:37 2020

@author: Derek
"""

# Load Libraries & Dependencies
from nltk.corpus import stopwords
import pandas as pd
import nltk



# First time download stop words
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Load Stop Words
stop_words = stopwords.words('english')

# Import data
df = pd.read_csv("./Data/MISSION.csv")

# Make mission lowercase & Remove Stop Words
df_missions = df["F9_03_PZ_MISSION"].apply(lambda x: [item for item in nltk.sent_tokenize(str(x).lower()) if item not in stop_words])
df["F9_03_PZ_MISSION"] = df_missions

# Print Example Mission Statement
print(df.iloc[0])









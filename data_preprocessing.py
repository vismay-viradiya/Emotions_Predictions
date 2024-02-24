# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 12:13:13 2024

@author: vbvir
"""
from dependencies import libraries as lib

df = lib.pd.read_parquet('go_emotion.parquet')
selected_col = ['text', 'id','admiration',
       'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion',
       'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust',
       'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 'joy',
       'love', 'nervousness', 'optimism', 'pride', 'realization', 'relief',
       'remorse', 'sadness', 'surprise', 'neutral']
df= df[selected_col]



selected_col = ['admiration',
       'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion',
       'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust',
       'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 'joy',
       'love', 'nervousness', 'optimism', 'pride', 'realization', 'relief',
       'remorse', 'sadness', 'surprise', 'neutral']

def preprocess(text):
    lemmatizer = lib.WordNetLemmatizer()
    words = lib.nltk.word_tokenize(text.lower())
    words = [lemmatizer.lemmatize(word) for word in words if word not in lib.stopwords.words('english') and word.isalpha()]
    return ' '.join(words)

# importing base libraries 
import numpy as np
import pandas as pd
from os import path
from PIL import Image
import matplotlib.pyplot as plt
import re

# language processing tools
import nltk; nltk.download('popular')
from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator


# load and read text
f = open('SessionI_500_500.txt', 'r')
tx = f.read()


# tokenize & lemmatize words
tx_tok = word_tokenize(tx)
lemmatizer = WordNetLemmatizer()
tx_lem = [lemmatizer.lemmatize(word) for word in tx_tok]


# filter #1 - remove stop words
stop_words = nltk.corpus.stopwords.words('english')
f = open('dnd_stopwords.txt', 'r')
dnd_words = f.read()
dnd_words = dnd_words.split(',')
stop_words.extend(dnd_words)

tx_filt = []
for word in tx_lem:
    if word.casefold() not in stop_words:
        tx_filt.append(word.casefold())


# filter #2 - remove words that are less than 3 letters or do not start with an alphanumeric
p = re.compile('^\w{3,}$')
tx_filt2 = list(filter(p.match, tx_filt))


# tag words by their parts of speech
tx_tag = nltk.pos_tag(tx_filt2)
tags = [x[1] for x in tx_tag]


# filter #3 - remove unimportant types of words such as conjunctions, auxiliary verbs, etc. 
f = open('stop_types2.txt', 'r')
stop_types = f.read()
tx_pos = []

for idx, tag in enumerate(tags):
    if tag not in stop_types:
        tx_pos.append(tx_tag[idx][0])
        

# generate word cloud image
wordcloud = WordCloud(colormap = 'twilight').generate(".".join(tx_pos))

# display word cloud
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis("off")
plt.show()

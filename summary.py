from transformers import pipeline
import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

os.environ["CUDA_VISIBLE_DEVICES"] = "0"



# load and read text
f = open('TAZEp1_500_500.txt', 'r')
tx = f.read()

# because abstractive summarization has a token limit, use extractive summarization first to truncate text
stopWords = set(stopwords.words("english"))
words = word_tokenize(tx)

#create frequency table to keep word score
freqTable = dict()
for word in words:
    word = word.lower()
    if word in stopWords:
        continue
    if word in freqTable:
        freqTable[word] += 1
    else:
        freqTable[word] = 1

# creating a dictionary to keep the score of each sentence
sentences = sent_tokenize(tx)
sentenceValue = dict()

for sentence in sentences:
    for word, freq in freqTable.items():
        if word in sentence.lower():
            if sentence in sentenceValue:
                sentenceValue[sentence] += freq
            else:
                sentenceValue[sentence] = freq

sumValues = 0
for sentence in sentenceValue:
    sumValues += sentenceValue[sentence]

# average value of a sentence from the original text
average = int(sumValues / len(sentenceValue))

# create extractive summary
summary_ex = ''
for sentence in sentences:
    if (sentence in sentenceValue) and (sentenceValue[sentence] > (1.325 * average)):
        summary_ex += " " + sentence
print(summary_ex)



# abstractive summarization (input needs to be 512 tokens or less)
summarizer = pipeline("summarization")

summary_text = summarizer(summary_ex, max_length=100, min_length=5, do_sample=False)[0]['summary_text']
print(summary_text)
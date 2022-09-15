import os
from tkinter import filedialog
from tkinter import *
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import string
import re
import numpy as np

# set the root path to the current python script location
abspath = os.path.abspath(__file__)
root = os.path.dirname(abspath)
os.chdir(root)


# open a new window to select the folder containing the text docs
# this function is finished
def get_folder():
    root = Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory()
    return folder_path


# build the doc name - text string dictionary
def get_docDict(path):
    doc_dict = {}
    # read the file names from directory path
    file_names = os.listdir(path)

    # for each file insert the name - text string into the dict
    for file in file_names:
        with open(path + "/" + file, 'r') as f:
            # read the whole content into text_string
            text_string = f.read()
            # remove all new lines from the text string
            text_string = re.sub("\n", " ", text_string)
            # insert the entry
            doc_dict[file] = text_string
    return doc_dict


#  clean the text by removing the unnecessary characters and split into tokens
def clean_text(doc_dict):
    clean_dict = {}
    # Refer to the code in python notebook exercise "Vector_Space_Model(VSM)"
    # cleaning process code is directly used
    stopwords_english = stopwords.words("english")
    # all removing are to replace the character with a space " "
    for filename, text_string in doc_dict.items():
        # remove extra white space chars from text string
        text_string = re.sub(r"\s+", " ", text_string)
        # remove extra dots from text string
        text_string = re.sub(r"\.+", " ", text_string)
        # remove hyphen inbetween two words, merge them into one word
        text_string = re.sub(r"-", "", text_string)
        # change all letters in text string to lowercase
        text_string = text_string.lower()
        # tokenize text strings to string tokens
        text_string_tokens = word_tokenize(text_string)
        text_string_clean = []
        # remove all stop words and punctuation from the tokens
        for word in text_string_tokens:
            if (word not in stopwords_english and word not in string.punctuation):
                text_string_clean.append(word)
        clean_dict[filename] = text_string_clean

    return clean_dict


# make the vocabulary of the doc collection
def make_vocab(doc_dict):
    # merge all string tokens in clean dict into one list
    all_string_tokens = []
    for text_string_tokens in doc_dict.values():
        all_string_tokens += text_string_tokens
    # use set method to remove duplicate
    vocab_set = set(all_string_tokens)
    # return as a list
    return list(vocab_set)


# get the Term Frequency table
def get_DocTF(doc_dict, vocab):
    tf_dict = {}
    # a dict with [key:filename] to [value:dict[word, freq]]
    for filename in doc_dict:
        tf_dict[filename] = {}
        # for each word in vocab insert the count of this word in text string
        # for word not appear in this file, mark as 0 as desired.
        for word in vocab:
            tf_dict[filename][word] = doc_dict[filename].count(word)
    return tf_dict


# get the doc frequency table
def get_DocDF(clean_dict, vocab):
    df_dict = {}
    # a dict with [key: word] to [value: frequency in files]
    for word in vocab:
        freq = 0
        for text_string_tokens in clean_dict.values():
            if word in text_string_tokens:
                # increment by one as it appears once more in a new file
                freq += 1
        # insert the entry, note freq can not be 0
        # assert(freq != 0)
        df_dict[word] = freq
    return df_dict

# get the inverse doc frequency table
def inverse_DF(df_dict, vocab, doc_length):
    idf_dict = {}
    # refer to inverse_DF function in VSM exercise code
    # algorithm for get the idf for a word
    # with padding 0.5 to avoid taking log to 0
    # log(N - n + padding) - log(N + padding)
    # and add 1 to the result to normalize since the above formula gives negative value
    padding = 0.5
    for word in vocab:
        idf = round(np.log(((doc_length - df_dict[word] + padding) / (df_dict[word] + padding)) + 1), 4)
        idf_dict[word] = idf
    return idf_dict


# calculate the TF-IDF table
def get_tf_idf(tf_dict, idf_dict, doc_dict, vocab):
    tf_idf_dict = {}
    # the value tf_idf for word is given by tf(word) * idf(word)
    for filename in doc_dict:
        # same structure as tf_dict
        tf_idf_dict[filename] = {}
        for word in vocab:
            tf_idf_dict[filename][word] = round(tf_dict[filename][word] * idf_dict[word], 4)
    return tf_idf_dict


# the VSM ranking function - return the top-5
def vectorSpaceModel(query, doc_dict, tfidf_dict):
    # implement here
    top_5 = {}
    # first we need to adjust the search query as we have done in clean dict
    temp = {"helper": query}
    # use the clean dict method above to achieve it
    query_tokens = clean_text(temp)["helper"]
    # get the weight of each unique word in query
    query_word_count = {}
    for word in query_tokens:
        if word not in query_word_count.keys():
            query_word_count[word] = 1
        else:
            query_word_count[word] += 1
    # get the score of relevance for each file
    # a dict of key: filename -> value: score
    score_of_relevance = {}
    for filename in doc_dict:
        score = 0
        # score is the sum of the score for each word in query w.r.t current file
        # score_for_word = weight_in_query(word) * tf_idf[filename][word]
        for word in query_word_count:
            score += query_word_count[word] * tfidf_dict[filename][word]
        score_of_relevance[filename] = round(score, 4)

    # return top 5 most relevant file
    # sort the score_of_relevance dict by its value
    sorted_key_for_relevance = sorted(score_of_relevance, key=score_of_relevance.get, reverse=True)
    # collect info the top five
    for key in sorted_key_for_relevance[:5]:
        top_5[key] = score_of_relevance[key]
    return top_5


# calculate average document length
def get_avgdl(clean_dict):
    # compute the average document length
    avgdl = 0  # you need to change the value of avgdl here
    for text_string_tokens in clean_dict.values():
        avgdl += len(text_string_tokens)
    return round(avgdl / len(clean_dict), 4)


# calculate the BM25 term table
def bm25(tf_dict, clean_dict, df_dict, vocab, k=1.2, b=0.75):
    bm25_dict = {}
    avgdl = get_avgdl(clean_dict)
    N = len(clean_dict)
    idf_dict = inverse_DF(df_dict, vocab, N)
    # bm25 dict has the same structure as tf-idf dict
    # first initialize the dict values
    for filename in clean_dict:
        bm25_dict[filename] = {}
    for word in vocab:
        for filename in clean_dict:
            # calculate the new tf
            freq = tf_dict[filename][word]
            # formula below refer to BM25_probabilistic_model.ipynb
            tf = (freq * (k + 1)) / (freq + k * (1 - b + b * len(clean_dict[filename]) / avgdl))
            # get idf
            # idf = idf_dict[word]
            # compute the un-rounded value again, so that the result agree with the assignment outline
            idf = np.log(((N - df_dict[word] + 0.5) / (df_dict[word] + 0.5)) + 1)
            # store the score
            bm25_dict[filename][word] = round(tf * idf, 4)
    return bm25_dict


# the BM25 ranking function - return the top-5
def BM25Model(query, doc_dict, bm25_dict):
    top_5 = {}
    # similar to VSM model
    # we weight the score based on the relevance
    # first we need to adjust the search query as we have done in clean dict
    temp = {"helper": query}
    # use the clean dict method above to achieve it
    query_tokens = clean_text(temp)["helper"]
    # get the weight of each unique word in query
    query_word_count = {}
    for word in query_tokens:
        if word not in query_word_count.keys():
            query_word_count[word] = 1
        else:
            query_word_count[word] += 1
    # get the score of relevance for each file
    # a dict of key: filename -> value: score
    score_of_relevance = {}
    for filename in doc_dict:
        score = 0
        # score is the sum of the score for each word in query w.r.t current file
        # score_for_word = weight_in_query(word) * tf_idf[filename][word]
        for word in query_word_count:
            score += query_word_count[word] * bm25_dict[filename][word]
        score_of_relevance[filename] = round(score, 4)

    # return top 5 most relevant file
    # sort the score_of_relevance dict by its value
    sorted_key_for_relevance = sorted(score_of_relevance, key=score_of_relevance.get, reverse=True)
    # collect info the top five
    for key in sorted_key_for_relevance[:5]:
        top_5[key] = score_of_relevance[key]
    return top_5

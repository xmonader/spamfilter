# SpamFilter from practical common lisp chapter 23

import os
import math
from functools import reduce
import operator

MAX_HAM_SCORE = 0.4
MIN_SPAM_SCORE = 0.6

FEATURES = {}
TOTAL_SPAM = 0
TOTAL_HAM = 0


def inverse_chi_square(chi, df):
    """CODE FROM http://www.linuxjournal.com/files/linuxjournal.com/linuxjournal/articles/064/6467/6467s2.html"""
    m = chi / 2.0
    sum = term = math.exp(-m)
    for i in range(1, df//2):
        term *= m / i
        sum += term

    return min(sum, 1.0)

def fisher(probs, number_of_probs):
    return inverse_chi_square(-2*math.log(reduce(operator.mul, probs)), 2*number_of_probs)


def score(features):
    spam_probs, ham_probs, number_of_probs = [], [], 0
    for f in features:
        spamprob = float(f.bayesian_spam_prob())
        spam_probs.append(spamprob)
        ham_probs.append(1 - spamprob)
        number_of_probs += 1
    ham = 1 - fisher(spam_probs, number_of_probs)
    spam = 1 - fisher(ham_probs, number_of_probs)

    return (1 + spam - ham)/ 2.0



def reset():
    global FEATURES, TOTAL_HAM, TOTAL_SPAM
    FEATURES = {}
    TOTAL_SPAM = 0
    TOTAL_HAM = 0

def extract_features(text):
    words = set([x for x in text.split() if len(x) > 2])
    features = []
    for word in words:
        feature = FEATURES.setdefault(word, WordFeature(word))
        features.append(feature)
    return features

class WordFeature:
    def __init__(self, word):
        self.word = word 
        self.ham_count = 0      # number of hams we saw the word in
        self.spam_count = 0     # number of spams we saw the word in 

    def spam_prob(self):
        spam_freq = self.spam_count/(max(1, TOTAL_SPAM))
        ham_freq = self.ham_count/(max(1, TOTAL_HAM))

        res = 0
        try:
            res = spam_freq / (spam_freq + ham_freq)
        except: 
            res = 0 
        return res

    def bayesian_spam_prob(self, assumed_prob=0.5, weight=1):
        basic_prob = self.spam_prob()
        data_points = self.spam_count + self.ham_count
        return ( (weight * assumed_prob) + (data_points * basic_prob) ) / (weight + data_points) 

    def inc(self, type='ham'):
        global TOTAL_SPAM, TOTAL_HAM
        if type == 'ham':
            self.ham_count += 1
            TOTAL_HAM += 1

        elif type == 'spam':
            TOTAL_SPAM += 1
            self.spam_count += 1

def train(text, type='ham'):
    features = extract_features(text)
    for f in features:
        f.inc(type=type)

def classification(score):
    if score <= MAX_HAM_SCORE:
        return 'ham'
    elif score >= MIN_SPAM_SCORE:
        return 'spam'
    else:
        return 'unsure'


def classify(text):
    return classification(score(extract_features(text)))

def import_messages(messages, type='ham'):
    for m in messages:
        train(m, type)

def import_messages_from_dir(messages, type='ham'):
    # walk files in the directory and open it for reading add content as `type`
    for entry in os.scandir(path):
        if entry.is_file():
            path = entry.path
            with open(path, "r") as f:
                train(f.read(), type)


def test():

    train("hello world", 'ham')
    train("bye world", 'ham')
    train("learn new", "ham")
    train("buy now", "spam")
    train("ad buy", "spam")
    print(classify("hello world"))
    print(classify("goodbye world"))
    print(classify("buy item now"))
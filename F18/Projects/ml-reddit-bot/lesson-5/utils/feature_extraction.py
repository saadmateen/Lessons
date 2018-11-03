""" A library of extracting features from text data
"""

# Natural Language Processing Tools
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from nltk.stem.snowball import SnowballStemmer
from nltk import pos_tag_sents
import re

# Machine Learning Tools
from sklearn.feature_extraction.text import TfidfVectorizer


def simplify(corpus):
    """ Takes in a list of strings and removes stopwords, converts to lowercase,
        removes non-alphanumeric characters, and stems each word
    """
    stop_words = set(stopwords.words('english'))
    stemmer = SnowballStemmer('english')
    
    def clean(text):
        text = re.sub('[^a-zA-Z0-9]', '', text)
        words = [stemmer.stem(w) for w in word_tokenize(text.lower()) if w not in stop_words] 
        return " ".join(words)

    return [clean(text) for text in corpus]

def bag_of_words(corpus, save=False):
    """ Takes in a corpus (list of strings) and returns a list of word count vectors (Bag of Words).
        save: (Boolean) chooses whether to save the fitted CountVectorizer
    """
    pass

def tfidf_words(corpus, save=False):
    """ Takes in a corpus (list of strings) and returns tfidf weighted word count vectors.
        save: (Boolean) chooses whether to save the fitted TfidfVectorizer
    """
    pass

def second_person_pronoun_count(text):
    pass

def third_person_pronoun_count(text):
    pass

def positive_word_count(text):
    pass

def negative_word_count(text):
    pass

def hate_word_count(text):
    pass

def caps_word_count(text):
    """ Returns the number of multi-character words in text that are fully capitalized
    """
    pass
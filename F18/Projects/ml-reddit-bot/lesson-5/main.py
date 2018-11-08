import pickle
import pandas

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Natural Language Processing Tools
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from nltk.stem.snowball import SnowballStemmer
from nltk import pos_tag_sents
import re


class HateSpeechDetectionEngine:
    """ Python class that deals with hate speech detection
    """
    def __init__(self):
        self.corpus = None
        self.tags = None
        self.vectorizer = None
        self.model = None
        self.metrics = None

    def _simplify(self, corpus):
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
    
    #Lesson 6
    def _get_lexicon(self, path):
        """ Takes in a path to a text file and returns a set
            containing every word in the file
        """
        words = set()
        with open(path) as file:
            for line in file:
                words.update(line.strip().split(' '))

        return words
    
    # Lesson 6
    def _model_metrics(self, corpus, tags):
        pass
    
    def load_corpus(self, path, corpus_col, tag_col):
        """ Takes in a path to a pickled pandas dataframe, the name of the corpus column,
            and the name of the tag column, and extracts a tagged corpus
        """
        data = pandas.read_pickle(path)[[corpus_col, tag_col]].values
        self.corpus = [row[0] for row in data]
        self.tags = [row[1] for row in data]
        
    def load_model(self, model_name):
        """ Loads a ML model and it's corresponding feature vectorizer
        """
        self.model = pickle.load(open('./models/' + model_name + '_ml_model.pkl', 'rb'))
        self.vectorizer = pickle.load(open('./models/' + model_name + '_vectorizer.pkl', 'rb'))
        self.metrics = pickle.load(open('./models/' + model_name + '_metrics.pkl', 'rb'))
    
    def train_using_bow(self):
        """ Trains a model using Bag of Words on the loaded corpus and tags
        """
        corpus = self._simplify(self.corpus)
        self.vectorizer = CountVectorizer()
        self.vectorizer.fit(corpus)

        bag_of_words = self.vectorizer.transform(corpus)
        x_train, x_test, y_train, y_test = train_test_split(bag_of_words, self.tags, test_size=0.2, stratify=self.tags)

        self.model = MultinomialNB()
        self.model.fit(x_train, y_train)

        self.metrics = self.model.score(x_test, y_test)

    # Lesson 6
    def train_using_tfidf(self):
        pass

    # Lesson 6
    def train_using_custom(self):
        """ Trains model using a more detailed feature extraction approach
        """
        pass

    def evaluate(self):
        """ Returns model performance metrics
        """
        return self.metrics

    def save_model(self, model_name):
        """ Saves the model for future use
        """
        pickle.dump(self.model, open('./models/' + model_name + '_ml_model.pkl', 'wb'))
        pickle.dump(self.vectorizer, open('./models/' + model_name + '_vectorizer.pkl', 'wb'))
        pickle.dump(self.metrics, open('./models/' + model_name + '_metrics.pkl', 'wb'))

    def predict(self, corpus):
        """ Takes in a text corpus and returns predictions
        """
        x = self.vectorizer.transform(self._simplify(corpus))
        return self.model.predict(x)

# Lesson 6
if __name__ == '__main__':
    pass

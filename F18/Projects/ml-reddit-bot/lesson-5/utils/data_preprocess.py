""" Preprocessing data from a pickled pandas dataframe to be ready
    for feature extraction tasks
"""

import pandas


def get_data(path):
    """ Takes in the path to the pickled pandas dataframe
        and returns the data as a numpy matrix
    """
    df = pandas.read_pickle(path)
    return df.values

def get_speech(data):
    """ Takes in the numpy matrix of labeled speech data
        and returns a dictionary with each classification of speech
    """
    return {
        'hate': [x[5] for x in data if x[4] == 0],
        'offensive': [x[5] for x in data if x[4] == 1],
        'regular': [x[5] for x in data if x[4] == 2]
    }
      
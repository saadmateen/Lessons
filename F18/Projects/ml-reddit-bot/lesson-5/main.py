from utils import data_preprocess

data = data_preprocess.get_data("./data/labelled_data.p")
corpus = data_preprocess.get_speech(data)

class HateSpeechDetection:
    def __init__(self):
        pass
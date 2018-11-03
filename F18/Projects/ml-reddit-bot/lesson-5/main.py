from utils import data_preprocess

data = data_preprocess.get_data("./data/raw_data.p")
corpus = data_preprocess.get_speech(data)

print(corpus)
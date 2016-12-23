import gensim, logging

# class WordGeneration():

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

sentences = [['first', 'sentence'], ['second', 'sentence']]

model = gensim.models.Word2Vec(sentences, min_count=1)

model = Word2Vec(sentences, size=100, window=5, min_count=5, workers=4)

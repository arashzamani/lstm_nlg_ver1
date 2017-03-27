# -*- coding: utf-8 -*-
import gensim, logging


class SemanticVector:
    model = ''

    def __init__(self, structure):
        self.structure = structure

    def model_word2vec(self, min_count=15, window=15, size=100):
        print 'preparing sentences list'
        sentences = self.structure.prepare_list_of_words_in_sentences()

        print 'start modeling'
        self.model = gensim.models.Word2Vec(sentences, size=size, window=window, min_count=min_count, workers=4, sample=0.001, sg=0)

        return self.model

    def save_model(self, name):
        self.model.save(name)

    def load_model(self, name):
        self.model = gensim.models.Word2Vec.load(name)

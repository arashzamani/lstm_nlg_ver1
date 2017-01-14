# -*- coding: utf-8 -*-
import gensim, logging


class SemanticVector:
    model = ''

    def __init__(self, structure):
        self.structure = structure

    def model_word2vec(self):
        print 'preparing sentences list'
        sentences = self.structure.prepare_list_of_words_in_sentences()

        print 'start modeling'
        self.model = gensim.models.Word2Vec(sentences, size=100, window=15, min_count=15, workers=4, sample=0.01)

        return self.model

    def save_model(self, name):
        self.model.sample(name)

    def load_model(self, name):
        self.model = gensim.models.Word2Vec.load(name)

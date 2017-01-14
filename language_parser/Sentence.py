from __future__ import unicode_literals
from hazm import *



class Sentence:
    normalizer = ""
    sentence = ""
    words = ""
    sentence_len = 0
    chunks = dict()

    def __init__(self, sentence):
        self.normalizer = Normalizer()
        self.sentence = self.normalizer.normalize(sentence)
        self.words = self.sentence_to_word(sentence)
        self.sentence_len = len(self.words)

    ###################################################
    def __iter__(self):
        yield self.sentence_to_word(self.sentence)
    ###################################################
    # return the normalized sentence
    def __get__sentence(self):
        return self.sentence

    ###################################################
    # this function parse all words of sentence
    def sentence_to_word(self, sentence):
        return word_tokenize(sentence)

    ###################################################
    ###################################################
    # this method tokenize the sentences of a text
    @classmethod
    def parse_sentences(cls, text):
        return sent_tokenize(text)




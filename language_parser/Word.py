from utility.UFile import *
from hazm import *

class Word:

    def __init__(self, word=None):
        if word is not None:
            self.__word = word

    @classmethod
    def get__stop_words(cls):
        stoplistfile = UFile('/home/arash/Downloads/stoplist.txt')
        stopwords = stoplistfile.text
        stop_words = stopwords.split()
        return stop_words

    @classmethod
    def get__words(cls, sentence):
        return word_tokenize(sentence)

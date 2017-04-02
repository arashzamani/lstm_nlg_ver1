from __future__ import unicode_literals
from hazm import *
import utility.TreeParser as treeParser


class Sentence:
    normalizer = ""
    sentence = ""
    words = ""
    sentence_len = 0
    word_with_tags = list()
    tags = list()
    chunks = dict()

    def __init__(self, sentence, tagger, chunker):
        self.normalizer = Normalizer()
        self.sentence = self.normalizer.normalize(sentence)
        self.words = self.sentence_to_word(sentence)
        self.sentence_len = len(self.words)
        self.sentence_to_tags(tagger)
        self.sentence_to_chunks(chunker)

    ###################################################
    def __iter__(self):
        yield self.sentence_to_word(self.sentence)

    ###################################################
    # return the normalized sentence
    def __get__sentence(self):
        return self.sentence

    ###################################################
    def sentence_to_tags(self, tagger):
        self.word_with_tags = tagger.tag(self.words)
        for i in range(0, len(self.word_with_tags), 1):
            self.tags.append(self.word_with_tags[i][1])

    ###################################################
    def sentence_to_chunks(self, chunker):
        self.chunks = treeParser.parse_tree_to_dict(tree2brackets(chunker.parse(self.word_with_tags)))

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

from __future__ import unicode_literals
from hazm import *
import utility.TreeParser as treeParser


class Sentence:
    normalizer = ""
    sentence = ""
    words = ""
    tagger = ""
    tags = ""
    chunks = dict()

    def __init__(self, sentence):
        self.normalizer = Normalizer()
        self.sentence = self.normalizer.normalize(sentence)
        self.words = self.sentence_to_word(sentence)
        self.tagger = POSTagger(model='resources/postagger.model')
        self.chunker = Chunker(model='resources/chunker.model')

    ###################################################
    # return the normalized sentence
    def __get__sentence(self):
        return self.sentence

    ###################################################
    # this function parse all words of sentence
    def sentence_to_word(self, sentence):
        return word_tokenize(sentence)

    ###################################################
    # this function parse usable words of sentence
    # def sentence_to_u_word(self, sentence):

    ###################################################
    # parse sentence based on tags
    def parse_tags(self):
        self.tags = self.tagger.tag(self.words)
        return self.tags

    ###################################################
    # return the dict of tags of this sentence
    def __get__tags(self):
        return self.tags

    ###################################################
    # parse sentence based on chunker, it returns a dict
    def parse_chunks(self):
        if not self.tags:
            self.parse_tags()
        self.chunks = treeParser.parse_tree_to_dict(tree2brackets(self.chunker.parse(self.tags)))
        return self.chunks

    ###################################################
    def __get_chunks(self):
        return self.chunks

    ###################################################
    # parse dependency tree of sentence
    def parse_dependency(self):
        lemmatizer = Lemmatizer()
        parser = DependencyParser(tagger=self.tagger, lemmatizer=lemmatizer)
        return parser.parse(self.words)

    ###################################################
    ###################################################
    # this method tokenize the sentences of a text
    @classmethod
    def parse_sentences(text):
        return sent_tokenize(text)

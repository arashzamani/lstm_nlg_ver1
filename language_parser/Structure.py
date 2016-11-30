from __future__ import unicode_literals
from hazm import *
import language_parser.Sentence as Sentence

class Structure:
    text = ""
    sentences = list()

    def __init__(self, text):
        self.text = text
        self.prepare_sentences()

    def prepare_sentences(self):
        temp_sentences = Sentence.sent_tokenize(self.text)
        for temp in temp_sentences:
            sent = Sentence.Sentence(temp)
            self.sentences.append(sent)
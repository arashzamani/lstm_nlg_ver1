from __future__ import unicode_literals
from hazm import *
import language_parser.Sentence as Sentence

class Structure:
    text = ""
    sentences = list()
    tags = dict()
    chunks = dict()

    def __init__(self, text):
        self.text = text
        self.prepare_sentences()

    def prepare_sentences(self):
        temp_sentences = Sentence.sent_tokenize(self.text)
        for temp in temp_sentences:
            sent = Sentence.Sentence(temp)
            self.sentences.append(sent)

    def generate_tags_dict(self):
        for temp in self.sentences:
            temp_tags = temp.parse_tags()
            for part in temp_tags:
                if part[1] in self.tags.keys():
                    self.tags[part[1]].append(part[0])
                else:
                    values = list()
                    values.append(part[0])
                    self.tags[part[1]] = values

    def generate_chunks_dict(self):
        for temp in self.sentences:
            temp_chunks = temp.parse_chunks()
            for key in temp_chunks.keys():
                if key in self.chunks.keys():
                    self.chunks[key].append(temp_chunks[key])
                else:
                    self.chunks[key] = temp_chunks[key]


from __future__ import unicode_literals
from hazm import *
import language_parser.Sentence as Sentence
import language_parser.word as w
import language_parser.Word as wordC
from gensim import corpora

class Structure:
    text = ""
    sentences = list()
    sentences_obj = list()
    words_in_sentences = list()
    tags = dict()
    chunks = dict()

    def __init__(self, text):
        self.text = text
        self.prepare_sentences()

    # def __iter__(self):
    #     for sentence in self.sentences:
    #         yield sentence.sentence_to_word(sentence)

    def prepare_sentences(self):
        self.sentences = Sentence.Sentence.parse_sentences(self.text)
        self.sentences_obj = [Sentence.Sentence(sentence) for sentence in self.sentences]

    def prepare_list_of_words_in_sentences(self):
        if len(self.words_in_sentences) == 0:
            stop_list = wordC.Word.get__stop_words()
            self.words_in_sentences = [[word for word in word_tokenize(sentence) if word not in w.specials] for sentence in self.sentences]
        return self.words_in_sentences

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
        self.chunks = corpora.Dictionary(sentence.parse_chunks() for sentence in self.sentences_obj)
        # for temp in self.sentences:
        #     temp_chunks = temp.parse_chunks()
        #     for key in temp_chunks.keys():
        #         if key in self.chunks.keys():
        #             self.chunks[key].append(temp_chunks[key])
        #         else:
        #             self.chunks[key] = temp_chunks[key]

from __future__ import unicode_literals
from hazm import *
import language_parser.Sentence as Sentence
import language_parser.word as w
import language_parser.Word as wordC
from gensim import corpora
import logging, sys, pprint
import utility.TreeParser as treeParser


class Structure:
    text = ""
    tagged_text = ""
    sentences = list()
    sentences_obj = list()
    tagged_sentences = list()  # replace tags instead of any word in real sentence
    pure_words_in_sentences = list()
    tags = dict()
    words_in_sentences = list()
    sentence_tags = ""  # for inner usage
    chunks = dict()

    def __init__(self, text):
        logging.basicConfig(stream=sys.stdout, level=logging.INFO)
        logging.log(logging.INFO, "Structure initialized")
        self.text = text
        self.tagger = POSTagger(model='resources/postagger.model')
        self.chunker = Chunker(model='resources/chunker.model')
        self.prepare_sentences()

    # def __iter__(self):
    #     for sentence in self.sentences:
    #         yield sentence.sentence_to_word(sentence)

    def prepare_sentences(self):
        self.sentences = Sentence.Sentence.parse_sentences(self.text)
        print "sentences detected: ", len(self.sentences)
        self.sentences_obj = [Sentence.Sentence(sentence) for sentence in self.sentences]
        print "number of sentences", len(self.sentences_obj)

    def prepare_list_of_words_in_sentences(self):
        if len(self.words_in_sentences) == 0:
            # stop_list = wordC.Word.get__stop_words()
            self.words_in_sentences = [[word for word in word_tokenize(sentence) if word not in w.specials] for sentence in self.sentences]
        return self.words_in_sentences

    def prepare_pure_list_of_words(self):
        temp = list()
        for sentence in self.sentences_obj:
            temp += sentence.words
        self.pure_words_in_sentences = temp
        return self.pure_words_in_sentences

    def generate_tags_dict(self):
        tagged_temp_list = list()
        for temp in self.sentences_obj:
            temp_tags = self.parse_tags(temp)
            tagged_sentence = list()
            for part in temp_tags:
                tagged_sentence.append(part[1])
                if part[1] in self.tags.keys():
                    self.tags[part[1]].add(part[0])
                else:
                    values = set()
                    values.add(part[0])
                    self.tags[part[1]] = values
            self.tagged_sentences.append(tagged_sentence)
            tagged_temp_list += tagged_sentence
        self.tagged_text = " ".join(tagged_temp_list)

    def generate_chunks_dict(self):
        # self.chunks = corpora.Dictionary(self.parse_chunks(sentence) for sentence in self.sentences_obj)
        for temp in self.sentences_obj:
            temp_chunks = self.parse_chunks(temp)
            for key in temp_chunks.keys():
                print key, temp_chunks[key]
                if key in self.chunks.keys():
                    self.chunks[key].append(temp_chunks[key])
                else:
                    self.chunks[key] = temp_chunks[key]


   ###################################################
    # this function parse usable words of sentence
    # def sentence_to_u_word(self, sentence):

    ###################################################
    # parse sentence based on tags
    def parse_tags(self, sentence):
        self.sentence_tags = self.tagger.tag(sentence.words)
        return self.sentence_tags

    ###################################################
    # return the dict of tags of this sentence
    def __get__tags(self):
        return self.tags

    ###################################################
    # parse sentence based on chunker, it returns a dict
    def parse_chunks(self, sentence):
        if not self.sentence_tags:
            self.parse_tags(sentence)
        self.chunks = treeParser.parse_tree_to_dict(tree2brackets(self.chunker.parse(self.sentence_tags)))
        return self.chunks

    ###################################################
    def __get_chunks(self):
        return self.chunks

    ###################################################
    # parse dependency tree of sentence
    def parse_dependency(self, sentence):
        lemmatizer = Lemmatizer()
        parser = DependencyParser(tagger=self.tagger, lemmatizer=lemmatizer)
        return parser.parse(sentence.words)

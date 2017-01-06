# -*- coding: utf-8 -*-
import gensim, logging
from utility.UFile import *

import language_parser.Structure as structure
import language_parser.Word as w

# class WordGeneration():

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

print('Started...')
myfile = UFile('/home/arash/Downloads/bbc30.12.txt')

print 'Reading text file'
struct = structure.Structure(myfile.text)

# sentences = [['first', 'sentence'], ['second', 'sentence']]

# model = gensim.models.Word2Vec(struct.sentences, min_count=1)
# for sentence in struct.sentences:
#     for words in sentence:
#         Sentences.append(words)
print 'preparing sentences list'
Sentences = struct.prepare_list_of_words_in_sentences()

print 'start modeling'
model = gensim.models.Word2Vec(Sentences, size=100, window=15, min_count=15, workers=4, sample=0.01)
# model = gensim.models.Word2Vec(Sentences, min_count=1)
#
# model.save('model_bbc_30_12')
# new_model = gensim.models.Word2Vec.load('/tmp/mymodel')
# u"\u0633\u06CC\u0628" sib
# string1 = u"\u0634\u06CC\u0631"
# string2 = u"\u0634\u06CC\u0631\u06CC\u0646"
# %u062F%u0639%u0648%u062A
string3 = u"\u062F\u0639\u0648\u062A" #davat
string4 = u"\u062C\u0627\u0645\u0639\u0647" #jame'e
string5 = u"\u0633\u06CC\u0627\u0633\u06CC" #siasi
string6 = u"\u0622\u0632\u0627\u062F\u06CC" #azadi
string7 = u"\u0627\u0646\u062A\u0642\u0627\u062F" #enteghad
string8 = u"\u0646\u0645\u0627\u06CC\u0634\u06AF\u0627\u0647" #namayeshgah
mylist = [string8, string4, string5, string6, string3]
print model.doesnt_match(mylist)
print model.similarity(string5, string7)
print model.similarity(string3, string7)
print model.similarity(string4, string8)
print model
# print model[string3]
# print Sentences[9][2]
# new_model[Sentences[9][2]]

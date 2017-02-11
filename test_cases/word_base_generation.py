# -*- coding: utf-8 -*-
import gensim, logging
from utility.UFile import *
import numpy as np
import array
import language_parser.Structure as structure
import language_parser.Word as w
from hazm import *

# class WordGeneration():

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

print('Started...')
myfile = UFile('/home/arash/Downloads/test1.txt')

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
# model = gensim.models.Word2Vec(Sentences, size=100, window=15, min_count=15, workers=4, sample=0.01)
model = gensim.models.Word2Vec.load('/home/arash/PycharmProjects/lstm_nlg_ver1/weights.02.11.hdf5')
# model = gensim.models.Word2Vec(Sentences, min_count=1)
#
# model.save('model_bbc_30_12')
# new_model = gensim.models.Word2Vec.load('/tmp/mymodel')
# u"\u0633\u06CC\u0628" sib
# string1 = u"\u0634\u06CC\u0631"
# string2 = u"\u0634\u06CC\u0631\u06CC\u0646"
# %u062F%u0639%u0648%u062A
string3 = u"\u062F\u0639\u0648\u062A"  # davat
string4 = u"\u062C\u0627\u0645\u0639\u0647"  # jame'e
string5 = u"\u0633\u06CC\u0627\u0633\u06CC"  # siasi
string6 = u"\u0622\u0632\u0627\u062F\u06CC"  # azadi
string7 = u"\u0627\u0646\u062A\u0642\u0627\u062F"  # enteghad
string8 = u"\u0646\u0645\u0627\u06CC\u0634\u06AF\u0627\u0647"  # namayeshgah
string9 = u"."
mylist = [string8, string4, string5, string6, string3]
# print model.doesnt_match(mylist)
# print model.similarity(string5, string7)
# print model.similarity(string3, string7)
# print model.similarity(string4, string8)
# print model
temp_point = [-0.12442809, -0.00681914, -0.00969085, -0.12324874, 0.08447362, -0.10430242,
              0.00822147, - 0.04874244, - 0.00094153, 0.01613271, - 0.18437766, 0.02274217,
              0.00588092, - 0.01656993, - 0.00719543, - 0.01817458, 0.10735682, - 0.08721689,
              0.07933711, - 0.04178236, - 0.02428118, - 0.12842347, - 0.04107852, 0.05975144,
              -0.02823395, - 0.12368862, - 0.11754932, - 0.07995055, - 0.06337339, 0.06340624,
              -0.05770004, 0.10570755, 0.03751742, - 0.07473251, - 0.09659681, - 0.00471876,
              -0.03323877, - 0.04957245, - 0.00725111, - 0.11395489, - 0.15221773, - 0.12679672,
              0.0044793, 0.03538996, 0.02058227, 0.04278617, - 0.04078601, - 0.0653117,
              - 0.01436101, - 0.13610736, - 0.12293012, - 0.08703791, - 0.02420867, - 0.12470925,
              - 0.09559713, - 0.10995527, - 0.00610521, - 0.04177226, - 0.00423405, - 0.06071534,
              - 0.14011784, - 0.10075258, - 0.1189483, - 0.09674383, - 0.00948784, - 0.13100001,
              - 0.02408187, - 0.0203993, - 0.05849553, - 0.0707489, - 0.18210988, 0.02524797,
              - 0.26622117, 0.03081921, - 0.10845306, - 0.01585765, - 0.01270937, - 0.01039595,
              - 0.10633096, 0.01354673, - 0.0683145, - 0.07541882, 0.06313634, 0.00712584,
              - 0.09705989, - 0.21312058, - 0.01063135, 0.05475186, - 0.10100268, 0.0874306,
              - 0.0532059, 0.09259485, 0.03804192, - 0.16444789, 0.07784274, - 0.11639207,
              - 0.05126157, - 0.09833513, 0.02350237, - 0.09992518]

model_word_vector = np.array(temp_point, dtype='f')

# y = model.most_similar(positive=[temp_point], topn=10)

topn = 20
most_similar_words = model.most_similar([model_word_vector], [], topn)
tagger = POSTagger(model='/home/arash/PycharmProjects/lstm_nlg_ver1/resources/postagger.model')

for item in most_similar_words:
    temp = tagger.tag(item[0])
    print item[0], temp[1], item[1]

print tagger.tag(string6)


# print Sentences[9][2]
# new_model[Sentences[9][2]]

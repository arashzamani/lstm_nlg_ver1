from __future__ import unicode_literals
import codecs
from hazm import *

# def _init(text):

f = codecs.open("/home/arash/Downloads/test.txt", 'r', encoding='utf8')
text = f.read()
sents = sent_tokenize(text)
print(len(sents))

normalizer = Normalizer()


tagger = POSTagger(model='../resources/postagger.model')

# for k in range(0, len(v), 1):
#     print(v[k][0], v[k][1])
tagged_sentences = list()
for sentence in sents:
    # print(sentence)
    normalized_sentence = normalizer.normalize(sentence)
    v = tagger.tag(word_tokenize(normalized_sentence))
    tagged_sentences.append(v)

tags = dict()
for sentence in tagged_sentences:
    for part in sentence:
        if part[1] in tags.keys():
            tags[part[1]].append(part[0])
        else:
            values = list()
            values.append(part[0])
            tags[part[1]] = values
for k in tags.keys():
    print (k, len(tags[k]))

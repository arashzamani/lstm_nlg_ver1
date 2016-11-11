from __future__ import print_function
from __future__ import unicode_literals
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
from hazm import *
import numpy as np
import locale
import random
import sys
import codecs
import math
import nltk.data
import language_parser.word as w

f = codecs.open("/home/arash/Downloads/bbc.txt", 'r', encoding='utf8')
text = f.read()
#text = open(path).read()
print('corpus length:', len(text))


sents = sent_tokenize(text)
words = w.pure_word_tokenize(text)
# words = word_tokenize(text)
print ('len of words are:', len(words))
unique_words = dict()

for i in range(0, len(words), 1):
    if words[i] in unique_words.keys():
        unique_words[words[i]] += 1
    else:
        unique_words[words[i]] = 1

print ("unique words: ", len(unique_words))
thefile = codecs.open("/home/arash/Downloads/unique_words.txt", 'w', encoding='utf8')

for keys in unique_words.keys():
  thefile.write("%s\t%s\n" % keys % unique_words[keys])

thefile.close()

#in maximum len bayady miangin tedad kalamat jomleh bashe
maxlen = 67
step = 3
sentences = []
next_words = []

indices_word = dict((i, c) for i, c in enumerate(unique_words.keys()))

for i in range(0, len(words) - maxlen, step):
    sentences.append(words[i: i + maxlen])
    next_words.append(words[i + maxlen])
print('nb sequences:', len(next_words))


print('Vectorization...')
X = np.zeros((len(sentences), maxlen, len(unique_words)), dtype=np.bool)
y = np.zeros((len(sentences), len(unique_words)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, word in enumerate(sentence):
        X[i, t, unique_words[word]] = 1
    y[i, unique_words[next_words[i]]] = 1


# build the model: a single LSTM
print('Build model...')
model = Sequential()
model.add(LSTM(128, input_shape=(maxlen, len(unique_words))))
model.add(Dense(len(unique_words)))
model.add(Activation('softmax'))

optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)



def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# train the model, output generated text after each iteration
for iteration in range(1, 20):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(X, y, batch_size=128, nb_epoch=1)

    start_index = random.randint(0, len(words) - maxlen - 1)

    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print()
        print('----- diversity:', diversity)

        generated = ''
        space = u' '
        sentence = words[start_index: start_index + maxlen - 1]
        newSentence = [sentence[0]]
        for temp in sentence:
            newSentence.append(space)
            newSentence.append(temp)
        for temp in newSentence:
            generated += temp
        print('----- Generating with seed: "', newSentence, '"')
        sys.stdout.write(generated)

        for i in range(400):
            x = np.zeros((1, maxlen, len(unique_words)))
            for t, word in enumerate(sentence):
                x[0, t, unique_words[word]] = 1.

            preds = model.predict(x, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_word = indices_word[next_index]

            generated += space + next_word
            newSentence = newSentence[1:] + [space, next_word]

            sys.stdout.write(next_word)
            sys.stdout.write(space)
            sys.stdout.flush()
        print()
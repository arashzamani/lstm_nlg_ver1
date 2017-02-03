# -- coding: utf-8 --
# Load Larger LSTM network and generate text
import codecs
import sys
from collections import Counter

import numpy
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers.recurrent import GRU
from keras.models import Sequential
from keras.utils import np_utils

# load ascii text and covert to lowercase
f = codecs.open("/home/arash/Downloads/test1.txt", 'r', encoding='utf8')
raw_text = f.read()
C = Counter(raw_text)
C = sorted([[c, C[c]] for c in C], key=lambda x: x[1], reverse=True)
st = set([c[0] for c in C if c[1] > 1])
print st
raw_text = [d for d in raw_text if d in st][0:200000]

# create mapping of unique chars to integers, and a reverse mapping
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))
# summarize the loaded data
n_chars = len(raw_text)
n_vocab = len(chars)
print "Total Characters: ", n_chars
print "Total Vocab: ", n_vocab

# prepare the dataset of input to output pairs encoded as integers
seq_length = 25
dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 1):
    seq_in = raw_text[i:i + seq_length]
    seq_out = raw_text[i + seq_length]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)
print "Total Patterns: ", n_patterns
# reshape X to be [samples, time steps, features]
X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
# normalize
X = X / float(n_vocab)
# one hot encode the output variable
y = np_utils.to_categorical(dataY)
# define the LSTM model
model = Sequential()
nn = 16

# model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
# model.add(Dropout(0.2))
# model.add(LSTM(256))
# model.add(Dropout(0.2))
# model.add(Convolution1D(nb_filter=nn*3, filter_length=4, border_mode='same', activation='relu'))
# model.add(MaxPooling1D(pool_length=4))
# model.add(Dropout(0.15))
print 'x shape 1', X.shape[1]
print 'x shape 2', X.shape[2]

model.add(GRU(nn * 4, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.02))

model.add(GRU(nn * 3, return_sequences=True))
model.add(Dropout(0.02))

model.add(GRU(nn * 2, return_sequences=True))
model.add(Dropout(0.02))

model.add(GRU(nn * 1, return_sequences=False))
model.add(Dropout(0.02))

model.add(Dense(y.shape[1], activation='softmax'))
model.add(Dropout(0.02))

print 'for arash y ', y.shape[1]
# load the network weights
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

print type(X)
print X.shape
print '--------------------'
print type(y)
print y.shape
#
# def sample(preds, temperature=1.0):
#     # helper function to sample an index from a probability array
#     # preds = np.asarray(preds).astype('float64')
#
#     id_probs = sorted(enumerate(preds), key=lambda x: x[1], reverse=True)[0:5]
#     ids = [v[0] for v in id_probs]
#     probs = numpy.array([v[1] for v in id_probs]) / sum([v[1] for v in id_probs])
#
#     return numpy.random.choice(ids, p=probs)
#
#     '''
#     return numpy.argmax(preds)
#     preds = numpy.log(preds) / temperature
#     exp_preds = numpy.exp(preds)
#     preds = exp_preds / numpy.sum(exp_preds)
#     probas = numpy.random.multinomial(1, preds, 1)
#     return numpy.random.randint(0, len(preds) - 1)
#     return numpy.argmax(probas)
#     '''
#
#
# for rn in range(400):
#     print rn
#     model.fit(X, y, nb_epoch=5, batch_size=64)  # , callbacks=callbacks_list)
#     # pick a random seed
#     start = numpy.random.randint(0, len(dataX) - 1)
#     pattern = dataX[start]
#     print "Seed:"
#     print "\"", ''.join([int_to_char[value] for value in pattern]), "\""
#     rs = []
#     for i in range(300):
#         x = numpy.reshape(pattern, (1, len(pattern), 1))
#         x = x / float(n_vocab)
#         prediction = model.predict(x, verbose=0)
#
#         # index = numpy.argmax(prediction[0])
#
#         index = sample(prediction[0], 2.0)
#         result = int_to_char[index]
#         seq_in = [int_to_char[value] for value in pattern]
#         sys.stdout.write(result)
#         rs.append(index)
#         pattern.append(index)
#         pattern = pattern[1:len(pattern)]
#     print "\nDone."
    # print rs



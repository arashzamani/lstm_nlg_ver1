import collections
from keras.models import load_model
import operator
import numpy
import sys
from keras.layers import Dense, Embedding
from keras.layers import Dropout
from keras.layers.recurrent import GRU
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers.convolutional import Convolution1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import LSTM

import language_parser.SemanticVector as sv
import language_parser.Structure as structure
from utility.UFile import *
import language_parser.Word as w


class StructureModel:
    def __init__(self, file):
        self.file = file

    def model(self):
        struct = structure.Structure(self.file.text)
        seq_length = 10
        word_list = struct.prepare_pure_list_of_words()
        # compute the vocabulary size
        vocabulary = sorted(list(set(word_list)))
        vocabulary.insert(0, 'UK')
        vocab_lenght = len(vocabulary)
        struct.generate_tags_dict()
        # semantic modeling
        semantic = StructureModel.semantic_model(struct, seq_length)

        StructureModel.word_model(struct, seq_length, semantic.model, word_list, vocabulary, vocab_lenght, self.file)

    @classmethod
    def word_model(cls, structure, seq_length, word2vec, word_list, vocabulary, vocab_length, file=None):
        total = 0
        for t in structure.sentences_obj:
            total += t.sentence_len
        print file
        avg = total / len(structure.sentences_obj)
        print "average length of sentence", avg

        word_to_int = dict((c, i) for i, c in enumerate(vocabulary))
        int_to_word = dict((i, c) for i, c in enumerate(vocabulary))

        dataX = []
        dataY = numpy.zeros((len(word_list) - seq_length, vocab_length))
        for i in range(0, len(word_list) - seq_length):
            words = word_list
            seq_in = words[i:i + seq_length]
            seq_out = words[i + seq_length]
            dataX.append([word_to_int[word] for word in seq_in])
            dataY[i, [word_to_int[seq_out]]] = 1

        n_patterns = len(dataX)
        for x in dataX:
            if len(x) != seq_length:
                print 'uneqal'
        X = numpy.reshape(dataX, (n_patterns, seq_length))
        y = dataY

        print X.shape, y.shape

        word_model = Sequential()
        nn = 16 * 2
        print vocab_length, seq_length

        embedding_layer = Embedding(50000, 100, input_length=seq_length)

        word_model.add(embedding_layer)

        word_model.add(Convolution1D(nb_filter=32 * 9, filter_length=3, border_mode='same', activation='relu'))
        word_model.add(MaxPooling1D(pool_length=3))

        word_model.add(LSTM(32 * 3, return_sequences=True))
        word_model.add(Dropout(0.05))

        word_model.add(LSTM(32 * 3 * 3, return_sequences=True))
        word_model.add(Dropout(0.01))

        word_model.add(Convolution1D(nb_filter=32 * 2, filter_length=3, border_mode='same', activation='relu'))
        word_model.add(MaxPooling1D(pool_length=3))

        word_model.add(LSTM(32 * 3, return_sequences=True))
        word_model.add(Dropout(0.05))

        word_model.add(LSTM(nn * 5, return_sequences=False))
        word_model.add(Dropout(0.05))

        word_model.add(Dense(y.shape[1], activation='softmax'))

        print word_model.summary()

        # word_model.load_weights('model_for_hafez.h5')

        word_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

        test_dataX = StructureModel.load_test_data(seq_length, word_to_int)

        # testing
        for rn in range(10000):
            print rn
            word_model.fit(X, y, nb_epoch=1, batch_size=512)  # , callbacks=callbacks_list)
            word_model.save('model_for_hafez.h5')
            # continue
            # pick a random seed
            start = numpy.random.randint(0, len(test_dataX) - 1)
            pattern = test_dataX[start]  # dataX
            print "Seed:"
            print start
            print(pattern)
            for p in pattern:
                if p in int_to_word:
                    sys.stdout.write(int_to_word[p] + ' ')
                else:
                    sys.stdout.write('UN ')
            sys.stdout.write('---')
            rs = []
            for i in range(30):
                x = numpy.reshape(pattern, (1, seq_length))
                preds = word_model.predict(x, verbose=0)[0]
                argm = numpy.argmax(preds)
                sys.stdout.write(int_to_word[argm] + ' ')
                pattern.append(argm)
                pattern = pattern[1:len(pattern)]
            print "\nDone."

    @classmethod
    def find_nearest_words(cls, word2vec, prediction_vec):
        model_word_vector = numpy.array(prediction_vec[0], dtype='f')
        topn = 20
        most_similar_words = word2vec.wv.most_similar([model_word_vector], [], topn)

        return most_similar_words[0]

    @classmethod
    def semantic_model(cls, structure, seq_length):
        semantic_model = sv.SemanticVector(structure)
        semantic_model.model_word2vec(15, seq_length)
        semantic_model.save_model('weights.02.11.hdf5')
        return semantic_model

    @classmethod
    def load_test_data(cls, seq_length, word_to_int):
        test_data_file = UFile('test_hafez.txt')
        test_data_structure = structure.Structure(test_data_file.text)
        test_data_word_list = test_data_structure.prepare_pure_list_of_words()

        dataX = []
        for i in range(0, len(test_data_word_list) - seq_length):
            words = test_data_word_list
            seq_in = words[i:i + seq_length]
            tempX = []
            for word in seq_in:
                if word in word_to_int:
                    tempX.append(word_to_int[word])
                else:
                    tempX.append(0)
            dataX.append(tempX)
        return dataX

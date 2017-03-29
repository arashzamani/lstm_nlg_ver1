import collections

import operator
import numpy
import random
import sys
from keras.layers import Dense, Activation, Dropout, Embedding, LSTM, Input
from keras.layers.recurrent import GRU
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasRegressor
from keras.utils import np_utils
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import language_parser.SemanticVector as sv
import language_parser.Structure as structure
import language_parser.Word as w


class embeddingWord:
    def __init__(self, file):
        self.file = file

    def model(self):
        struct = structure.Structure(self.file.text)
        seq_length = 5
        word_list = struct.prepare_pure_list_of_words()
        # compute the vocabulary size
        vocabulary = sorted(list(set(word_list)))
        vocab_lenght = len(vocabulary)
        struct.generate_tags_dict()
        # semantic modeling
        semantic = embeddingWord.semantic_model(struct, seq_length)

        embeddingWord.word_model(struct, seq_length, semantic.model, word_list, vocabulary, vocab_lenght)

    @classmethod
    def word_model(cls, structure, seq_length, word2vec, word_list, vocabulary, vocab_length):
        total = 0
        for t in structure.sentences_obj:
            total += t.sentence_len

        avg = total / len(structure.sentences_obj)
        print "average length of sentence", avg
        print "number of unique words: ", vocab_length
        word_to_int = dict((c, i) for i, c in enumerate(vocabulary))
        int_to_word = dict((i, c) for i, c in enumerate(vocabulary))

        non_word2vec_list = [-1.0] * 100
        dataX = []
        dataY = list()
        n_words_in_text = len(word_list)

        for sentence in structure.sentences_obj:
            for i in range(0, len(sentence.words) - seq_length, 1):
                seq_in = word_list[i:i + seq_length]
                seq_out = word_list[i + seq_length]
                dataX.append([word_to_int[word] for word in seq_in])
                # dataY.append(word_to_int[seq_out])
                if seq_out not in word2vec.wv.vocab:
                    dataY.append(non_word2vec_list)
                else:
                    dataY.append(word2vec[seq_out])
        nb_patterns = len(dataX)

        X = numpy.reshape(dataX, (nb_patterns, seq_length))
        print X.shape

        Y = numpy.reshape(dataY, (nb_patterns, 100))
        print Y.shape

        print('Build model...')
        model = Sequential()

        embedding_layer = Embedding(vocab_length, input_length=seq_length, output_dim=10)

        model.add(embedding_layer)
        # define the LSTM model
        nn = 16
        model.add(LSTM(nn * 4, return_sequences=True))
        model.add(Dropout(0.02))

        model.add(LSTM(nn * 3, return_sequences=True))
        model.add(Dropout(0.02))

        model.add(LSTM(nn * 2, return_sequences=True))
        model.add(Dropout(0.02))

        model.add(LSTM(nn * 1, return_sequences=False))
        model.add(Dropout(0.02))

        model.add(LSTM(Y.shape[1], activation='tanh'))
        model.add(Dropout(0.02))

        # load the network weights
        model.compile(loss='mean_squared_error', optimizer='adam',
                      metrics=['accuracy'])

        print('Train...')
        model.fit(X, Y, batch_size=16, nb_epoch=15)

    @classmethod
    def semantic_model(cls, structure, seq_length):
        semantic_model = sv.SemanticVector(structure)
        semantic_model.model_word2vec(15, seq_length)
        semantic_model.save_model('weights.02.11.hdf5')
        return semantic_model

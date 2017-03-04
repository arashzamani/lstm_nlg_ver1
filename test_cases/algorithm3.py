import collections

import operator
import numpy
import sys
from keras.layers import Dense, Activation, Dropout
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


class StructureModel:
    def __init__(self, file):
        self.file = file

    def model(self):
        struct = structure.Structure(self.file.text)
        seq_length = 15
        word_list = struct.prepare_pure_list_of_words()
        # compute the vocabulary size
        vocabulary = sorted(list(set(word_list)))
        vocab_lenght = len(vocabulary)
        struct.generate_tags_dict()
        y_len = 5
        # semantic modeling
        semantic = StructureModel.semantic_model(struct, seq_length, y_len)

        StructureModel.word_model(struct, seq_length, semantic.model, word_list, vocabulary, vocab_lenght, y_len)

    @classmethod
    def word_model(cls, structure, seq_length, word2vec, word_list, vocabulary, vocab_length, y_len):
        total = 0

        for t in structure.sentences_obj:
            total += t.sentence_len

        avg = total / len(structure.sentences_obj)
        print "average length of sentence", avg

        word_to_int = dict((c, i) for i, c in enumerate(vocabulary))
        int_to_word = dict((i, c) for i, c in enumerate(vocabulary))

        non_word2vec_list = [-1.0] * y_len
        dataX = []
        dataY = list()
        n_words_in_text = len(word_list)

        for i in range(0, n_words_in_text - seq_length, 1):
            seq_in = word_list[i:i + seq_length]
            seq_out = word_list[i + seq_length]
            for word in seq_in:
                if word not in word2vec.wv.vocab:
                    dataX.append(non_word2vec_list)
                else:
                    dataX.append(word2vec[word])
            if seq_out not in word2vec.wv.vocab:
                dataY.append(non_word2vec_list)
            else:
                dataY.append(word2vec[seq_out])
        n_patterns = len(dataX)

        # reshape X to be [samples, time steps, features]
        X = numpy.reshape(dataX, (n_patterns, seq_length, y_len))
        # normalize
        # X = X / float(vocab_length)

        print X.shape
        y = numpy.reshape(dataY, (n_patterns, y_len))

        print y.shape

        # define the LSTM model

        estimator = KerasRegressor(build_fn=StructureModel.define_model(X, y_len), nb_epoch=10, batch_size=128,
                                   verbose=0)

        # word_model.fit(X, y, nb_epoch=5, batch_size=512)
        # testing
        for rn in range(2):
            print rn
            estimator.fit(X, y)  # , callbacks=callbacks_list)
            # pick a random seed
            start = numpy.random.randint(0, len(dataX) - 1)
            pattern = dataX[start]
            print "Seed:"
            print "\"", ' '.join([int_to_word[value] for value in pattern]), "\""
            rs = []
            for i in range(15):
                x = numpy.reshape(pattern, (1, len(pattern), 1))
                x = x / float(vocab_length)
                prediction = estimator.predict(x, batch_size=512)
                prd_word = StructureModel.find_nearest_words(word2vec, prediction)
                # seq_in = [int_to_word[value] for value in pattern]
                print prediction[0]
                sys.stdout.write(prd_word[0])
                # rs.append(index)
                pattern.append(word_to_int[prd_word[0]])
                pattern = pattern[1:len(pattern)]
            print "\nDone."

    @classmethod
    def define_model(cls, X, y_len):
        word_model = Sequential()
        nn = 16

        word_model.add(Dense(nn * 4, init='normal', activation='relu', return_sequences=True,
                             input_shape=(X.shape[1], X.shape[2])))
        word_model.add(Dense(nn * 3, init='normal'))
        word_model.add(Dense(nn * 2, init='normal'))
        word_model.add(Dense(nn * 1, init='normal'))
        word_model.add(Dense(y_len, init='normal'))

        # load the network weights
        word_model.compile(loss='mean_squared_error', optimizer='adam')
        return word_model

    @classmethod
    def find_nearest_words(cls, word2vec, prediction_vec):
        model_word_vector = numpy.array(prediction_vec[0], dtype='f')
        topn = 20
        most_similar_words = word2vec.wv.most_similar([model_word_vector], [], topn)

        return most_similar_words[0]

    @classmethod
    def semantic_model(cls, structure, seq_length, y_len):
        semantic_model = sv.SemanticVector(structure)
        semantic_model.model_word2vec(1, seq_length, y_len)
        semantic_model.save_model('weights.02.11.hdf5')
        return semantic_model

import collections

import operator
import numpy
import sys
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers.recurrent import GRU
from keras.models import Sequential
from keras.utils import np_utils

import language_parser.SemanticVector as sv
import language_parser.Structure as structure
import language_parser.Word as w


class StructureModel:
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
        semantic = StructureModel.semantic_model(struct, seq_length)
        StructureModel.word_model(struct, seq_length, semantic.model, word_list, vocabulary, vocab_lenght)

    @classmethod
    def word_model(cls, structure, seq_length, word2vec, word_list, vocabulary, vocab_length):
        total = 0
        for t in structure.sentences_obj:
            total += t.sentence_len
        avg = total / len(structure.sentences_obj)
        print "average length of sentence", avg

        word_to_int = dict((c, i) for i, c in enumerate(vocabulary))
        int_to_word = dict((i, c) for i, c in enumerate(vocabulary))

        non_word2vec_list = [0.0] * 100
        dataX = []
        dataY = list()
        n_words_in_text = len(word_list)
	for t in structure.sentences_obj:
            for i in range(0, t.sentence_len - seq_length, 1):
                seq_in = word_list[i:i + seq_length]
                seq_out = word_list[i + seq_length]
                #dataX.append([word_to_int[word] for word in seq_in])
                dataX.append([word2vec[word] if word in word2vec else non_word2vec_list for word in seq_in])
	        if seq_out not in word2vec.wv.vocab:
                    dataY.append(non_word2vec_list)
                else:
                    dataY.append(word2vec[seq_out])
        n_patterns = len(dataX)

        # reshape X to be [samples, time steps, features]
        X = numpy.reshape(dataX, (n_patterns, seq_length, 100))
        # normalize
        #X = X / float(vocab_length)

        print X.shape

        y = numpy.reshape(dataY, (n_patterns, 100))

        print y.shape

        # define the LSTM model
        word_model = Sequential()
        nn = 16*2

        word_model.add(GRU(nn * 4, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
        word_model.add(Dropout(0.05))

        word_model.add(GRU(nn * 3, return_sequences=True))
        word_model.add(Dropout(0.05))

        word_model.add(GRU(nn * 2, return_sequences=True))
        word_model.add(Dropout(0.05))

        word_model.add(GRU(nn * 1, return_sequences=False))
        word_model.add(Dropout(0.05))

        word_model.add(Dense(y.shape[1], activation='tanh'))
        word_model.add(Dropout(0.05))

        # load the network weights
        word_model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['accuracy'])

        # testing
        for rn in range(100):
            print rn
            word_model.fit(X, y, nb_epoch=5, batch_size=512)  # , callbacks=callbacks_list)
            # pick a random seed
            start = numpy.random.randint(0, len(dataX) - 1)
            pattern = dataX[start] #dataX
	    print "Seed:"
            #for value in pattern:
	    #	aa = StructureModel.find_nearest_words(word2vec, value)
	    #	sys.stdout.write(aa[0]+' ')
	    rs = []
            for i in range(10):
		x = numpy.reshape(pattern, (1, seq_length, 100))
                prediction = word_model.predict(x, verbose=0)
		prd_word = StructureModel.find_nearest_words(word2vec, prediction)
		#print prediction[0]
                sys.stdout.write(prd_word[0]+' ')
		#print prd_word[0]
		pattern.append(prediction[0])
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

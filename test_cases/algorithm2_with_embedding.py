import collections

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
        #for sentence in structure.sentences_obj:
        if 1==1:
	     for i in range(0, len(word_list) - seq_length):
	    #for i in range(0, len(sentence.words) - seq_length, 1):
                seq_in = word_list[i:i + seq_length] #sentence.words[i:i + seq_length]
                seq_out = word_list[i + seq_length] #sentence.words[i + seq_length]
	        dataX.append([word_to_int[word] for word in seq_in])
	        if seq_out not in word2vec.wv.vocab:
                    dataY.append(non_word2vec_list)
                else:
                    dataY.append(word2vec[seq_out])
        n_patterns = len(dataX)
	for x in dataX:
	    if len(x) != seq_length:
		print 'uneqal'
        X = numpy.reshape(dataX, (n_patterns, seq_length))

        print X.shape

        y = numpy.reshape(dataY, (n_patterns, 100))

        print X.shape, y.shape
        
	#X = X[0:1000, :]
	#y = y[0:1000, :]
	#y = numpy.reshape(y, (1000, 1))
	print X.shape
	print X
	print y
        word_model = Sequential()
        nn = 16*2
	print vocab_length, seq_length	
        
	embedding_layer = Embedding(50000, 100, input_length=seq_length)
	
	word_model.add(embedding_layer)
	
	word_model.add(Convolution1D(nb_filter=32*9, filter_length=3, border_mode='same', activation='relu'))
	word_model.add(MaxPooling1D(pool_length=3))

        word_model.add(LSTM(32*3*3, return_sequences=True))
        word_model.add(Dropout(0.05))

	word_model.add(Convolution1D(nb_filter=32*3, filter_length=3, border_mode='same', activation='relu'))
        word_model.add(MaxPooling1D(pool_length=3))	

        word_model.add(LSTM(32*3, return_sequences=True))
        word_model.add(Dropout(0.05))

	#word_model.add(Convolution1D(nb_filter=32*2, filter_length=3, border_mode='same', activation='relu'))
        #word_model.add(MaxPooling1D(pool_length=3))

	word_model.add(LSTM(32*3, return_sequences=True))
        word_model.add(Dropout(0.05))

        word_model.add(LSTM(32*3, return_sequences=True))
        word_model.add(Dropout(0.05))

        word_model.add(LSTM(32*3, return_sequences=True))
        word_model.add(Dropout(0.05))

        word_model.add(LSTM(nn * 4, return_sequences=False))
        word_model.add(Dropout(0.05))

        word_model.add(Dense(y.shape[1], activation='tanh'))
        
	print word_model.summary()
        
	# load the network weights
        word_model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

        # testing
        for rn in range(100):
            print rn
            word_model.fit(X, y, nb_epoch=5, batch_size=512)  # , callbacks=callbacks_list)
            # pick a random seed
	    start = numpy.random.randint(0, len(dataX) - 1)
            pattern = dataX[start] #dataX
	    print "Seed:"
	    print start
	    print(pattern)
	    rs = []
            for i in range(30):
		x = numpy.reshape(pattern, (1, seq_length))
                prediction = word_model.predict(x, verbose=0)
		prd_word = StructureModel.find_nearest_words(word2vec, prediction)
                sys.stdout.write(prd_word[0]+' ')
		pattern.append(word_to_int[prd_word[0]])
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

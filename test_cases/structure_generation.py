import collections

import numpy
import sys
from keras.layers import Embedding, Input, Dropout, Dense, Merge, Activation
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import GRU, LSTM
from keras.layers.core import RepeatVector
from keras.models import Sequential
from keras.utils import np_utils
from keras.preprocessing import sequence

import language_parser.SemanticVector as sv
import language_parser.Structure as structure


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
        # semantic modeling
        semantic = StructureModel.semantic_model(struct)

        # tags modeling
        tag_dict, tag_model = StructureModel.tags_model(struct, seq_length)


        #data prepration
        #tags data
        tag_list = struct.tagged_text.split()
        tag_set = sorted(list(set(tag_list)))
        tags_array, tags_to_int, int_to_tags, tagsX, tagsY = StructureModel.data_preparation(tag_set, seq_length, tag_list)
        #words data

        words_array, words_to_int, int_to_words, wordsX, wordsY = StructureModel.data_preparation(vocabulary, seq_length, word_list)

        nb_patterns = len(wordsX)
        print 'nb_patt', nb_patterns
        # word modeling
        word_model = StructureModel.word_model(struct, seq_length, vocab_lenght, nb_patterns)

        model = StructureModel.combine_model(struct, tag_model, word_model, seq_length, vocab_lenght)
        # word_model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
        # model.fit([tagsX, wordsX], wordsY, batch_size=300, nb_epoch=2)

        # model.fit([tagsX, wordsX], wordsY, batch_size=300, nb_epoch=2)



    @classmethod
    def word_model(cls, structure, seq_length, vacab_length, nb_pattern):
        nn = 128
        print 'vocab_length = ', vacab_length
        print 'seq_length = ', seq_length

        word_model = Sequential()

        word_model.add(GRU(output_dim=nn, return_sequences=True, input_shape=(seq_length, 1)))
        word_model.add(Dropout(0.02))
        #
        word_model.add(GRU(output_dim=nn / 4, return_sequences=True))
        word_model.add(Dropout(0.02))
        #
        word_model.add(GRU(output_dim=nn / 8, return_sequences=True))
        word_model.add(Dropout(0.02))
        #
        # word_model.add(Dense(vacab_length, activation='relu'))
        # word_model.add(Dropout(0.02))

        return word_model

    @classmethod
    def tags_model(cls, structure, seq_length):
        total = 0
        for t in structure.sentences_obj:
            total += t.sentence_len

        avg = total / len(structure.sentences_obj)
        print "average length of sentence", avg

        tags_dict = collections.OrderedDict(sorted(structure.tags.items()))
        tags_len = len(tags_dict)

        tag_to_int = dict((c, i) for i, c in enumerate(tags_dict))
        int_to_tag = dict((i, c) for i, c in enumerate(tags_dict))

        dataX = []
        dataY = []
        tagged_text = structure.tagged_text.split()
        n_tags_in_text = len(tagged_text)

        for i in range(0, n_tags_in_text - seq_length, 1):
            seq_in = tagged_text[i:i + seq_length]
            seq_out = tagged_text[i + seq_length]
            dataX.append([tag_to_int[char] for char in seq_in])
            dataY.append(tag_to_int[seq_out])
        n_patterns = len(dataX)

        # reshape X to be [samples, time steps, features]
        X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
        # normalize
        X = X / float(tags_len)
        # one hot encode the output variable
        y = np_utils.to_categorical(dataY)
        # define the LSTM model
        tag_model = Sequential()
        nn = 16

        tag_model.add(GRU(nn * 4, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
        tag_model.add(Dropout(0.02))

        tag_model.add(GRU(nn * 3, return_sequences=True))
        tag_model.add(Dropout(0.02))

        tag_model.add(GRU(nn * 2, return_sequences=True))
        tag_model.add(Dropout(0.02))

        tag_model.add(GRU(nn * 1, return_sequences=True))
        tag_model.add(Dropout(0.02))

        # tag_model.add(TimeDistributed(Dense(nn)))
        # tag_model.add(Dense(y.shape[1], activation='relu'))
        # tag_model.add(Dropout(0.02))

        return tags_dict, tag_model

        # # load the network weights
        # model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
        # # testing
        # for rn in range(10):
        #     print rn
        #     model.fit(X, y, nb_epoch=5, batch_size=64)  # , callbacks=callbacks_list)
        #     # pick a random seed
        #     start = numpy.random.randint(0, len(dataX) - 1)
        #     pattern = dataX[start]
        #     print "Seed:"
        #     print "\"", ' '.join([int_to_tag[value] for value in pattern]), "\""
        #     rs = []
        #     for i in range(300):
        #         x = numpy.reshape(pattern, (1, len(pattern), 1))
        #         x = x / float(tags_len)
        #         prediction = model.predict(x, verbose=0)
        #
        #         # index = numpy.argmax(prediction[0])
        #
        #         index = StructureModel.sample(prediction[0], 2.0)
        #         result = int_to_tag[index]
        #         seq_in = [int_to_tag[value] for value in pattern]
        #         sys.stdout.write(result)
        #         sys.stdout.write(" ")
        #         rs.append(index)
        #         pattern.append(index)
        #         pattern = pattern[1:len(pattern)]
        #     print "\nDone."

    # def words_model(self, struct):

    @classmethod
    def sample(cls, preds, temperature=1.0):
        # helper function to sample an index from a probability array
        # preds = np.asarray(preds).astype('float64')

        id_probs = sorted(enumerate(preds), key=lambda x: x[1], reverse=True)[0:5]
        ids = [v[0] for v in id_probs]
        probs = numpy.array([v[1] for v in id_probs]) / sum([v[1] for v in id_probs])

        return numpy.random.choice(ids, p=probs)

    @classmethod
    def semantic_model(cls, structure):
        semantic_model = sv.SemanticVector(structure)

        return semantic_model


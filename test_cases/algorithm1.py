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
        seq_length = 15
        word_list = struct.prepare_pure_list_of_words()
        # compute the vocabulary size
        vocabulary = sorted(list(set(word_list)))
        vocab_lenght = len(vocabulary)
        struct.generate_tags_dict()
        # semantic modeling
        semantic = StructureModel.semantic_model(struct)
        # tags modeling
        tag_model = StructureModel.tags_model(struct, seq_length, semantic)

    @classmethod
    def tags_model(cls, structure, seq_length, word2vec):
        total = 0
        for t in structure.sentences_obj:
            total += t.sentence_len

        avg = total / len(structure.sentences_obj)
        print "average length of sentence", avg

        tags_dict = collections.OrderedDict(sorted(structure.tags.items()))
        tags_len = len(tags_dict)

        word_list = structure.prepare_pure_list_of_words()
        vocabulary = sorted(list(set(word_list)))

        word_to_int = dict((c, i) for i, c in enumerate(vocabulary))
        int_to_word = dict((i, c) for i, c in enumerate(vocabulary))
        tag_to_int = dict((c, i) for i, c in enumerate(tags_dict))
        int_to_tag = dict((i, c) for i, c in enumerate(tags_dict))

        dataX = []
        wordsX = []
        dataY = []
        tagged_text = structure.tagged_text.split()
        n_tags_in_text = len(tagged_text)

        for i in range(0, n_tags_in_text - seq_length, 1):
            seq_in = tagged_text[i:i + seq_length]
            word_in = word_list[i:i + seq_length]
            seq_out = tagged_text[i + seq_length]
            dataX.append([tag_to_int[char] for char in seq_in])
            wordsX.append([word_to_int[word] for word in word_in])
            dataY.append(tag_to_int[seq_out])
        n_patterns = len(dataX)

        # reshape X to be [samples, time steps, features]
        X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
        # normalize
        X = X / float(tags_len)
        # one hot encode the output variable
        y = np_utils.to_categorical(dataY)

        print y.shape
        # define the LSTM model
        tag_model = Sequential()
        nn = 16

        tag_model.add(GRU(nn * 4, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
        tag_model.add(Dropout(0.02))

        tag_model.add(GRU(nn * 3, return_sequences=True))
        tag_model.add(Dropout(0.02))

        tag_model.add(GRU(nn * 2, return_sequences=True))
        tag_model.add(Dropout(0.02))

        tag_model.add(GRU(nn * 1, return_sequences=False))
        tag_model.add(Dropout(0.02))

        tag_model.add(Dense(y.shape[1], activation='sigmoid'))
        tag_model.add(Dropout(0.02))

        # # load the network weights
        tag_model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
        # testing
        for rn in range(1):
            print rn
            tag_model.fit(X, y, nb_epoch=1, batch_size=512)  # , callbacks=callbacks_list)
            # pick a random seed
            start = numpy.random.randint(0, len(dataX) - 1)
            pattern = dataX[start]
            word_pattern = wordsX[start]
            print "Seed:"
            # print "\"", ' '.join([int_to_tag[value] for value in pattern]), "\""
            print "\"", ' '.join([int_to_word[value] for value in word_pattern]), "\""
            rs = []
            for i in range(1):
                x = numpy.reshape(pattern, (1, len(pattern), 1))
                x = x / float(tags_len)
                prediction = tag_model.predict(x, verbose=0)

                # index = numpy.argmax(prediction[0])
                print prediction
                print 'size: ', len(prediction)
                index = StructureModel.sample(prediction[0], 2.0)

                result = int_to_tag[index]
                w_p_list = w.word_tokenize(" ".join(word_pattern))
                word_window = w_p_list[len(pattern) - 5: len(pattern)]
                word_window = [int_to_word[value] for value in word_window]
                StructureModel.find_nearest_word(word2vec, word_window, result, structure)
                # sys.stdout.write(result)
                # sys.stdout.write(" ")
                rs.append(index)
                pattern.append(index)
                pattern = pattern[1:len(pattern)]
            print "\nDone."

    # def words_model(self, struct):
    @classmethod
    def find_nearest_word(cls, word2vec, word_window, tag, structure):
        value_list = structure.tags[tag]
        comparison = dict()
        for new_word in value_list:
            temp = 0
            for window in word_window:
                temp += word2vec.model.similarity(new_word, window)
            temp /= len(word_window)
            comparison[new_word] = temp
        sorted_comparison = sorted(comparison.items(), key=operator.itemgetter(1))
        type(sorted_comparison)

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
        semantic_model.model_word2vec()
        return semantic_model
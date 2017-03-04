import collections

import operator
import numpy
import random
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
        y_len = 100
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
            dataX.append([word_to_int[word] for word in seq_in])
            if seq_out not in word2vec.wv.vocab:
                dataY.append(non_word2vec_list)
            else:
                dataY.append(word2vec[seq_out])
        n_patterns = len(dataX)

        # reshape X to be [samples, time steps, features]
        X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
        # normalize
        # X = X / float(vocab_length)

        print X.shape
        y = numpy.reshape(dataY, (n_patterns, y_len))

        print y.shape

        # define the LSTM model
        tag_model, tag_to_int, int_to_tag, tag_X, tag_y, tag_dataX, tags_dict = StructureModel.tags_model(structure,
                                                                                                          seq_length)

        # this loop reverse the tag dictionary
        word_to_tag_dict = dict()
        for new_value, new_keys in tags_dict.iteritems():
            for temp_key in new_keys:
                if temp_key in word_to_tag_dict.keys():
                    word_to_tag_dict[temp_key].append(new_value)
                else:
                    value_list = list()
                    value_list.append(new_value)
                    word_to_tag_dict[temp_key] = value_list

        # word_model.fit(X, y, nb_epoch=5, batch_size=512)
        # testing
        for rn in range(2):
            print rn
            tag_model.fit(tag_X, tag_y, nb_epoch=1, batch_size=32)
            # pick a random seed
            start = numpy.random.randint(0, len(dataX) - 1)
            tag_pattern = tag_dataX[start]
            pattern = dataX[start]
            print "Seed:"
            print "\"", ' '.join([int_to_word[value] for value in pattern]), "\""
            rs = []
            for i in range(15):
                tag_x = numpy.reshape(tag_pattern, (1, len(pattern), 1))
                tag_x = tag_x / float(len(tags_dict))
                tag_prediction = tag_model.predict(tag_x, verbose=0)
                index = StructureModel.sample(tag_prediction[0], 2.0)

                result = int_to_tag[index]
                temp_list = list()  # it is equivalents vectors to each word
                word_item_list = list()
                for word_item in pattern:
                    temp = int_to_word[word_item]
                    if temp in word2vec.wv.vocab:
                        word_item_list.append(temp)
                        temp_list.append(word2vec[temp].tolist())

                prd_word = StructureModel.find_nearest_words(word2vec, word_item_list, temp_list, result,
                                                             word_to_tag_dict, tags_dict)
                # seq_in = [int_to_word[value] for value in pattern]
                # print prediction[0]
                # word_for_print = prd_word.encode('utf-8')
                sys.stdout.write(prd_word[0])
                sys.stdout.write(' ')
                # rs.append(index)
                tag_pattern.append(index)
                tag_pattern = tag_pattern[1:len(tag_pattern)]
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
    def find_nearest_words(cls, word2vec, word_item_list, prediction_vec, tag, word_to_tags, tag_dict):
        model_word_vector = numpy.array(prediction_vec, dtype='f')
        topn = 20
        most_similar_words = word2vec.wv.most_similar(
            model_word_vector[len(model_word_vector) - 5:len(model_word_vector)], [], topn)
        # most_similar_words = most_similar_words.tolist()
        bad_list = list()
        for item in most_similar_words:
            test = False  # this word is not in this tag
            if item[0] in word_item_list:  # it is temporary. it is remove the repetitive word
                bad_list.append(item)
            for value in word_to_tags[item[0]]:
                if value == tag:
                    test = True
                    break
            if test is False:
                bad_list.append(item)

        most_similar_words = [word for word in most_similar_words if word not in bad_list]
        if len(most_similar_words) != 0:
            return random.choice(most_similar_words)
        else:
            return StructureModel.find_nearest_word_to_this_tag(tag, word_to_tags, word2vec, word_item_list, prediction_vec, tag_dict)

    @classmethod
    def find_nearest_word_to_this_tag(cls, tag, words_to_tags, word2vec, word_item_list, prediction_vec, tag_dict):
        value_list = tag_dict[tag]
        words_dict = dict()
        for value in value_list:
            if value not in word2vec.wv.vocab:
                continue
            temp_sim = word2vec.wv.similarity(word_item_list, value)
            words_dict[value] = numpy.mean(temp_sim)
        temp_list = sorted(words_dict.items(), key=operator.itemgetter(1))
        return temp_list[0]

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
            dataX.append([tag_to_int[tag] for tag in seq_in])
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
        tag_model.compile(loss='categorical_crossentropy', optimizer='adam')
        return tag_model, tag_to_int, int_to_tag, X, y, dataX, tags_dict

    @classmethod
    def semantic_model(cls, structure, seq_length, y_len):
        semantic_model = sv.SemanticVector(structure)
        semantic_model.model_word2vec(10, seq_length, y_len)
        semantic_model.save_model('weights.02.11.hdf5')
        return semantic_model

    @classmethod
    def sample(cls, preds, temperature=1.0):
        # helper function to sample an index from a probability array
        # preds = np.asarray(preds).astype('float64')

        id_probs = sorted(enumerate(preds), key=lambda x: x[1], reverse=True)[0:5]
        ids = [v[0] for v in id_probs]
        probs = numpy.array([v[1] for v in id_probs]) / sum([v[1] for v in id_probs])

        return numpy.random.choice(ids, p=probs)

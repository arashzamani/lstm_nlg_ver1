import os

import numpy as np
import collections
from collections import Counter

import language_parser.SemanticVector as sv
from utility.UFile import *
from utility.constants import *
import language_parser.Structure as structure
from keras.utils.np_utils import to_categorical

non_word2vec_list = [0.0] * embedding_dim


def prepare_multi_layer_train_sequence(tag2int, word2int, word_list, sentence_obj, vocab_length, is_sparse):
    tagX = prepare_tag_train_sequences(tag2int, sentence_obj)
    if not is_sparse:
        wordX, wordY = prepare_train_sequences(word2int, word_list, sentence_obj, vocab_length)
    else:
        wordX, wordY = prepare_train_sequences_for_sparse(word2int, word_list, sentence_obj)

    train_X = [tagX, wordX]
    return train_X, wordY


def prepare_tag_train_sequences(tag2int, sentences_obj):
    print "preparing tag sequences..."

    dataX = []
    for sentence in sentences_obj:
        for i in range(0, sentence.sentence_len - seq_length, 1):
            seq_in = sentence.tags[i:i + seq_length]
            dataX.append([tag2int[tag] for tag in seq_in])

    nb_pattern = len(dataX)
    train_tag_X = np.reshape(dataX, (nb_pattern, seq_length, 1))

    return train_tag_X


def prepare_test_sequences(tag2int, word2int):
    print "preparing test sequences"
    txt_file = UFile(os.getcwd() + test_text_file)
    structure_obj = structure.Structure(txt_file.text)
    word_list = structure_obj.prepare_pure_list_of_words()
    structure_obj.generate_tags_dict()
    vocabulary = sorted(list(set(word_list)))

    test_tagX = prepare_tag_train_sequences(tag2int, structure_obj.sentences_obj)
    test_word_X, test_word_Y = prepare_train_sequences(word2int, word_list, structure_obj.sentences_obj,
                                                       len(vocabulary), is_test=True)

    return [test_tagX, test_word_X]


def prepare_train_sequences(word2int, word_list, sentences_obj, vocab_length, is_test=False):
    print "preparing word sequences..."

    dataX = []
    dataY = np.zeros((len(word_list) - seq_length, vocab_length))

    for sentence in sentences_obj:
        for i in range(0, len(sentence.words) - seq_length, 1):
            seq_in = sentence.words[i:i + seq_length]
            seq_out = sentence.words[i + seq_length]
            temp = list()
            for word in seq_in:
                if word in word2int:
                    temp.append(word2int[word])
                else:
                    temp.append((word2int[unknown]))
            dataX.append(temp)
            if not is_test:
                if seq_out in word2int:
                    dataY[i, [word2int[seq_out]]] = 1
                else:
                    dataY[i, [word2int[unknown]]] = 1
    nb_pattern = len(dataX)
    if not is_test: dataY = dataY[0:nb_pattern, :]
    print "number of patterns=", nb_pattern

    train_X = np.reshape(dataX, (nb_pattern, seq_length))
    if not is_test:
        train_Y = np.reshape(dataY, (nb_pattern, vocab_length))
    else:
        train_Y = ''

    return train_X, train_Y


def prepare_train_sequences_for_sparse(word2int, word_list, sentences_obj):
    print "preparing word sequences..."

    dataX = []
    dataY = []

    for sentence in sentences_obj:
        for i in range(0, len(sentence.words) - seq_length, 1):
            seq_in = sentence.words[i:i + seq_length]
            seq_out = sentence.words[i + seq_length]
            dataX.append([word2int[word] for word in seq_in])
            dataY.append(word2int[seq_out])
            # if seq_out in word2vec.wv.vocab:
            #     dataY[i, [word2int[seq_out]]] = 1
            # else:
            #     dataY[i] = non_word2vec

    nb_pattern = len(dataX)
    print "number of patterns=", nb_pattern

    train_X = np.reshape(dataX, (nb_pattern, seq_length))
    train_Y = np.reshape(dataY, (nb_pattern, 1))

    return train_X, train_Y


def prepare_embedding(word_list, word2vec, word2int):
    print "Embedding data..."
    embeddings_index = dict()
    for word in word_list:
        if word in word2vec.wv.vocab:
            coefs = np.asarray(word2vec[word], dtype='float32')
        else:
            coefs = np.asanyarray(non_word2vec_list, dtype='float32')
        embeddings_index[word] = coefs

    embedding_matrix = np.zeros((len(word_list), embedding_dim))

    for word, i in word2int.iteritems():
        embedding_vector = embeddings_index[word]
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return embedding_matrix


def prepare_word_2_vec(semantic_vector_object):
    # check the existing word2vec
    # if os.path.exists(word_2_vec_filename):
    #     word2vec = semantic_vector_object.load_model(word_2_vec_filename)
    #     return word2vec.model

    semantic_vector_object.model_word2vec(min_count, seq_length)

    # semantic_vector_object.save_model(word_2_vec_filename)

    return semantic_vector_object.model


def equivalent_word_to_int(vocabulary):
    word_to_int = dict((c, i) for i, c in enumerate(vocabulary))
    int_to_word = dict((i, c) for i, c in enumerate(vocabulary))

    return word_to_int, int_to_word


def equivalent_tag_to_int(tags_dict):
    tag_to_int = dict((c, i) for i, c in enumerate(tags_dict))
    int_to_tag = dict((i, c) for i, c in enumerate(tags_dict))

    return tag_to_int, int_to_tag


def discover_characters(raw_text):
    C = Counter(raw_text)
    C = sorted([[c, C[c]] for c in C], key=lambda x: x[1], reverse=True)
    st = set([c[0] for c in C if c[1] > 1])
    raw_text = [d for d in raw_text if d in st]

    # create mapping of unique chars to integers, and a reverse mapping
    chars = sorted(list(set(raw_text)))
    char_to_int = dict((c, i) for i, c in enumerate(chars))
    int_to_char = dict((i, c) for i, c in enumerate(chars))
    # summarize the loaded data

    return chars, char_to_int, int_to_char


def generate_sequence_character(chars, char_to_int, sentences_obj, nb_patterns):
    print "generate character sequences..."
    dataY = np.zeros((nb_patterns, word_length))
    count_i = 0
    for sentence in sentences_obj:
        for i in range(0, len(sentence.words) - seq_length, 1):
            seq_out = sentence.words[i + seq_length]
            for j in range(0, len(seq_out), 1):
                temp = seq_out[j]
                if temp == u'_':
                    temp = u' '
                dataY[count_i, j] = char_to_int[temp]
            count_i += 1

    # train_y = right_align(dataY, len(chars))
    train_y = to_categorical(dataY, len(chars))
    return dataY


def right_align(seq, lengths):
    v = np.zeros(np.shape(seq))
    N = np.shape(seq)[1]
    for i in range(np.shape(seq)[0]):
        v[i][N - lengths[i]:N] = seq[i][0:lengths[i]]
    return v

import os

import numpy as np

import language_parser.SemanticVector as sv
from utility.constants import *
non_word2vec_list = [0.0] * embedding_dim


def prepare_train_sequences(word2int, word_list, sentences_obj, word2vec):
    print "preparing train sequences..."

    dataX = []
    dataY = list()

    for sentence in sentences_obj:
        for i in range(0, len(sentence.words) - seq_length, 1):
            seq_in = word_list[i:i + seq_length]
            seq_out = word_list[i + seq_length]
            dataX.append([word2int[word] for word in seq_in])
            if seq_out in word2vec.wv.vocab:
                dataY.append(word2vec[seq_out])
            else:
                dataY.append(non_word2vec_list)

    nb_pattern = len(dataX)

    print "number of patterns=", nb_pattern

    train_X = np.reshape(dataX, (nb_pattern, seq_length))
    train_Y = np.reshape(dataY, (nb_pattern, embedding_dim))

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

    semantic_vector_object.model_word2vec(5, seq_length)

    # semantic_vector_object.save_model(word_2_vec_filename)

    return semantic_vector_object.model


def equivalent_word_to_int(word_list):
    vocabulary = sorted(list(set(word_list)))
    word_to_int = dict((c, i) for i, c in enumerate(vocabulary))
    int_to_word = dict((i, c) for i, c in enumerate(vocabulary))

    return word_to_int, int_to_word

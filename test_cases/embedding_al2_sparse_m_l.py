import sys

from keras.layers import Embedding, LSTM, Dropout, Dense, Conv1D, MaxPooling1D, Merge
from keras.models import Sequential

import language_parser.Structure as structure
from language_parser.prepare_data import *
from utility.UFile import *


def get_model():
    print "Reading text file..."
    dir_path = os.getcwd() + text_file
    txt_file = UFile(dir_path)
    structure_obj = structure.Structure(txt_file.text)
    word_list = structure_obj.prepare_pure_list_of_words()
    word_list.append(unknown)
    vocabulary = sorted(list(set(word_list)))
    word_to_int, int_to_word = equivalent_word_to_int(vocabulary)
    structure_obj.generate_tags_dict()
    tags_dict = collections.OrderedDict(sorted(structure_obj.tags.items()))
    tag_to_int, int_to_tag = equivalent_tag_to_int(tags_dict)
    semantic_vector_obj = sv.SemanticVector(structure_obj)
    word2vec = prepare_word_2_vec(semantic_vector_obj)
    print "Start Modeling..."
    embedding_matrix = prepare_embedding(word_list, word2vec, word_to_int)

    nb_classes = len(vocabulary)
    model = modeling(embedding_matrix, len(word_list), 0.05, nb_classes)

    print len(word2vec.wv.vocab)
    train_X, train_y = prepare_multi_layer_train_sequence(tag_to_int, word_to_int, word_list,
                                                          structure_obj.sentences_obj, len(vocabulary), is_sparse=True)
    test_X = prepare_test_sequences(tag_to_int, word_to_int)
    train_model(model, train_X, train_y, 1, 128, int_to_word, test_X, structure_obj, tag_to_int)


def train_model(model, train_X, train_y, nb_epoch, batch_size, int_to_word, test_X, structure_obj, tag_to_int):
    if os.path.exists(os.getcwd() + word_2_vec_filename_embedding_al2_sparse_m_l):
        print "Loading Weights..."
        model.load_weights(os.getcwd() + word_2_vec_filename_embedding_al2_sparse_m_l)
    for rn in range(100):
        print rn
        model.fit(train_X, train_y, epochs=nb_epoch, batch_size=batch_size)  # , callbacks=callbacks_list)
        model.save(os.getcwd() + word_2_vec_filename_embedding_al2_sparse_m_l)
        # pick a random seed
        start = np.random.randint(0, len(test_X[0]) - 1)
        test_tag_list = [(test_X[0][start])]
        test_word_list = [(test_X[1][start])]
        pattern = [np.reshape(test_tag_list, (1, seq_length, 1)), np.reshape(test_word_list, (1, seq_length))]  # dataX

        print "Seed:"
        print start
        for p in pattern[1][0]:
            if p in int_to_word:
                sys.stdout.write(int_to_word[p] + ' ')
            else:
                sys.stdout.write(unknown)
        sys.stdout.write('---')
        for i in range(15):
            # x = np.reshape(pattern, (2, 1, seq_length))
            preds = model.predict(pattern, verbose=0)[0]
            index = sample(preds, 2.0)
            sys.stdout.write(int_to_word[index] + ' ')
            pattern[1][0] = np.append(pattern[1][0][1:seq_length], index)
            temp = find_predicted_tag(int_to_word, pattern[1][0], structure_obj, tag_to_int)
            pattern[0][0] = temp
        print "\nDone."


def find_predicted_tag(int_to_word, word_sequence, structure_obj, tag_to_int):
    temp = list()
    for word in word_sequence:
        temp.append(int_to_word[word])
    # temp_sentence = " ".join(temp)
    tags = structure_obj.public_parse_tags(temp)

    tags_sequence = list()
    for tag in tags:
        tags_sequence.append(tag_to_int[tag[1]])
    tags_sequence = np.reshape(tags_sequence, (seq_length, 1))
    return tags_sequence


def modeling(embedding_matrix, nb_words, dropout_rate, nb_classes):
    print "modeling..."
    t_model = tag_model(dropout_rate, nb_classes)
    word_model = word2vec_model(embedding_matrix, nb_words, dropout_rate, nb_classes)

    merged_model = Sequential()
    print "final model merging..."
    merged_model.add(Merge([t_model, word_model], mode='concat'))
    merged_model.add(Conv1D(activation="relu", padding="same", filters=32 * 9, kernel_size=3))
    merged_model.add(MaxPooling1D(pool_size=3))

    merged_model.add(LSTM(128 * 3 * 3, return_sequences=True))
    merged_model.add(Dropout(dropout_rate))

    merged_model.add(LSTM(128 * 3, return_sequences=True))
    merged_model.add(Dropout(dropout_rate / 5))

    merged_model.add(Conv1D(filters=32 * 2, kernel_size=3, padding='same', activation='relu'))
    merged_model.add(MaxPooling1D(pool_size=3))

    word_model.add(LSTM(32 * 3 * 3 * 3, return_sequences=True))
    word_model.add(Dropout(dropout_rate))

    merged_model.add(LSTM(units=512 * 4, return_sequences=False))
    merged_model.add(Dropout(dropout_rate))
    merged_model.add(Dense(512 * 4, activation='tanh'))
    merged_model.add(Dropout(dropout_rate))
    merged_model.add(Dense(nb_classes, activation='softmax'))
    merged_model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy',
                         metrics=['accuracy'])
    print merged_model.summary()
    return merged_model


def tag_model(dropout_rate, nb_classes):
    print "tag Modeling..."
    model = Sequential()
    model.add(LSTM(units=512, return_sequences=True, input_shape=(seq_length, 1)))
    # model.add(Conv1D(activation="relu", padding="same", filters=32 * 9, kernel_size=3))
    # model.add(MaxPooling1D(pool_size=3))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(units=512 * 2, return_sequences=True))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1024, activation='sigmoid'))
    print model.summary()
    return model


def word2vec_model(embedding_matrix, nb_words, dropout_rate, nb_classes):
    print "text Modeling..."
    model = Sequential()
    model.add(Embedding(nb_words, embedding_dim, weights=[embedding_matrix], input_length=seq_length, trainable=False))
    # model.add(Conv1D(activation="relu", padding="same", filters=32 * 9, kernel_size=3))
    # model.add(MaxPooling1D(pool_size=3))
    model.add(LSTM(units=512, return_sequences=True, input_shape=(seq_length, embedding_dim)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(512 * 2, activation='tanh'))
    print model.summary()
    return model


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def find_nearest_words(word2vec, prediction_vec):
    model_word_vector = np.array(prediction_vec, dtype='f')
    topn = 20
    most_similar_words = word2vec.wv.most_similar([model_word_vector], [], topn)

    return most_similar_words[0]


def start_embedding_al2_sparse_m_l():
    print "Starting..."
    get_model()

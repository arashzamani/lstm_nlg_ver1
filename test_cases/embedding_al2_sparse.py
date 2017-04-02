import sys

from keras.layers import Embedding, LSTM, Dropout, Dense, Conv1D, MaxPooling1D
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
    vocabulary = sorted(list(set(word_list)))
    word_to_int, int_to_word = equivalent_word_to_int(vocabulary)
    semantic_vector_obj = sv.SemanticVector(structure_obj)
    word2vec = prepare_word_2_vec(semantic_vector_obj)
    print "Start Modeling..."
    embedding_matrix = prepare_embedding(word_list, word2vec, word_to_int)

    nb_classes = len(vocabulary)
    model = word2vec_model(embedding_matrix, len(word_list), 0.05, nb_classes)

    print len(word2vec.wv.vocab)
    train_X, train_y = prepare_train_sequences_for_sparse(word_to_int, word_list, structure_obj.sentences_obj)

    train_model(model, train_X, train_y, 1, 128, int_to_word, word2vec, word_to_int)


def train_model(model, train_X, train_y, nb_epoch, batch_size, int_to_word, word2vec, word_to_int):
    if os.path.exists(os.getcwd() + word_2_vec_filename_embedding_al2_sparse):
        print "Loading Weights..."
        model.load_weights(os.getcwd() + word_2_vec_filename_embedding_al2_sparse)
    for rn in range(100):
        print rn
        model.fit(train_X, train_y, epochs=nb_epoch, batch_size=batch_size)  # , callbacks=callbacks_list)
        model.save(os.getcwd() + word_2_vec_filename_embedding_al2_sparse)
        # pick a random seed
        start = np.random.randint(0, len(train_X) - 1)
        pattern = train_X[start].tolist()  # dataX

        print "Seed:"
        print start
        print(pattern)
        for p in pattern:
            sys.stdout.write(int_to_word[p] + ' ')
        sys.stdout.write('---')
        for i in range(15):
            x = np.reshape(pattern, (1, seq_length))
            preds = model.predict(x, verbose=0)[0]
            index = sample(preds, 2.0)
            sys.stdout.write(int_to_word[index] + ' ')
            pattern.append(index)
            pattern = pattern[1:len(pattern)]
        print "\nDone."


def word2vec_model(embedding_matrix, nb_words, dropout_rate, nb_classes):
    print "text Modeling..."
    model = Sequential()
    model.add(Embedding(nb_words, embedding_dim, weights=[embedding_matrix], input_length=seq_length, trainable=False))
    model.add(Conv1D(activation="relu", padding="same", filters=32 * 9, kernel_size=3))
    model.add(MaxPooling1D(pool_size=3))
    model.add(LSTM(32 * 3, return_sequences=True))
    model.add(Dropout(dropout_rate))

    model.add(LSTM(32 * 3 * 3, return_sequences=True))
    model.add(Dropout(dropout_rate / 5))

    model.add(Conv1D(filters=32 * 2, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=3))

    model.add(LSTM(32 * 3 * 3 * 3, return_sequences=True))
    model.add(Dropout(dropout_rate))

    model.add(LSTM(units=512 * 2, return_sequences=True))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(units=512 * 4, return_sequences=False))
    model.add(Dense(512 * 4, activation='tanh'))
    # model.add(LSTM(units=nb_classes, return_sequences=False))
    model.add(Dropout(dropout_rate))
    model.add(Dense(2000, activation='tanh'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(nb_classes, activation='softmax'))
    # model.add(Dropout(dropout_rate))
    # model.add(Dense(embedding_dim, activation='softmax'))
    model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    print model.summary()
    return model


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    # preds = np.asarray(preds).astype('float64')

    id_probs = sorted(enumerate(preds), key=lambda x: x[1], reverse=True)[0:5]
    ids = [v[0] for v in id_probs]
    probs = np.array([v[1] for v in id_probs]) / sum([v[1] for v in id_probs])

    return np.random.choice(ids, p=probs)


def find_nearest_words(word2vec, prediction_vec):
    model_word_vector = np.array(prediction_vec, dtype='f')
    topn = 20
    most_similar_words = word2vec.wv.most_similar([model_word_vector], [], topn)

    return most_similar_words[0]


def start_embedding_al2_sparse():
    print "Starting..."
    get_model()

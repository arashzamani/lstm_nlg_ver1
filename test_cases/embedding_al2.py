import sys

from keras.layers import Embedding, LSTM, Dropout, Dense
from keras.models import Sequential

import language_parser.Structure as structure
from language_parser.prepare_data import *
from utility.UFile import *


def get_model():
    print "Reading text file..."
    txt_file = UFile(text_file)
    structure_obj = structure.Structure(txt_file.text)
    word_list = structure_obj.prepare_pure_list_of_words()
    word_to_int, int_to_word = equivalent_word_to_int(word_list)
    semantic_vector_obj = sv.SemanticVector(structure_obj)
    word2vec = prepare_word_2_vec(semantic_vector_obj)
    print "Start Modeling..."
    embedding_matrix = prepare_embedding(word_list, word2vec, word_to_int)

    model = word2vec_model(embedding_matrix, len(word_list), 0.05)

    train_X, train_y = prepare_train_sequences(word_to_int, word_list, structure_obj.sentences_obj, word2vec)

    train_model(model, train_X, train_y, 1, 128, int_to_word,  word2vec, word_to_int)


def train_model(model, train_X, train_y, nb_epoch, batch_size, int_to_word, word2vec, word_to_int):
    for rn in range(100):
        print rn
        model.fit(train_X, train_y, nb_epoch=nb_epoch, batch_size=batch_size)  # , callbacks=callbacks_list)
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
            prd_word = find_nearest_words(word2vec, preds)
            print preds
            sys.stdout.write(prd_word[0] + ' ')
            pattern.append(word_to_int[prd_word[0]])
            pattern = pattern[1:len(pattern)]
        print "\nDone."


def word2vec_model(embedding_matrix, nb_words, dropout_rate):
    print "text Modeling..."
    model = Sequential()
    model.add(Embedding(nb_words, embedding_dim, weights=[embedding_matrix], input_length=seq_length, trainable=False))
    model.add(LSTM(output_dim=512, return_sequences=True, input_shape=(seq_length, embedding_dim)))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(output_dim=512, return_sequences=True))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1024, activation='tanh'))
    model.add(LSTM(output_dim=1000, return_sequences=False))
    model.add(Dropout(dropout_rate))
    model.add(Dense(embedding_dim, activation='softmax'))
    model.compile(optimizer='rmsprop', loss='mean_squared_error',
                  metrics=['accuracy'])

    return model

def find_nearest_words( word2vec, prediction_vec):
    model_word_vector = np.array(prediction_vec, dtype='f')
    topn = 20
    most_similar_words = word2vec.wv.most_similar([model_word_vector], [], topn)

    return most_similar_words[0]

print "Starting..."
get_model()

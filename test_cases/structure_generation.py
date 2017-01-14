import language_parser.Sentence as Sentence
from utility.UFile import *
import language_parser.Structure as structure
import language_parser.SemanticVector as sv
from keras.layers import Input, Embedding, LSTM, Dense, merge
from keras.models import Model
import language_parser.Word as word
import utility.TreeParser as treeParser


class StructureModel:
    def __init__(self, file):
        self.file = file

    def model(self):
        struct = structure.Structure(self.file.text)
        struct.generate_tags_dict()
        # tags modeling
        tags_input, tags_output = self.tags_model(struct)
        # semantic modeling
        semantic = self.semantic_model(struct)

    @classmethod
    def tags_model(cls, structure):
        total = 0
        for t in structure.sentences_obj:
            total += t.sentence_len

        avg = total / len(structure.sentences_obj)
        print "average length of sentence", avg

        tags_len = len(structure.tags.keys())

        # tag_to_int = dict((c, i) for i, c in enumerate(structure.tagged_text))
        # int_to_tag = dict((i, c) for i, c in enumerate(structure.tagged_text))
        tags_input = Input(shape=(tags_len,), dtype='int32', name='tags_input')

        x = Embedding(output_dim=tags_len, input_dim=tags_len, input_length=100)(tags_input)

        x = Dense(tags_len * 2, activation='relu')(x)
        x = Dense(tags_len * 2, activation='relu')(x)
        x = Dense(tags_len * 2, activation='relu')(x)

        tags_output = Dense(1, activation='sigmoid', name='main_output')(x)

        return tags_input, tags_output
    # def words_model(self, struct):

    @classmethod
    def semantic_model(cls, structure):
        semantic_model = sv.SemanticVector(structure)

        return semantic_model

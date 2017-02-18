import language_parser.Sentence as Sentence
from utility.UFile import *
import language_parser.Structure as structure
import language_parser.SemanticVector as sm
import language_parser.Word as word
import utility.TreeParser as treeParser
import test_cases.structure_generation as sg
import test_cases.algorithm1 as al1
import test_cases.algorithm2 as al2
import test_cases.algorithm3 as al3


def main():
    print('Started...')
    #myfile = UFile('bbc_shortened.txt')
    myfile = UFile('bbc.txt')
    #myfile = open('~/Arash_z/lstm_nlg_ver1/bbc_shortened.txt', 'rb')
    # model = sg.StructureModel(myfile)
    # model.model()

    al = al3.StructureModel(myfile)
    al.model()
    # struct = structure.Structure(myfile.text)
    # # struct.generate_chunks_dict()
    # struct.generate_tags_dict()
    # print struct.tagged_text

main()

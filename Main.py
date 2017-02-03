import language_parser.Sentence as Sentence
from utility.UFile import *
import language_parser.Structure as structure
import language_parser.SemanticVector as sm
import language_parser.Word as word
import utility.TreeParser as treeParser
import test_cases.structure_generation as sg
import test_cases.algorithm1 as al1


def main():
    print('Started...')
    myfile = UFile('/home/arash/Downloads/bbc30.12.txt')
    model = sg.StructureModel(myfile)
    model.model()
    # al = al1.StructureModel(myfile)
    # al.model()
    # struct = structure.Structure(myfile.text)
    # # struct.generate_chunks_dict()
    # struct.generate_tags_dict()
    # print struct.tagged_text

main()

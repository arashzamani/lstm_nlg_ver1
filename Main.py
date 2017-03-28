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
import test_cases.word_embedding_al1 as we 
import test_cases.algorithm2_with_embedding as al2we
import test_cases.algorithm2_w_emb_cls as al2wecls 
def main():
    print('Started...')
    #myfile = UFile('bbc_shortened.txt')
    #myfile = UFile('shams.txt')
    #myfile = UFile('masnavi2.txt')
    myfile = UFile('baba_taher.txt')
    #myfile = UFile('bbc.txt')
    #myfile = UFile('hafez.txt')
    # model = sg.StructureModel(myfile)
    # model.model()

    #al = al3.StructureModel(myfile)
    #al = al2.StructureModel(myfile)
    #al = al2we.StructureModel(myfile)
    al = al2wecls.StructureModel(myfile)
    al.model()
    # struct = structure.Structure(myfile.text)
    # # struct.generate_chunks_dict()
    # struct.generate_tags_dict()
    # print struct.tagged_text

main()

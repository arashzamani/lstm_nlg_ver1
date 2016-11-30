import language_parser.Sentence as Sentence
from utility.UFile import *
import language_parser.Structure as structure
import utility.TreeParser as treeParser


def main():
    print('Started...')
    myfile = UFile('/home/arash/Downloads/test1.txt')
    struct = structure.Structure(myfile.text)
    struct.generate_chunks_dict()
    for item in struct.chunks.keys():
        print item, len(struct.chunks[item])



    # struct.generate_chunks_dict()
    # for item in struct.chunks.keys():
    #     print item, len(struct.chunks[item])


main()

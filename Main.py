import language_parser.Sentence as Sentence
from utility.UFile import *
import language_parser.Structure as structure
import language_parser.Word as word
import utility.TreeParser as treeParser


def main():
    print('Started...')
    myfile = UFile('/home/arash/Downloads/test1.txt')
    struct = structure.Structure(myfile.text)
    # struct.generate_chunks_dict()
    struct.generate_tags_dict()
    print struct.tagged_text

main()

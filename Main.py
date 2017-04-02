import test_cases.algorithm2_w_emb_cls as al2wecls
import test_cases.algorithm3_w_emb_cls as al3wecls
from test_cases.embedding_al2 import *
from test_cases.embedding_al2_sparse import *
from test_cases.embedding_al2_sparse_m_l import *


def main(algorithm_no, text_file_no):
    print('Started...')
    filename = ''
    if text_file_no == 1:
        filename = 'hafez.txt'
    elif text_file_no == 2:
        filename = 'shahnameh.txt'
    elif text_file_no == 3:
        filename = 'bbc.txt'
    elif text_file_no == 4:
        filename = 'bbc_shortened.txt'

    myfile = UFile(filename)

    if algorithm_no == 1:
        al = al2wecls.StructureModel(myfile)
        al.model()
    elif algorithm_no == 2:
        al = al3wecls.StructureModel(myfile)
        al.model()
    elif algorithm_no == 3:
        start_embedding_al2()
    elif algorithm_no == 4:
        start_embedding_al2_sparse()
    elif algorithm_no == 5:
        start_embedding_al2_sparse_m_l()


main(3, 1)

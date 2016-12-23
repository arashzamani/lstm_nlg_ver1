from __future__ import unicode_literals
import codecs

class UFile:

    #consturctor get the file path to load file in a utf8 text
    def __init__(self, file_name):
        self.__fileName = file_name
        loaded_text = codecs.open(file_name, 'r', encoding='utf8')
        self.text = loaded_text.read()

    def __get__text(self):
        return self.text



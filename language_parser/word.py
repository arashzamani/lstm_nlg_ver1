from __future__ import unicode_literals
from hazm import *
import codecs

specials = [u'\u0020', u'\u0027', u'\u0022',
            u'\u0021', u'\u061F', u'\u060c',
            u'\u061B', u'\u003A', u'\u003E',
            u'\u003C', u'\u002D', u'\u005F',
            u'\u0025', u'\u0029', u'\u0028',
            u'\u002A', u'\u005C', u'\u002F',
            u'\u002E', u'\u00D8\u008C']


def pure_word_tokenize(text):
    words = word_tokenize(text)
    words = remove_special_character(words)
    words = remove_numbers(words)
    words = remove_bad_chars_from_word(words)
    return words

# def pure_word_tokenize_list(list):


def remove_numbers(words):
    bad_chars = dict()
    for j in range(0, len(words), 1):
        if check_number(words[j]) == 1:
            bad_chars[words[j]] = 1
    for keys in bad_chars.keys():
        words = [value for value in words if value != keys]
    return words

# this function get an string and examining does it number or not
# if it was a number it return 1
# else it return 0


def check_number(item):
    try:
        temp = int(item)
        return 1
    except ValueError:
        return 0


def remove_special_character(words):
    bad_chars = dict()
    for j in range(0, len(words), 1):
        if check_special_characters(words[j]) == 1:
            bad_chars[words[j]] = 1
    for keys in bad_chars.keys():
        words = [value for value in words if value != keys]
    return words


def check_special_characters(item):
    # space ' " ! ? , ; : > < - _ % ) ( * \ / . virgulfarsi

    check = 0
    for temp in specials:
        if item == temp:
            check = 1
            break

    return check


def remove_bad_chars_from_word(words):
    for i in range(0, len(words), 1):
        for bads in specials:
            if words[i][0] == bads:
                words[i] = words[i][1:]
            if words[i][len(words[i]) - 1] == bads:
                words[i] = words[i][:-1]
    return words

# f = codecs.open("/home/arash/Downloads/bbc.txt", 'r', encoding='utf8')
# text = f.read()
#
# words1 = pure_word_tokenize(text)
# print(len(words1))
#
#
# unique_words = dict()
#
# for i in range(0, len(words1), 1):
#     if words1[i] in unique_words.keys():
#         unique_words[words1[i]] += 1
#     else:
#         unique_words[words1[i]] = 1
#
# print ("unique words: ", len(unique_words))
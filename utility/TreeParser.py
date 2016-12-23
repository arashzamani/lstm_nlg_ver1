from __future__ import unicode_literals
from hazm import *
import re


def parse_tree_to_dict(tree):
    result = dict()
    temp_list = re.findall('\[.*?\]', tree)
    for item in temp_list:
        item = item.replace('[', "")
        item = item.replace(']', "")
        q = item.split()
        temp_key = q[len(q)-1]
        value = ""
        for i in range(0, len(q) - 1, 1):
            value += " " + q[i]
        value = value.strip()
        if temp_key in result.keys():
            result[temp_key].append(value)
        else:
            temp_value = list()
            temp_value.append(value)
            result[temp_key] = temp_value

    return result



# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 14:39:52 2020

@author: nikolaos damianos
"""
import re
import numpy as np


def classes_str_to_int(targets):
    #classe from string to int 
    # for example cat = 0 , dog =1 , bird = 3
    v = {}

    uni = np.unique(targets)

    for i in range(len(uni)):
        v[uni[i]] = i

    ret_targets = []
    for i in range(len(targets)):
        ret_targets.append(v[targets[i]])

    return ret_targets




def clean_str(string):
    """
    Tokenization/string cleaning for datasets.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"\'s", "", string)
    string = re.sub(r"\'ve", "", string)
    string = re.sub(r"n\'t", "", string)
    string = re.sub(r"\'re", "", string)
    string = re.sub(r"\'d", "", string)
    string = re.sub(r"\'ll", "", string)
    string = re.sub(r",", "", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", "", string)
    string = re.sub(r"\)", "", string)
    string = re.sub(r"\?", "", string)
    string = re.sub(r"'", "", string)
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"[0-9]\w+|[0-9]", "", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()
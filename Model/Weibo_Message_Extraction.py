#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Graduate Project: Named Entity Recognition Task in Chinese Media
Author: Bin Wei
Description: This is part of my graduate project, which tries to explore some
            models targeting NER tasks in Chinese social media.
"""

def extract_weibo_message():
    """
    This method selects any Weibo messages with length over 60 for later manual annotations.
    """
    chars_ls = []
    with open("../Data/train.txt") as file:
        for line in file:
            line_ls = line.split(",")
            if len(line_ls[2]) > 60:
                char_ls = [char for char in line_ls[2]]
                for char in char_ls:
                    chars_ls.append(char)

    with open("../Data/Weibo_Messages.txt", "w") as file:
        for char in chars_ls:
            file.write(char)
            file.write("\n")

if __name__ == "__main__":
    extract_weibo_message()
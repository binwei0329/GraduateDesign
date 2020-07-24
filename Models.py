import numpy as np
import regex as re
def read_file(file):
    tag_list = []
    word_ls = []
    with open(file) as f:
        for line in f:
            if line == "\n":
                continue
            else:
                # Resolve the encoding problem.
                if "\u200b" in line:
                    line = re.sub("\u200b", "X", line)
                ls = line.split("\t")

                word_ls.append(ls[0])
                tag_list.append(ls[-1])

    tag_list = sorted(set(tag_list))

    # Exclude the last tag symbol, "X".
    tag_list = tag_list[:-1]

    # Clean the items in the tag_list and construct a dictionary out of the list.
    for i in range(len(tag_list)):
        s = tag_list[i].rstrip("\n")
        tag_list[i] = s
    tag_dic = {n:m for m, n in enumerate(tag_list)}

    print(tag_dic)
    # word_ls = sorted(set(word_ls))
    print(len(word_ls))
    # print(word_ls)
    lsl = []
    temp = ""
    char = []
    for i in range(len(word_ls)):
        char_ls = [char for char in word_ls[i]]
        char.extend(char_ls)
        # print(char_ls)
        # s = "".join(char_ls[0])
        temp += char_ls[0]
        # print(s)
    # for i in range(len(word_ls)):
        if word_ls[i].endswith("0") and i > 0:
            lsl.append(temp[:-1])
            temp = temp[-1] + ""
            # print(True)
            # # print(word_ls[i].split())
            # print([char for char in word_ls[i]])

    lsl = sorted(set(lsl))
    # print(lsl)
    print(len(lsl))

    # print(lsl.index("一"))
    # print(lsl.index("龟"))
    # print(lsl[lsl.index("一"):lsl.index("龟")])
    # print(ij)
    dic = {v:k for k, v in enumerate(lsl)}
    # print(dic)
    l = sorted(set(char))
    print(len(l[l.index("一"):l.index("龟")]))
    print(len(l))



read_file("Weibo_NER_Corpus.train")



def read_file(file):
    tag_list = []
    word_ls = []
    with open(file) as f:
        for line in f:
            if line == "\n":
                continue
            else:
                if "\u200b" in line:
                    str.replace("\u200b", " ", " ")
                ls = line.split("\t")
                # str.replace("\u200b"," ")
                # if ls.index("\u200b") >= 0:
                #     str.replace("\u200b", " ")
                word_ls.append(ls[0])
                tag_list.append(ls[-1])

    tag_list = sorted(set(tag_list))
    ij = tag_list[-1]
    # tag_list = tag_list[:-1]

    for i in range(len(tag_list)):
        s = tag_list[i].rstrip("\n")
        tag_list[i] = s
    print(tag_list)

    tag_dic = {n:m for m, n in enumerate(tag_list)}

    print(tag_dic)
    # word_ls = sorted(set(word_ls))
    print(word_ls)
    for i in range(len(word_ls)):
        if word_ls[i].endswith("0"):
            pass
            


    # print(ij)
read_file("Weibo_NER_Corpus.train")


# print("\u200b")
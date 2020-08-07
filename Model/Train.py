import pickle
import numpy as np
import tensorflow as tf
from collections import Counter
from Model.BiLSTM_CRF import BiLSTM_CRF, train_one_step, predict, calculate_metrics
from Model.Preprocess import convert_data, format_data, load_char_embeddings

def traing_BiLSTM_CRF():
    EMBED_DIM = 50
    HIDDEN_DIM = 32
    EPOCH = 10
    LEARNING_RATE = 0.01

    char_embed_dict = load_char_embeddings("../Data/vec.txt")
    with open ("../Data/weiboNER_2nd_conll.train.pkl", "rb") as file_test:
        tag_dic_t = pickle.load(file_test)
        char_dic_t = pickle.load(file_test)
        word_dic_t = pickle.load(file_test)
        sentence_list_t = pickle.load(file_test)
        tags_t = pickle.load(file_test)
        data_test = pickle.load(file_test)
        label_test = pickle.load(file_test)
    length = len(data_test)

    with open ("../Data/weiboNER_Corpus.train.pkl", "rb") as file_train:
        tag_dic = pickle.load(file_train)
        char_dic = pickle.load(file_train)
        word_dic = pickle.load(file_train)
        sentence_list = pickle.load(file_train)
        tags = pickle.load(file_train)
        data_train = pickle.load(file_train)
        label_train = pickle.load(file_train)

    data_duplicate = data_train[length:]
    label_duplicate = label_train[length:]

    # Oversampling the named entities.
    for i in range(3):
        data_train.extend(data_duplicate)
        label_train.extend(label_duplicate)
    # print(len(data_train))
    # Undersampling the data without named entities.
    for i in range(length):
        label = label_train[i]
        # for l in label:
        #     if label
        dic = Counter(label)
        # print(dic, "\t", len(label))
        if dic[16] == len(label):
            del label_train[i]
            del data_train[i]

    data_train, label_train = format_data(data_train, label_train)
    vocab_size = len(char_dic)
    tag_size = len(tag_dic)

    # print(len(data_train))
    tf.print(tf.shape(data_train))
    tf.print(tf.shape(label_train))

    # print(3622 // 128) 28
    train_dataset = tf.data.Dataset.from_tensor_slices((data_train, label_train))
    train_dataset = train_dataset.shuffle(len(data_train)).batch(128, drop_remainder=True)
    #
    #
    model = BiLSTM_CRF(HIDDEN_DIM, vocab_size, tag_size, EMBED_DIM)
    optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)

    pre, rec, f1score = 0, 0, 0
    for e in range(EPOCH):
        for _, (data_batch, label_batch) in enumerate(train_dataset):
            # step += 1
            loss, logits, text_lens = train_one_step(model, data_batch, label_batch, optimizer)
            # if step % 20 == 0:
            #     accuracy = get_acc_one_step(logits, text_lens, label_batch, model)
            #     if accuracy > best_acc:
            #         best_acc = accuracy
    # tf.print(best_acc)
            labels = np.array(label_batch)
            alt_labels = []
            for label, text_len in zip(labels, text_lens):
                alt_labels.append(label[:text_len])
            alt_labels = np.array(alt_labels)
            # print(alt_labels.shape)
            # print("*" * 68)
            predictions = predict(logits, text_lens, label_batch, model)
            predictions = np.array(predictions)
            # print(predictions.shape)
            # print(predictions)
            # print("*" * 100)
            # print(alt_labels)
            # print("-" * 100)
            # precision, recall, f1 = calculate_metrics(predictions, alt_labels)
            # pre += precision
            # rec += recall
            # f1score += f1
            print(loss)
    #
    # pre /= EPOCH
    # rec /= EPOCH
    # f1score /= EPOCH
    # print("Precision:", pre, " Recall:", rec, " f1score:", f1score)
    # with open ("../Data/weiboNER_2nd_conll.train.pkl", "rb") as file_test:
    #     tag_dic_t = pickle.load(file_test)
    #     char_dic_t = pickle.load(file_test)
    #     word_dic_t = pickle.load(file_test)
    #     sentence_list_t = pickle.load(file_test)
    #     tags_t = pickle.load(file_test)
    #     data_test = pickle.load(file_test)
    #     label_test = pickle.load(file_test)
    # data_test, label_test = format_data(data_test, label_test)
    # vocab_size_t = len(char_dic_t)
    # tag_size_t = len(tag_dic_t)
    # tf.print(data_test.shape)
    # tf.print(label_test.shape)
    # # print(tag_dic)
    # # print(char_dic)
    # test_dataset = tf.data.Dataset.from_tensor_slices((data_test, label_test))
    # test_dataset = test_dataset.shuffle(len(data_test)).batch(64, drop_remainder=True)
    # #
    # # model = BiLSTM_CRF(HIDDEN_DIM, vocab_size_t, tag_size_t, EMBED_DIM)
    # # optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)
    # #
    # best_acc_t, step_t = 0, 0
    # for e in range(EPOCH):
    #     for _, (data_batch, label_batch) in enumerate(test_dataset):
    #         step_t += 1
    #         loss_t, logits_t, text_lens_t = train_one_step(model, data_batch, label_batch, optimizer)
    #         # tf.print(loss_t)
    #         if step_t % 20 == 0:
    #             accuracy_t = get_acc_one_step(logits_t, text_lens_t, label_batch, model)
    #             # tf.print(accuracy_t)
    #             if accuracy_t > best_acc_t:
    #                 best_acc_t = accuracy_t
    #
    # tf.print(best_acc_t)

    # with open ("../Data/weiboNER_2nd_conll.test.pkl", "rb") as file_test:
    #     tag_dic_t = pickle.load(file_test)
    #     char_dic_t = pickle.load(file_test)
    #     word_dic_t = pickle.load(file_test)
    #     sentence_list_t = pickle.load(file_test)
    #     tags_t = pickle.load(file_test)
    #     data_test_t = pickle.load(file_test)
    #     label_test_t = pickle.load(file_test)
    # data_test_t, label_test_t = format_data(data_test_t, label_test_t)
    # vocab_size_t = len(char_dic_t)
    # tag_size_t = len(tag_dic_t)
    # tf.print(data_test_t.shape)
    # tf.print(label_test_t.shape)
    # # print(tag_dic)
    # # print(char_dic)
    # test_dataset_t = tf.data.Dataset.from_tensor_slices((data_test_t, label_test_t))
    # test_dataset_t = test_dataset_t.shuffle(len(data_test_t)).batch(128, drop_remainder=True)
    # #
    # # model = BiLSTM_CRF(HIDDEN_DIM, vocab_size, tag_size, EMBED_DIM)
    # # optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)
    # #
    # best_acc_t, step_t = 0, 0
    # for e in range(EPOCH):
    #     for _, (data_batch, label_batch) in enumerate(test_dataset_t):
    #         step_t += 1
    #         loss_t, logits_t, text_lens_t = train_one_step(model, data_batch, label_batch, optimizer)
    #         if step_t % 20 == 0:
    #             accuracy_t = get_acc_one_step(logits_t, text_lens_t, label_batch, model)
    #             # tf.print(accuracy_t)
    #             if accuracy_t > best_acc_t:
    #                 best_acc_t = accuracy_t
    #
    # tf.print(best_acc_t)

if __name__ == "__main__":
    traing_BiLSTM_CRF()
    # with open ("../Data/weiboNER_2nd_conll.train.pkl", "rb") as file_test:
    #     tag_dic_t = pickle.load(file_test)
    #     char_dic_t = pickle.load(file_test)
    #     word_dic_t = pickle.load(file_test)
    #     sentence_list_t = pickle.load(file_test)
    #     tags_t = pickle.load(file_test)
    #     data_test_t = pickle.load(file_test)
    #     label_test_t = pickle.load(file_test)
    # dataset = np.array(data_test_t)
    # labelset = np.array(label_test_t)
    #
    # with open ("../Data/weiboNER_2nd_conll.test.pkl", "rb") as file:
    #     tag_dic = pickle.load(file)
    #     char_dic = pickle.load(file)
    #     word_dic = pickle.load(file)
    #     sentence_list = pickle.load(file)
    #     tags = pickle.load(file)
    #     data = pickle.load(file)
    #     label = pickle.load(file)
    # data = np.array(data)
    # label = np.array(label)
    #
    # print(data[0])
    # print(label[0])
    #
    # print(dataset[0])
    # print(labelset[0])
    # print(tag_dic)

    # p = [[16, 16, 7, 15, 16, 16, 2, 8],
    #      [16, 16, 16, 5, 8, 16, 16, 16]]
    # l = [[16, 7, 15, 16, 16, 16, 2, 8],
    #      [16, 16, 16, 5, 8, 16, 8, 6]]
    # precision, recall, f1 = calculate_metrics(p, l)
    # print(precision, "\t", recall, "\t", f1)
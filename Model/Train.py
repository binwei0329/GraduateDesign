import pickle
import tensorflow as tf
from Model.BiLSTM_CRF import BiLSTM_CRF, train_one_step, get_acc_one_step
from Model.Preprocess import convert_data, format_data

def traing_BiLSTM_CRF():
    EMBED_DIM = 64
    HIDDEN_DIM = 32
    EPOCH = 5
    LEARNING_RATE = 0.005

    with open ("../Data/BiLSTM_CRF_data.pkl", "rb") as file:
        tag_dic = pickle.load(file)
        char_dic = pickle.load(file)
        word_dic = pickle.load(file)
        sentence_list = pickle.load(file)
        tags = pickle.load(file)
        data = pickle.load(file)
        label = pickle.load(file)
    # print(data, "\n", label)
    data, label = format_data(data, label)
    # tf.print(data)
    # tf.print(label)
    vocab_size = len(char_dic)
    tag_size = len(tag_dic)
    # print(char_dic)
    # print(tag_dic)
    tf.print(data.shape)
    tf.print(label.shape)

    train_dataset = tf.data.Dataset.from_tensor_slices((data, label))
    train_dataset = train_dataset.shuffle(len(data)).batch(128, drop_remainder=True)

    model = BiLSTM_CRF(HIDDEN_DIM, vocab_size, tag_size, EMBED_DIM)
    optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)

    best_acc, step = 0, 0
    for e in range(EPOCH):
        for _, (data_batch, label_batch) in enumerate(train_dataset):
            step += 1
            loss, logits, text_lens = train_one_step(model, data_batch, label_batch, optimizer)
            if step % 20 == 0:
                accuracy = get_acc_one_step(logits, text_lens, label_batch, model)
                if accuracy > best_acc:
                    best_acc = accuracy

    tf.print(best_acc)

if __name__ == "__main__":
    traing_BiLSTM_CRF()
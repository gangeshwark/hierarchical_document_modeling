import argparse
import re

import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf
from bs4 import BeautifulSoup
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.utils.np_utils import to_categorical
from sklearn.metrics import classification_report
from tensorflow.python.ops.rnn_cell_impl import GRUCell
from tqdm import tqdm

from model import HAN_Model
from utils import batch_iter

MAX_SENT_LENGTH = 200  # max in doc is 966
MAX_SENTS = 50  # max in doc is 282
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2


def clean_str(string):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    return string.strip().lower()


data_train = pd.read_csv('data/labeledTrainData.tsv', sep='\t')
print(data_train.shape)

from nltk import tokenize

reviews = []
labels = []
texts = []

for idx in range(data_train.review.shape[0]):
    text = BeautifulSoup(data_train.review[idx])
    text = clean_str(text.get_text())
    texts.append(text)
    sentences = tokenize.sent_tokenize(text)
    reviews.append(sentences)

    labels.append(data_train.sentiment[idx])

tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)

data = np.zeros((len(texts), MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')
word_length = np.zeros((len(texts), MAX_SENTS), dtype='int32')
sent_length = np.zeros((len(texts)), dtype='int32')
# word_length = []
for i, sentences in enumerate(reviews):
    # sent_len is number of sentences in a doc
    sent_len = 0
    for j, sent in enumerate(sentences):
        if j < MAX_SENTS:
            sent_len += 1
            wordTokens = text_to_word_sequence(sent)
            # k is number of words in a sentence
            k = 0
            for _, word in enumerate(wordTokens):
                if k < MAX_SENT_LENGTH and tokenizer.word_index[word] < MAX_NB_WORDS:
                    data[i, j, k] = tokenizer.word_index[word]
                    k = k + 1
            word_length[i, j] = k
    sent_length[i] = sent_len
# print(data[:5])
# print(sent_length[:5])
# print(word_length[:5])

print(max(sent_length))
print(max(np.reshape(word_length, [-1])))
# exit(0)
print(data.shape)
word_index = tokenizer.word_index
vocab = list(word_index.keys())
print('Total %s unique tokens.' % len(vocab))
vocab_size = len(vocab)
print(vocab[:10])
# exit(0)

labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)
classes = labels.shape[-1]

indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
sent_length = sent_length[indices]
word_length = word_length[indices]
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

X_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
sent_length_train = sent_length[:-nb_validation_samples]
word_length_train = word_length[-nb_validation_samples:]

X_val = data[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]
sent_length_val = sent_length[-nb_validation_samples:]
word_length_val = word_length[-nb_validation_samples:]

print('x_train', X_train.shape)
print('y_train', y_train.shape)
print('x_val', X_val.shape)
print('y_val', y_val.shape)

print('Number of positive and negative reviews in training and validation set')
print(y_train.sum(axis=0))
print(y_val.sum(axis=0))

# building Hierachical Attention network
allow_soft_placement = True
log_device_placement = False


def validate(epoch, model, sess, X_val, sent_length_val, word_length_val, y_val, batch_size, is_training=False):
    batches = batch_iter(list(zip(X_val, sent_length_val, word_length_val, y_val)),
                         batch_size)

    l = []
    a = []
    all_preds = []
    for i, batch in tqdm(enumerate(batches)):
        X_batch, sent_len_batch, word_lenght_batch, y_batch = zip(
            *batch)
        # print('batch_hist_v', len(batch_utt_v))
        feed_dict = {
            model.inputs: X_batch,
            model.sentence_lengths: sent_len_batch,
            model.word_lengths: word_lenght_batch,
            model.labels: y_batch,
            model.is_training: is_training,
        }

        step, loss, accuracy, predictions = sess.run([model.global_step, model.loss, model.accuracy, model.prediction],
                                                     feed_dict)

        l.append(loss)
        a.append(accuracy)
        all_preds.append(predictions)

    all_preds = np.concatenate(all_preds, axis=0)
    acc = np.average(a)
    print("EVAL: Epoch {}:, loss {:g}, Accuracy {:g}".format(epoch, np.average(l), acc))
    precision = sklearn.metrics.precision_score(np.argmax(y_val, 1), all_preds,
                                                average='weighted')

    # print(precision)
    recall = sklearn.metrics.recall_score(np.argmax(y_val, 1), all_preds, average='weighted')
    # print(recall)
    F1 = sklearn.metrics.f1_score(np.argmax(y_val, 1), all_preds, average='weighted')
    print("\tPrecision: {:g} ; Recall: {:g} ; F1 {:g}".format(precision, recall, F1))
    report = classification_report(np.argmax(y_val, 1), all_preds)
    return acc, report


def train(epochs, batch_size):
    session_conf = tf.ConfigProto(
        # device_count={'GPU': gpu_count},
        allow_soft_placement=allow_soft_placement,
        log_device_placement=log_device_placement,
        gpu_options=tf.GPUOptions(allow_growth=True))
    # Training
    # ==================================================
    best_acc = 0
    best_epoch = 0
    best_report = ''
    gpu_device = 0
    with tf.device('/device:GPU:%d' % gpu_device):
        print('Using GPU - ', '/device:GPU:%d' % gpu_device)
        with tf.Graph().as_default():
            sess = tf.Session(config=session_conf)
            with sess.as_default():
                seed = 1227

                kernel_init = tf.glorot_uniform_initializer(seed=seed, dtype=tf.float32)
                bias_init = tf.zeros_initializer()
                word_cell = GRUCell(150, name='gru', activation=tf.nn.tanh,
                                    kernel_initializer=kernel_init, bias_initializer=bias_init)
                sent_cell = GRUCell(150, name='gru', activation=tf.nn.tanh,
                                    kernel_initializer=kernel_init, bias_initializer=bias_init)

                model = HAN_Model(
                    vocab_size=vocab_size,
                    embedding_size=200,
                    classes=classes,
                    word_cell=word_cell,
                    sentence_cell=sent_cell,
                    word_output_size=100,
                    sentence_output_size=100,
                    device=args.device,
                    learning_rate=args.lr,
                    dropout_keep_proba=0.5,
                    scope='HANModel'
                )
                sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
                # tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

                print("\nEvaluation before training:")
                # Evaluation after epoch
                validate(-1, model, sess, X_val, sent_length_val, word_length_val, y_val, batch_size)

                for epoch in range(epochs):
                    epoch += 1

                    batches = batch_iter(zip(X_train, sent_length_train, word_length_train, y_train),
                                         batch_size)

                    # Training loop. For each batch...
                    print('\nTraining epoch {}'.format(epoch))
                    l = []
                    a = []
                    for i, batch in tqdm(enumerate(list(batches))):
                        X_batch, sent_len_batch, word_lenght_batch, y_batch = zip(
                            *batch)
                        # print('batch_hist_v', len(batch_utt_v))
                        feed_dict = {
                            model.inputs: X_batch,
                            model.sentence_lengths: sent_len_batch,
                            model.word_lengths: word_lenght_batch,
                            model.labels: y_batch,
                            model.is_training: True,
                        }

                        _, step, loss, accuracy = sess.run(
                            [model.train_op, model.global_step, model.loss, model.accuracy], feed_dict)
                        l.append(loss)
                        a.append(accuracy)

                    print("\t \tEpoch {}:, loss {:g}, Accuracy {:g}".format(epoch, np.average(l), np.average(a)))
                    # Evaluation after epoch
                    accuracy, report = validate(epoch, model, sess, X_val, sent_length_val, word_length_val, y_val,
                                                batch_size)

                    if accuracy > best_acc:
                        best_epoch = epoch
                        best_acc = accuracy
                        best_report = report

                print("\n\nBest epoch: {}\nBest test accuracy: {}".format(best_epoch, best_acc))
                print("\n\nBest epoch: {}\nBest test report: \n{}".format(best_epoch, best_report))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='imdb', choices=['imdb'])
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=30)
    parser.add_argument("--device", default="/gpu:0")
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()
    train(args.epochs, args.batch_size)

import tensorflow as tf
import os
import re
import unicodedata
import io
import csv
import emoji

from nltk.tokenize import TweetTokenizer
from tf_glove.tf_glove import GloVeModel


def preprocess_sentence(sentence):
    sentence = '<start> ' + ' '.join(sentence.split()) + ' <end>'
    sentence = sentence.lower()
    return sentence


def create_dataset(path, num_examples):
    dataset = []
    with open(path, newline='',  encoding="utf8") as data_file:
        reader = csv.reader(data_file, delimiter=',', quotechar='\"')
        for row in reader:
            side_a = '<start> ' + row[0] + ' <end>'
            side_b = '<start> ' + row[1] + ' <end>'
            # normalize all whitespaces to space
            side_a = ' '.join(side_a.split()).lower()
            side_b = ' '.join(side_b.split()).lower()
            dataset.append([side_a, side_b])
    return dataset


def tokenize(sentences, model):
    tokenizer = TweetTokenizer()
    tensor = []
    for sentence in sentences:
        tokens = [emoji.demojize(token.lower())
                  for token in tokenizer.tokenize(sentence)]
        vector = []
        for token in tokens:
            try:
                vector.append(model.id_for_word(token))
            except:
                pass
        tensor.append(vector)

    tensor = tf.keras.preprocessing.sequence.pad_sequences(
        tensor, padding='post')

    print(tensor.shape)
    return tensor


def load_dataset(glove_model, path, num_examples=None):
    dataset = create_dataset(path, num_examples)

    input_data = map(lambda x: x[0], dataset)
    output_data = map(lambda x: x[1], dataset)

    input_tensor = tokenize(input_data, glove_model)
    target_tensor = tokenize(output_data, glove_model)

    return input_tensor, target_tensor

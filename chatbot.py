#!/usr/bin/python3

import tensorflow as tf

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split

import unicodedata
import re
import numpy as np
import os
import io
import time
import fileinput
from optparse import OptionParser
from model import Encoder, Decoder
from dataset import load_dataset, preprocess_sentence
from tf_glove.tf_glove import GloVeModel
import emoji

parser = OptionParser()
parser.add_option('-t', '--train', action='store_true', dest='train')

(options, args) = parser.parse_args()

glove_model = GloVeModel()
glove_model.load_from_file('data/glove.local.txt')

num_examples = 100000
input_tensor, target_tensor = load_dataset(
    glove_model, 'data/reddit_merged.csv', num_examples)

max_length_targ, max_length_inp = target_tensor.shape[1], input_tensor.shape[1]
input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(
    input_tensor, target_tensor, test_size=0.2)

BUFFER_SIZE = len(input_tensor_train)
BATCH_SIZE = 64
steps_per_epoch = len(input_tensor_train)//BATCH_SIZE
units = 256

embedding_dim = glove_model.embedding_size
vocab_size = glove_model.vocab_size

dataset = tf.data.Dataset.from_tensor_slices(
    (input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

embedding_matrix = np.zeros((vocab_size, embedding_dim))
for i in range(vocab_size):
    embedding_matrix[i] = np.asfarray(glove_model.embeddings[i])

encoder = Encoder(vocab_size, embedding_dim, embedding_matrix,
                  max_length_inp, units, BATCH_SIZE)
decoder = Decoder(vocab_size, embedding_dim,
                  embedding_matrix, max_length_targ, units)

optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


@tf.function
def train_step(inp, targ, enc_hidden):
    loss = 0

    with tf.GradientTape() as tape:
        enc_output, enc_hidden_h, enc_hidden_c = encoder(inp, enc_hidden)
        enc_hidden = [enc_hidden_h, enc_hidden_c]
        dec_hidden = enc_hidden_h
        dec_input = tf.expand_dims(
            [glove_model.id_for_word('<start>')] * BATCH_SIZE, 1)

        for t in range(1, targ.shape[1]):
            predictions, dec_hidden, _, _ = decoder(
                dec_input, dec_hidden, enc_output)
            loss += loss_function(targ[:, t], predictions)
            dec_input = tf.expand_dims(targ[:, t], 1)

    batch_loss = (loss / int(targ.shape[1]))
    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    return batch_loss


if options.train:
    EPOCHS = 10
    for epoch in range(EPOCHS):
        start = time.time()

        enc_hidden = encoder.initialize_hidden_state()
        total_loss = 0

        for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
            batch_loss = train_step(inp, targ, enc_hidden)
            total_loss += batch_loss

            print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                         batch,
                                                         batch_loss.numpy()))
        checkpoint.save(file_prefix=checkpoint_prefix)

        print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                            total_loss / steps_per_epoch))
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
else:
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))


def evaluate(sentence):
    attention_plot = np.zeros((max_length_targ, max_length_inp))

    sentence = preprocess_sentence(sentence)

    inputs = []
    for word in sentence.split(' '):
        try:
            inputs.append(glove_model.id_for_word(emoji.demojize(word)))
        except:
            pass

    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                           maxlen=max_length_inp,
                                                           padding='post')
    inputs = tf.convert_to_tensor(inputs)

    result = ''

    hidden = [tf.zeros((1, units)), tf.zeros((1, units))]
    enc_out, enc_hidden_h, enc_hidden_c = encoder(inputs, hidden)

    dec_hidden_h = enc_hidden_h
    dec_input = tf.expand_dims([glove_model.id_for_word('<start>')], 0)

    for t in range(max_length_targ):
        predictions, dec_hidden_h, dec_hidden_c, _ = decoder(dec_input,
                                                             dec_hidden_h,
                                                             enc_out)

        predicted_id = tf.argmax(predictions[0]).numpy()
        result += glove_model.words[predicted_id] + ' '

        if glove_model.words[predicted_id] == '<end>':
            return result, sentence
        dec_input = tf.expand_dims([predicted_id], 0)

    return result, sentence


def translate(sentence):
    result, sentence = evaluate(sentence)
    return result


while True:
    text = input('User: ')
    response = None
    try:
        response = translate(text)
    except Exception as exc:
        print(exc)
        print('Unknown word!')

    if response:
        print('Bot: ', end='')
        print(response)

#!/usr/bin/python3

import tensorflow as tf

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split

import traceback
import unicodedata
import re
import numpy as np
import os
import io
import time
import fileinput
from optparse import OptionParser
from model import Encoder, Decoder
from dataset import load_dataset, preprocess_sentence, tokenize_sentence
from tf_glove.tf_glove import GloVeModel
import emoji
from pathlib import Path
import datetime

parser = OptionParser()
parser.add_option('-t', '--train', action='store_true', dest='train')
parser.add_option('-d', '--dir', action='store', dest='dir')

(options, args) = parser.parse_args()

glove_path = 'data/glove.local.txt'
dataset_path = 'data/reddit_merged.csv'
checkpoint_dir = './training_checkpoints'
BATCH_SIZE = 256
EPOCHS = 40
units = 768

if options.dir:
    with open(os.path.join(options.dir, 'config'), 'r', encoding='utf-8', newline='') as file:
        lines = file.read().splitlines()
        BATCH_SIZE = int(lines[0])
        units = int(lines[1])
        glove_path = lines[2]
        dataset_path = lines[3]
        checkpoint_dir = options.dir

checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

glove_model = GloVeModel()
glove_model.load_from_file(glove_path)

num_examples = 100000
input_tensor, target_tensor = load_dataset(
    glove_model, dataset_path, num_examples)

max_length_targ, max_length_inp = target_tensor.shape[1], input_tensor.shape[1]
input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(
    input_tensor, target_tensor, test_size=0.2)

BUFFER_SIZE = len(input_tensor_train)
steps_per_epoch = len(input_tensor_train)//BATCH_SIZE

embedding_dim = glove_model.embedding_size
vocab_size = glove_model.vocab_size

dataset = tf.data.Dataset.from_tensor_slices(
    (input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

embedding_matrix = np.zeros((vocab_size, embedding_dim))
for i in range(vocab_size):
    if len(glove_model.embeddings[i]) != 100:
        print(i)
        continue
    embedding_matrix[i] = np.asfarray(glove_model.embeddings[i])

print('created embedding_matrix')

print("dataset_size: {}, vocab_size: {}, input max length: {}, output max_length {}".format(
    input_tensor.shape[0], vocab_size, max_length_inp, max_length_targ))

encoder = Encoder(vocab_size, embedding_dim, embedding_matrix,
                  max_length_inp, units, BATCH_SIZE)
decoder = Decoder(vocab_size, embedding_dim,
                  embedding_matrix, max_length_targ, units)

optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

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
            [glove_model.word_to_id['<start>']] * BATCH_SIZE, 1)

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
    print('training')
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(checkpoint_dir, "config"), 'w', newline='',  encoding="utf8") as config_file:
        config_file.write(str(BATCH_SIZE) + '\n')
        config_file.write(str(units) + '\n')
        config_file.write(glove_path + '\n')
        config_file.write(dataset_path + '\n')

    start_epoch = 0
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    if latest_checkpoint:
        print("latest checkpoint: {}".format(latest_checkpoint))
        checkpoint.restore(latest_checkpoint)
        start_epoch = int(latest_checkpoint[len(checkpoint_prefix) + 1:])
        print("last checkpoint epoch: {}".format(start_epoch + 1))

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    for epoch in range(start_epoch, EPOCHS):
        start = time.time()

        enc_hidden = encoder.initialize_hidden_state()
        total_loss = 0

        batch_number = 0

        for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
            batch_loss = train_step(inp, targ, enc_hidden)
            total_loss += batch_loss

            print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                         batch,
                                                         batch_loss.numpy()))
            batch_number += 1
            if batch_number % 5 == 0:
                with test_summary_writer.as_default():
                    tf.summary.scalar('loss', batch_loss.numpy(),
                                      step=epoch*steps_per_epoch + batch_number)

        checkpoint.save(file_prefix=checkpoint_prefix)

        print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                            total_loss / steps_per_epoch))
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
else:
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))


def evaluate(sentence):
    attention_plot = np.zeros((max_length_targ, max_length_inp))

    inputs = tokenize_sentence(sentence, glove_model)

    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                           maxlen=max_length_inp,
                                                           padding='post')
    inputs = tf.convert_to_tensor(inputs)

    result = ''

    hidden = [tf.zeros((1, units)), tf.zeros((1, units))]
    enc_out, enc_hidden_h, enc_hidden_c = encoder(inputs, hidden)

    dec_hidden_h = enc_hidden_h
    dec_input = tf.expand_dims([glove_model.word_to_id['<start>']], 0)

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
        traceback.print_exc()
        print(exc)
        print('Unknown word!')

    if response:
        print('Bot: ', end='')
        print(response)

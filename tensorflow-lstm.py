import os
import argparse

import numpy as np
import tensorflow as tf

import audio_utilities

#import dask.array as da


def prepare_input(inputs):
    _input=[]
    for entry in inputs:
        for k in entry:
            if k.startswith('neutral-mel-new') :
                loaded=np.load(os.path.join('training-new', k))
            #    print(loaded.shape)
                difference = 214-loaded.shape[0]
                loaded= np.pad(loaded, [(0, difference), (0, 0)], mode='constant')
                #loaded=loaded.flatten()
                _input.append(loaded)
                # _input.append(loaded.shape[0])
    _input= np.asarray(_input)
    # print(max(_input))
   # _input= da.from_array(_input, chunks=(100))
    return _input


def prepare_input_test(inputs):
  test=[]
  for entry in inputs:
        for k in entry:
            if k.startswith('neutral-mel-new') :
                loaded=np.load(os.path.join('testing-new', k))
            #    print(loaded.shape)
                difference = 214-loaded.shape[0]
                loaded= np.pad(loaded, [(0, difference), (0, 0)], mode='constant')
                #loaded=loaded.flatten()
                test.append(loaded)
                # test.append(loaded.shape[0])

  test= np.asarray(test)
  # print(max(test))

  #test= da.from_array(test, chunks=(100))

  return test

def prepare_targets(targets):
    target=[]
    for entry in targets:
        for k in entry:
            if k.startswith('happy-mel-new') :
                loaded=np.load(os.path.join('training-new', k))
                #print(loaded.shape)
                difference = 214-loaded.shape[0]
                loaded= np.pad(loaded, [(0, difference), (0, 0)], mode='constant')
                #loaded=loaded.flatten()
                target.append(loaded)
                # target.append(loaded.shape[0])
    target= np.asarray(target)

    #target= da.from_array(target, chunks=(100))
    # print(max(target))
    return target

def prepare_target_test(targets):
  test=[]
  for entry in targets:
        for k in entry:
            if k.startswith('happy-mel-new') :
                loaded=np.load(os.path.join('testing-new', k))
            #    print(loaded.shape)
                difference = 214-loaded.shape[0]
                loaded= np.pad(loaded, [(0, difference), (0, 0)], mode='constant')
                # #loaded=loaded.flatten()
                test.append(loaded)
                #test.append(loaded.shape[0])

  test= np.asarray(test)
  # print(max(test))
   #test = da.from_array(test, chunks=(100))
  return test

session = tf.Session()

#
input_placeholder = tf.placeholder(tf.float32, (32, 189, 1025))
target_placeholder = tf.placeholder(tf.float32, (32, 189, 1025))

# input_placeholder = tf.placeholder(tf.float32, (39, 214, 200))
# target_placeholder = tf.placeholder(tf.float32, (39, 214, 200))

#input_test = tf.placeholder(tf.float32, (39, 214, 200))

# lstm = tf.contrib.rnn.BasicLSTMCell(100)
lstm = tf.nn.rnn_cell.BasicLSTMCell(100)

rnn, states = tf.nn.dynamic_rnn(lstm, input_placeholder, dtype=tf.float32)


# session.run(tf.initialize_all_variables)


layer_1 = tf.layers.dense(rnn, 512)
layer_2 = tf.layers.dense(layer_1, 2000)
layer_3 = tf.layers.dense(layer_2, 1025)
# sigmoid = tf.nn.sigmoid(layer_3)


# x = np.random.rand(32, 189,1025)
# y = np.random.rand(32, 189,1025)

parser = argparse.ArgumentParser()
parser.add_argument('--base_dir', default=os.path.expanduser('/home/youmna/Documents/Neural Network Happy'))
args = parser.parse_args()

input_path = os.path.join(args.base_dir, 'training-new/train.txt')

with open(input_path, encoding='utf-8') as t:
    training_set_path = [line.strip().split('|') for line in t]
#
test_path = os.path.join(args.base_dir, 'testing-new/train.txt')
#
with open(test_path, encoding='utf-8') as t:
    test_set_path = [line.strip().split('|') for line in t]

x = prepare_input(training_set_path)
y = prepare_targets(training_set_path)
#
x_test = prepare_input_test(test_set_path)
y_test = prepare_target_test(test_set_path)
epochs = 50000
batch_size = 32
rate = 0.01
PRINT_RATE = 2
save_ckpt=500
#
#
x_dataset = tf.data.Dataset.from_tensor_slices(x)
x_dataset =  x_dataset.repeat()
x_dataset = x_dataset.batch(batch_size)
x_tensor = x_dataset.make_one_shot_iterator()
next_input = x_tensor.get_next()

y_dataset = tf.data.Dataset.from_tensor_slices(x)
y_dataset =  y_dataset.repeat()
y_dataset = y_dataset.batch(batch_size)
y_tensor = y_dataset.make_one_shot_iterator()
next_target =y_tensor.get_next()


# print(session.run(y_tensor.get_next()).shape)
# print(layer_3.shape)
cross_entropy = tf.losses.sigmoid_cross_entropy(y_tensor.get_next(), layer_3)

training_step = tf.train.AdadeltaOptimizer(rate).minimize(cross_entropy)

print("x_")

init_op = [tf.global_variables_initializer(), tf.local_variables_initializer()]
session.run(init_op)
saver = tf.train.Saver()

#saver.save(session, '/home/youmna/Documents/Neural Network Happy/ckpts/model.ckpt', global_step=500)
# input_test = tf.placeholder(tf.float32, (1, 214, 200))

# input_lengths = tf.placeholder(tf.int32, [1], 'input_lengths')
# saver.restore(session, '/home/youmna/Documents/Neural Network Happy/ckpts/model.ckpt-500')
#
# predictions = session.run(layer_3, feed_dict={input_placeholder: x_test})
# print(predictions[0])
#
#
# min_freq_hz = 70
# max_freq_hz = 8000
# mel_bin_count = 200
# hopsamp = 2048 // 8
#
#
# linear_bin_count = 1 + 2048//2
# filterbank = audio_utilities.make_mel_filterbank(min_freq_hz, max_freq_hz, mel_bin_count,
#                                                  linear_bin_count , 44100)
# inverted_mel_to_linear_freq_spectrogram = np.dot(filterbank.T, predictions[0].T)
#
# stft_modified = inverted_mel_to_linear_freq_spectrogram.T
#
# # stft_modified_scaled = stft_modified / scale
# stft_modified = stft_modified**0.5
#
# x_reconstruct = audio_utilities.reconstruct_signal_griffin_lim(stft_modified,
#                                                                2048, hopsamp,
#                                                                300)
# max_sample = np.max(abs(x_reconstruct))

# print(max_sample)
#
# if max_sample > 1.0:
#     x_reconstruct = x_reconstruct / max_sample
#
# audio_utilities.save_audio_to_file(x_reconstruct, 44100)





for i in range(epochs):
    # session.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
    _, error = session.run([training_step, cross_entropy], feed_dict={input_placeholder: session.run(next_input), target_placeholder: session.run(next_target)})


    if i % PRINT_RATE == 0:
        print("ERROR:", error)
    if i % save_ckpt == 0:
        saver.save(session, '/home/youmna/Documents/Neural Network Happy/ckpts/model.ckpt', global_step=500)
    #   save_path = saver.save(session, "/ckpts/model.ckpt- %i" % i)
        print("Model Saved")
    # session.close()
session.close()

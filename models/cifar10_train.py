#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 qizai <jianhao2@illinois.edu>
#
# Distributed under terms of the MIT license.

"""
This is a naive training resnet.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import base64
import argparse
import numpy as np
import tensorflow as tf


# class TEMP_opt:
#     def __init__(self):
#         self.nClass = 10
#         self.stride = 1
#         self.sparsity = 0.9
#         self.nInputPlane = 3
#         # self.numChannels = 128 # number of intermediate layers between blocks, i.e. nChIn
#         # self.number_of_b = 512 # number of binary filters in LBC, i.e. nChTmp
#         # self.full = 512 # number of hidden units in FC
#         self.numChannels = 1 # number of intermediate layers between blocks, i.e. nChIn
#         self.number_of_b = 9 # number of binary filters in LBC, i.e. nChTmp
#         self.full = 20 # number of hidden units in FC
#         self.convSize = 3 # LB convolutional filter size
#         self.depth = 20 # number of blocks
#         self.weightDecay = 1e-4
#         self.LR = 1e-4 #initial learning rate
#         self.nEpochs = 3 # number of total epochs to run
#         # self.epochNumber = 1 # manual epoch number
#         self.batch_size = 128
#         self.data_format = 'channels_last'
#         self.shared_weights = False


# these two params are use in batch norm
_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5

def str2bool(v):
    if v.lower() in ('yes', 'true', '1', 'y', 't'):
        return True
    elif v.lower() in ('no', 'false', '0', 'n', 'f'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected')

def batch_norm_relu(inputs, is_training, data_format):
  """Performs a batch normalization followed by a ReLU."""
  # We set fused=True for a significant performance boost. See
  # https://www.tensorflow.org/performance/performance_guide#common_fused_ops
  # # another error cause by r1.2. Unfortunately.
  # inputs = tf.layers.batch_normalization(
  #     inputs=inputs, axis=1 if data_format == 'channels_first' else 3,
  #     momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
  #     scale=True, training=is_training, fused=True)
  inputs = tf.layers.batch_normalization(
      inputs=inputs, axis=1 if data_format == 'channels_first' else 3,
      momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
      scale=True, training=is_training)
  inputs = tf.nn.relu(inputs)
  return inputs

def random_binary_convlution(inputs, nChIn, nChOut, kW, kH, dW, dH, 
        padding, data_format, sparsity, shared_weights):
    """
    inputs: a tensor
    nChIn/nChOut: number of input/output channels
    kW/kH : kernel size in LBC
    dW/dH : stride size in LBC
    padding : str, padding or not in RBC.
    # padW/padH: padding size
    sparsity: proportion of non zero elements in ancher weights
    shared_weights: boolean, whether uses same binary filter or not.
    """
    ancher_shape = np.array([kW, kH, nChIn, nChOut])
    num_elements = np.product(ancher_shape, dtype = int)
    num_non_zero = num_elements * sparsity
    num_non_zero = num_non_zero.astype(int)

    #initialize ancher weights
    ancher_weights = np.zeros(shape = ancher_shape, dtype = np.float)
    ancher_weights = np.reshape(ancher_weights, newshape = [num_elements])
    if shared_weights:
        np.random.seed(42)
    index_non_zero = np.random.choice(num_elements, num_non_zero, replace = False)
    for i in index_non_zero:
        ancher_weights[i] = np.random.binomial(1, 0.5) * 2 - 1
    ancher_weights = np.reshape(ancher_weights, newshape = ancher_shape)
    ancher_weights_tensor = tf.constant(ancher_weights, dtype = tf.float32)

    if data_format == 'channels_first':
        tf_format = 'NCHW'
        tf_strides = [1, 1, dH, dW]
    else:
        tf_format = 'NHWC'
        tf_strides = [1, dH, dW, 1]
    
    diff_map = tf.nn.conv2d(inputs, filter = ancher_weights_tensor,
                            strides = tf_strides, padding = padding,
                            data_format = tf_format)
    return diff_map
    

def basic_block_LBC(inputs, nChIn, nChTmp, kSize, is_training, data_format, 
        sparsity, shared_weights, block_name):
    """
    basic resnet block, with LBC module replacement.
    nChIn : number of input channels to the block. Notice that the in/out of a
            block has the same 'depth'/number of channels
    nChTmp: number of binary filters used in the block. Cancel out by the 
            second conv in block.
    kSize:  filter size in RBC.
    is_training: a boolean, tell if training
    data_format: 'channels_first' or 'channels_last'
    sparsity/shared_weights: params used in RBC.
    block_name: string. name of block.
    """
    with tf.name_scope(block_name):
        shortcut = inputs
        with tf.name_scope('batch_normalization'):
            inputs = batch_norm_relu(inputs, is_training, data_format)
        with tf.name_scope('random_binary_conv'):
            inputs = random_binary_convlution(inputs, nChIn = nChIn, nChOut = nChTmp,
                    kW = kSize, kH = kSize, dW = 1, dH = 1, padding = 'SAME',
                    data_format = data_format, sparsity = sparsity,
                    shared_weights = shared_weights)

        inputs = tf.nn.relu(inputs)
        with tf.name_scope('1x1_conv'):
            # the second conv doesn't need any padding, since it's 1x1.
            inputs = tf.layers.conv2d(inputs = inputs, filters = nChIn,
                                    kernel_size = [1, 1],
                                    padding = 'valid',
                                    data_format = data_format,
                                    use_bias = False)
        output = shortcut + inputs
    return output

def cifar10_resnet_LBC_generator(depth, nClass, kSize, numChannels, 
        units_in_FC, data_format, number_of_b, sparsity, shared_weights):
    """
    depth: how many blocks to use in resnet.
    nClass: how many classes in the output layer
    kSize: convolution size in resnet.
    numChannels: how many filters to use in the first conv layer.
                 i.e. number of input channels to the blocks chain.
    units_in_FC: number of units in the first fully connected layer.
    data_format: 'channels_first' or 'channels_last'
    number_of_b: number of binary filters. i.e. the filters used in RBC
    sparsity/shared_weights: params used in LBC
    returns a model function that takes inputs and is_training and compute the output
    tensor.
    """
    nChIn = numChannels
    nChTmp = number_of_b
    # after a 5x5 non-overlapping average pooling, the cifar10 image origin
    # size is 32x32, and now is only 6x6 left.
    shape_after_avg = 6 * 6
    
    if data_format is None:
        data_format = (
                'channels_first' if tf.test.is_built_with_cuda() else 'channels_last')

    def model(inputs, is_training):
        """
        Constructs the ResNet model given the inputs.
        """
        if data_format == 'channels_first':
            inputs = tf.transpose(inputs, [0, 3, 1, 2])

        inputs = tf.layers.conv2d(inputs = inputs, filters = nChIn,
                kernel_size = [kSize, kSize], strides = (1, 1),
                padding = 'SAME', data_format = data_format)
        # not necessary to add batch normalization, since each basic block hase bn
        # inputs = batch_norm_relu(inputs, is_training, data_format)
        
        for i in range(depth):
            block_name = 'LBC_residual_block_' + str(i)
            inputs = basic_block_LBC(inputs, nChIn, nChTmp, kSize, is_training,
                    data_format = data_format, sparsity = sparsity,
                    shared_weights = shared_weights, block_name = block_name)

        inputs = batch_norm_relu(inputs, is_training, data_format)
        inputs = tf.layers.average_pooling2d(inputs, pool_size = 5,
                strides = 5, padding = 'valid', data_format = data_format)
        inputs = tf.identity(inputs, name = 'final_avg_pool')
        
        # a two layer FC network with dropout while training.
        inputs = tf.reshape(inputs, [-1, numChannels * shape_after_avg])
        inputs = tf.layers.dropout(inputs, training = is_training)
        inputs = tf.layers.dense(inputs, units = units_in_FC, activation = tf.nn.relu)
        inputs = tf.layers.dropout(inputs, training = is_training)
        inputs = tf.layers.dense(inputs, units = nClass)
        inputs = tf.identity(inputs, name = 'final_dense_out')
        return inputs

    return model

# --- argument parser ----
parser = argparse.ArgumentParser()

parser.add_argument('--train_data_dir', type = str, default = '../data/cifar10_train.tfrecords',
                    help='The directory to the stored cifar10 train data in tfrecords form.')
                    
parser.add_argument('--test_data_dir', type = str, default = '../data/cifar10_test.tfrecords',
                    help='The directory to the stored cifar10 test data in tfrecords form.')
                    
parser.add_argument('--summaries_dir', type = str, default = '../results/summaries/',
                    help='The directory to write summaries')

parser.add_argument('--model_dir', type = str, default = '../results/trained_models/',
                    help='The directory to write summaries')

parser.add_argument('--nEpochs', type=int, default=1,
                    help='# of total epochs to run')

parser.add_argument('--batch_size', type=int, default=128,
                    help='mini-batch size (1 = pure stochastic)')

# optimization option
parser.add_argument('--LR', type=float, default=1e-4,
                    help='initial learning rate')

parser.add_argument('--weightDecay', type=float, default=1e-4, 
                    help='weight decay')

# model option
parser.add_argument('--nClass', type=int, default=10,
                    help='number of classes in the output layer')

parser.add_argument('--depth', type=int, default=1,
                    help='ResNet depth: 18 | 34 | 50 | 101 | ...number')

parser.add_argument('--shared_weights', type=str2bool,
                    default=False, help='share weight or Not')

parser.add_argument('--stride', type=int, default=1,
                    help='Striding for Convolution, equivalent to pooling')

parser.add_argument('--sparsity', type=float, default=0.9,
                    help='Percentage of sparsity in pre-defined LB filters')

parser.add_argument('--nInputPlane', type=int, default=3,
                    help='number of input channels')

parser.add_argument('--numChannels', type=int, default=3,
                    help='number of intermediate channels')

parser.add_argument('--full', type=int, default=20,
                    help='number of hidden units in FC')

parser.add_argument('--number_of_b', type=int, default=10,
                    help='number of fixed binary weights')

parser.add_argument('--convSize', type=int, default=3,
                    help='LB convolutional filter size')

parser.add_argument('--data_format', type=str, default='channels_last', 
                    help='either channels_last or channels_first')

parser.add_argument('--momentum', type=float, default=0.9,
                    help='momentum in optimizer')

opt, unparsed = parser.parse_known_args()

path2TrainData = opt.train_data_dir
path2testData = opt.test_data_dir


_image_width = 32
_image_height = 32
_channels = opt.nInputPlane
_train_dataset_size = 50000
_test_dataset_size = 10000
# _WEIGHT_DECAY = 2e-4
_WEIGHT_DECAY = opt.weightDecay
_momentum = opt.momentum

# _BATCH_NORM_DECAY = 0.997
# _BATCH_NORM_EPSILON = 1e-5

# google cloud can't use unpickle directly. 
# instead, let's use tfrecords!
# def unpickle(file):
#     import pickle
#     with open(file, 'rb') as fo:
#         dict = pickle.load(fo, encoding='bytes')
#     return dict
# 
# def shuffleDataSet(images, labels):
#     assert len(images) == len(labels)
#     p = np.random.permutation(len(images))
#     return images[p], labels[p]
# 
# print('extracting data from: {}cifar-10-batches-py/'.format(path2Data))
# # unpacking training and test data
# b1 = unpickle(path2Data + 'cifar-10-batches-py/data_batch_1')
# b2 = unpickle(path2Data + 'cifar-10-batches-py/data_batch_2')
# b3 = unpickle(path2Data + 'cifar-10-batches-py/data_batch_3')
# b4 = unpickle(path2Data + 'cifar-10-batches-py/data_batch_4')
# b5 = unpickle(path2Data + 'cifar-10-batches-py/data_batch_5')
# 
# test = unpickle(path2Data + 'cifar-10-batches-py/test_batch')
# for key, _ in test.items():
#     print(repr(key), type(key))
# for key, _ in b1.items():
#     print(repr(key), type(key))
# 
# # preparing test data
# test_data = test[b'data']
# test_label = test[b'labels']
# 
# # preparing training data
# train_data = np.concatenate([b1[b'data'],b2[b'data'],b3[b'data'],b4[b'data'],b5[b'data']],axis=0)
# train_label = np.concatenate([b1[b'labels'],b2[b'labels'],b3[b'labels'],b4[b'labels'],b5[b'labels']],axis=0)
# 
# #Reshaping data
# train_data = np.reshape(train_data, newshape = 
#     [-1, _channels, _image_height, _image_width])
# test_data = np.reshape(test_data, newshape = 
#     [-1, _channels, _image_height, _image_width])
# train_data = np.array(train_data, dtype=float) / 255.0
# test_data = np.array(test_data, dtype=float) /255.0
# # reshape the data format to NHWC. channel_last!!!.
# train_data = train_data.transpose([0, 2, 3, 1])
# test_data = test_data.transpose([0, 2, 3, 1])
# =========== end of data generating ===========

# fname = path2Data
def _parse_function(serialized_example):
  features = tf.parse_single_example(
      serialized_example,
      features={
          'image_raw': tf.FixedLenFeature([], tf.string),
          'label': tf.FixedLenFeature([], tf.int64),
      })

  image = tf.decode_raw(features['image_raw'], tf.uint8)
  image = tf.reshape(image, shape = [_image_height, _image_width, _channels])
  image = tf.cast(image, tf.float32) * (1. / 255)
  label = tf.cast(features['label'], tf.float32)

  return image, label
  
def my_input_fn(filename, is_training, batch_size, nEpoch = 1):
    # # unfortunatly, google tf.data was introduced in r1.4.
    # # yet google cloud runs on r1.2
    # dataset = tf.data.TFRecordDataset(filename)
    dataset = tf.contrib.data.TFRecordDataset(filename)
    dataset = dataset.map(_parse_function)
    if is_training:
        dataset = dataset.shuffle(buffer_size = 256)
    dataset = dataset.repeat(nEpoch)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()


print('depth of resnet = {}'.format(opt.depth))
# network = resnet_LBC.cifar10_resnet_vanilla_generator(depth = opt.depth,
#         nClass = opt.nClass, kSize = opt.convSize, numChannels = opt.numChannels,
#         units_in_FC = opt.full, data_format = opt.data_format,
#         number_of_b = opt.number_of_b, sparsity = opt.sparsity,
#         shared_weights = opt.shared_weights)

network = cifar10_resnet_LBC_generator(depth = opt.depth,
        nClass = opt.nClass, kSize = opt.convSize, numChannels = opt.numChannels,
        units_in_FC = opt.full, data_format = opt.data_format,
        number_of_b = opt.number_of_b, sparsity = opt.sparsity,
        shared_weights = opt.shared_weights)

# reg = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])


# my_input_fn will return a pair of tensor, (images, labels), so no need for placeholder.
train_batch_images, train_batch_labels = my_input_fn(filename = path2TrainData, 
        is_training = True, batch_size = opt.batch_size, nEpoch = opt.nEpochs)
# train_one_hot_labels = tf.one_hot(tf.cast(train_batch_labels, tf.int32), depth = opt.nClass)
# Build the INFERENCE net.
# for test dataset, no need to shuffle, but still need to run #epoch times.
test_batch_images, test_batch_labels = my_input_fn(filename = path2testData, 
        is_training = False, batch_size = opt.batch_size, nEpoch = opt.nEpochs)
# test_one_hot_labels = tf.one_hot(tf.cast(test_batch_labels, tf.int32), depth = opt.nClass)

# construct the graph
images = tf.placeholder(tf.float32, shape = [None, _image_height, _image_width, _channels])
labels = tf.placeholder(tf.float32, shape = [None])
one_hot_labels = tf.one_hot(tf.cast(labels, tf.int32), depth = opt.nClass)

# the left two parameters are rate and training flag(pass into the LBC module)
learning_rate = tf.placeholder(tf.float32, shape = [])
training_rate = opt.LR
is_training = tf.placeholder(tf.bool, shape = [], name = 'training_flag')
# we can probably add drop-out rate here. As a placeholder.
# # ----- why not just gen the data while training and feed them?

logits = network(inputs = images, is_training = is_training)
cross_entropy = tf.losses.softmax_cross_entropy(one_hot_labels, logits)
summray_train_xentropy = tf.summary.scalar('cross_entropy', cross_entropy)

loss = cross_entropy + _WEIGHT_DECAY * tf.add_n(
        [tf.nn.l2_loss(v) for v in tf.trainable_variables()])
summary_train_loss = tf.summary.scalar('loss', loss)

optimizer = tf.train.MomentumOptimizer(learning_rate = learning_rate, momentum = _momentum)

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_op = optimizer.minimize(loss)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
summary_train_acc = tf.summary.scalar('accuracy', accuracy)

# --- the code below is to help test accuracy --- 
test_accuracy_summary = tf.placeholder(dtype = tf.float32, shape = [])
summary_test_acc = tf.summary.scalar('test_accuracy', test_accuracy_summary)
test_loss_summary = tf.placeholder(dtype = tf.float32, shape = [])
summary_test_loss = tf.summary.scalar('test_loss', test_loss_summary)

# # ---------------
# # compute the output of a graph and see the loss/accuracy in the TRAINING NET!
# train_logits = network(train_batch_images, is_training = is_training)
# train_cross_entropy = tf.losses.softmax_cross_entropy(train_one_hot_labels, train_logits)
# 
# train_loss = train_cross_entropy + _WEIGHT_DECAY * tf.add_n(
#    [tf.nn.l2_loss(v) for v in tf.trainable_variables()])
# # train_loss = train_cross_entropy + _WEIGHT_DECAY * reg
# 
# optimizer = tf.train.MomentumOptimizer(learning_rate = learning_rate, momentum = _momentum)
# update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
# with tf.control_dependencies(update_ops):
#     train_op = optimizer.minimize(train_loss)
# 
# train_correct_prediction = tf.equal(tf.argmax(train_logits, 1), tf.argmax(train_one_hot_labels, 1))
# train_accuracy = tf.reduce_mean(tf.cast(train_correct_prediction, tf.float32))
# 
# # --------------
# 
# # can't do this. need to save the training model and test on the loaded model.
# test_logits = network(inputs = test_batch_images, is_training = is_training)
# test_cross_entropy = tf.losses.softmax_cross_entropy(test_one_hot_labels, test_logits)
# 
# # test_loss = test_cross_entropy + _WEIGHT_DECAY * reg
# test_loss = test_cross_entropy
# test_correct_prediction = tf.equal(tf.argmax(test_logits, 1), tf.argmax(test_one_hot_labels, 1))
# test_accuracy = tf.reduce_mean(tf.cast(test_correct_prediction, tf.float32))


with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    saver = tf.train.Saver()
    merged_train = tf.summary.merge(
            inputs = [summary_train_loss, summray_train_xentropy, summary_train_acc])
    merged_test = tf.summary.merge(
            inputs = [summary_test_loss, summary_test_acc])
    train_writer = tf.summary.FileWriter(opt.summaries_dir + 'train', sess.graph)
    test_writer = tf.summary.FileWriter(opt.summaries_dir + 'test')
    total_step = opt.nEpochs * _train_dataset_size // opt.batch_size
    for epoch in range(opt.nEpochs):
        # Now that we use the my_input_fn function. no need to sample
        # # shuffle the data set
        # train_data, train_label = shuffleDataSet(train_data, train_label)
        for iter in range(_train_dataset_size // opt.batch_size):
            # decrease the learning rate in a naive way.
            step = epoch * (_train_dataset_size//opt.batch_size) + iter
            if step in [total_step//4, total_step//2, total_step * 3//4, total_step * 7 // 8]:
                training_rate *= 0.1
            # sample batchs from training data.
            # images_batch = train_data[iter : iter + opt.batch_size]
            # labels_batch = train_label[iter : iter + opt.batch_size]
            images_batch, labels_batch = sess.run(
                    [train_batch_images, train_batch_labels])
            feed_dict_1 = {images : images_batch,
                         labels : labels_batch,
                         learning_rate : training_rate,
                         is_training : True}
            # sess.run(train_op, feed_dict = feed_dict_1)
            _, train_loss_w_bn, train_xentro_w_bn, train_acc_w_bn, summary = sess.run(
                    [train_op, loss, cross_entropy, accuracy, merged_train], feed_dict = feed_dict_1)
            train_writer.add_summary(summary, step)
            if step%5000 == 0:
                path = opt.model_dir + repr(step) + '.ckpt'
                saver.save(sess, path)
            # feed_dict_2 = {images : images_batch,
            #              labels : labels_batch,
            #              learning_rate : training_rate,
            #              is_training : False}
            # train_loss_wo_bn, train_xentro_wo_bn, train_acc_wo_bn = sess.run(
            #         [loss, cross_entropy, accuracy], feed_dict = feed_dict_2)
            
            # # IMPORTANT!! NOW WE CAN ONLY sess.run ONCE!!!!  
            # feed_dict_1 = {learning_rate : training_rate,
            #                is_training   : True}
            # _, train_loss_w_bn, train_xentropy_w_bn, train_acc_w_bn = sess.run(
            #         [train_op, train_loss, train_cross_entropy, train_accuracy],
            #         feed_dict = feed_dict_1)
            if iter%50 == 0:
                print('learning_rate = {}'.format(training_rate))

                print('step {}, with batch norm training accuracy = {}, training loss = {}, cross_entropy = {}'.format(
                    step, train_acc_w_bn, train_loss_w_bn, train_xentro_w_bn))

                # print('step {}, without batch norm training loss = {}, cross_entropy = {}, training accuracy = {}'.format(
                #     step, train_loss_wo_bn, train_xentro_wo_bn, train_acc_wo_bn))

                print('----')
        # do evaluation every epoch.
        eval_loss = 0
        eval_acc = 0
        for i in range(_test_dataset_size//opt.batch_size):
            # eval_images = test_data
            # eval_labels = test_label
            # test_dict = {images : eval_images[i: i + opt.batch_size],
            #              labels : eval_labels[i: i + opt.batch_size],
            #              is_training: False}
            eval_images, eval_labels = sess.run([test_batch_images, test_batch_labels])
            test_dict = {images : eval_images,
                         labels : eval_labels,
                         is_training: False}
            test_batch_loss, test_batch_acc = sess.run(
                    [loss, accuracy], feed_dict = test_dict)
            # test_batch_loss, test_batch_acc, test_batch_xentropy = sess.run(
            #         [test_loss, test_accuracy, test_cross_entropy],
            #         feed_dict = {is_training : False})
            eval_loss += test_batch_loss
            eval_acc += test_batch_acc
        eval_loss = eval_loss/(_test_dataset_size//opt.batch_size)
        eval_acc = eval_acc/(_test_dataset_size//opt.batch_size)

        test_summary_dict = {test_loss_summary: eval_loss,
                             test_accuracy_summary: eval_acc}
        summary = sess.run(merged_test, feed_dict = test_summary_dict)
        # _, _, summary = sess.run([test_loss_summary, test_accuracy_summary, merged_test],
        #         feed_dict = test_summary_dict)
        test_writer.add_summary(summary, global_step = epoch)
        
        print('>'*20)
        print(' | epoch# {} | evaluation loss = {} | accuracy = {} |'.format(
            epoch, eval_loss, eval_acc))
        print('<'*20)
        print('----')


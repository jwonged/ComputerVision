'''
Created on 25 Feb 2018

@author: jwong
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

import tensorflow as tf
import numpy as np
class CNNModel(object):

    def __init__(self, config, trainData, trainlabels, eval_data, eval_labels):
        self.config = config
        self.sess = None
        self.saver = None
        self.trainData = trainData
        self.trainlabels = trainlabels
        self.testData = eval_data
        self.testlabels =  eval_labels
    
    def constructModel(self):
        with tf.name_scope('input'):
            self.x = tf.placeholder(tf.float32, [None, 784], name='img-input')
            self.labels = tf.placeholder(tf.int64, [None], name='label-input')
            
        with tf.name_scope('input_reshape'):
            shaped_x = tf.reshape(self.x, [-1, 28, 28, 1])
            tf.summary.image('input', shaped_x, 10)
        
        with tf.variable_scope('conv1'):
            #cross correlation
            conv1 = tf.layers.conv2d(inputs=shaped_x, 
                                     filters=32,
                                     kernel_size=[5, 5],
                                     padding="same",
                                     activation=tf.nn.relu)
            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
        
        with tf.variable_scope('conv2'):
            conv2 = tf.layers.conv2d(inputs=pool1,
                                    filters=64,
                                    kernel_size=[5, 5],
                                    padding="same",
                                    activation=tf.nn.relu)
            pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
            
        #flatten
        pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
            
        with tf.variable_scope('fc_layers'):
            fcLayer1 = tf.layers.dense(inputs=pool2_flat, 
                                          units=1024, 
                                          activation=tf.nn.relu,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer())
            if (self.config.dropoutVal < 1.0):
                fcLayer1 = tf.nn.dropout(fcLayer1, self.config.dropoutVal)
            self.y = tf.layers.dense(inputs=fcLayer1, units=10)
            
        with tf.variable_scope('loss'):
            crossEntropyLoss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                        logits=self.y, labels=self.labels)
            self.loss = tf.reduce_mean(crossEntropyLoss)
        tf.summary.scalar('cross_entropy', self.loss)
        
        self._addOptimizer()
        
        with tf.variable_scope('Pred'):
            correct_prediction = tf.equal(tf.argmax(self.y, 1), self.labels)
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('Pred', self.accuracy)
        
        self._initSession()
            
    def _addOptimizer(self):
        with tf.variable_scope("train_step"):
            if self.config.modelOptimizer == 'adam': 
                print('Using adam optimizer')
                optimizer = tf.train.AdamOptimizer(self.config.lossRate)
            elif self.config.modelOptimizer == 'adagrad':
                print('Using adagrad optimizer')
                optimizer = tf.train.AdagradOptimizer(self.config.lossRate)
            else:
                print('Using grad desc optimizer')
                optimizer = tf.train.GradientDescentOptimizer(self.config.lossRate)
        
        self.train_op = optimizer.minimize(self.loss, name='trainModel')
        
    def _initSession(self):
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        self.merged = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter('./train', self.sess.graph)
        self.test_writer = tf.summary.FileWriter('./test')
        
        print('Initializing model...')
    
    def train(self):
        for i in range(self.config.nEpoch):
            acc = self.run_epoch()
            print('Epoch {} accuracy: {}'.format(i,acc))
        
    def run_epoch(self):
        print('Starting training')
        for index, (x,labels) in enumerate(self._getNextBatch(self.config.batchSize)):
            if (index % 10 == 0):
                summary, acc = self.sess.run(
                    [self.merged, self.accuracy], 
                    feed_dict={self.x : self.testData, self.labels : self.testlabels})
                self.test_writer.add_summary(summary, index)
                print('Accuracy at batch {} : {:>6.1%}'.format(index, acc))
            _, summary, _ = self.sess.run(
                [self.y, self.merged, self.train_op], 
                feed_dict={self.x : x, self.labels : labels})
            self.train_writer.add_summary(summary, index)
            
        summary, acc = self.sess.run(
                    [self.merged, self.accuracy], 
                    feed_dict={self.x : self.testData, self.labels : self.testlabels})
        self.test_writer.add_summary(summary, index)
        return acc
            
    def _getNextBatch(self, batchSize):
        start = 0
        end = start + batchSize
        while (end < len(self.trainData)):
            yield self.trainData[start:end], self.trainlabels[start:end]
            start += batchSize
            end += batchSize
        
        if (start < len(self.trainData)):
            yield self.trainData[start:], self.trainlabels[start:]
        
            
    

class Config():
    def __init__(self):
        pass
    
    dropoutVal = 0.5
    lossRate = 0.001
    batchSize = 100
    nTrainData = 55000
    modelOptimizer = 'adam'
    nEpoch = 35
    
    
if __name__ == '__main__':
    # Load training and eval data
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images  # Returns np.array
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images  # Returns np.array
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
    
    config = Config()
    model = CNNModel(config, train_data, train_labels, eval_data, eval_labels)
    model.constructModel()    
    model.train()
    
    
    
    
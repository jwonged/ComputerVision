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

class Config():
    #99.3% accuracy
    batchSize = 100
    nTrainData = 55000
    nEpoch = 40
    nEpochsWithoutImprov = 4
    
    dropoutVal = 0.5
    lossRate = 0.001
    modelOptimizer = 'adam'
    lossRateDecay = 0.95
    trainValSplit = 0.9
    
    saveModelFile = './yolo'
    restoreModelPath = './'
    restoreModel = './yolo.meta'

class CNNModel(object):
    def __init__(self, config):
        self.config = config
        self.sess = None
        self.saver = None
    
    def constructModel(self):
        self.x = tf.placeholder(tf.float32, [None, 784], name='img-input')
        self.labels = tf.placeholder(tf.int64, [None], name='label-input')
        self.lr = tf.placeholder(dtype=tf.float32, shape=[], name="lr")
        self.dropout = tf.placeholder(dtype=tf.float32, shape=[], name="dropout")
            
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
                fcLayer1 = tf.nn.dropout(fcLayer1, self.dropout)
            self.y = tf.layers.dense(inputs=fcLayer1, units=10)
            
        with tf.variable_scope('loss'):
            crossEntropyLoss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                        logits=self.y, labels=self.labels)
            self.loss = tf.reduce_mean(crossEntropyLoss)
        tf.summary.scalar('cross_entropy', self.loss)
        
        self._addOptimizer()
        
        with tf.variable_scope('Pred'):
            correct_prediction = tf.equal(tf.argmax(self.y, 1), self.labels)
            self.accuracy = tf.reduce_mean(
                tf.cast(correct_prediction, tf.float32), name='accuracy')
        tf.summary.scalar('Pred', self.accuracy)
        
        self._initSession()
            
    def _addOptimizer(self):
        with tf.variable_scope("train_step"):
            if self.config.modelOptimizer == 'adam': 
                print('Using adam optimizer')
                optimizer = tf.train.AdamOptimizer(self.lr)
            elif self.config.modelOptimizer == 'adagrad':
                print('Using adagrad optimizer')
                optimizer = tf.train.AdagradOptimizer(self.lr)
            else:
                print('Using grad desc optimizer')
                optimizer = tf.train.GradientDescentOptimizer(self.lr)
        
        self.train_op = optimizer.minimize(self.loss, name='trainModel')
        
    def _initSession(self):
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        self.merged = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter('./train', self.sess.graph)
        self.test_writer = tf.summary.FileWriter('./test')
        
        print('Initializing model...')
    
    def train(self, trainData, trainlabels):
        print('Starting training')
        splitIndex = int(round(len(trainData)*self.config.trainValSplit))
        self.trainData = trainData[:splitIndex]
        self.trainlabels = trainlabels[:splitIndex]
        self.valData = trainData[splitIndex:]
        self.vallabels = trainlabels[splitIndex:]
        
        highestScore = 0
        nEpochWithoutImprovement = 0
        
        
        for nEpoch in range(self.config.nEpoch):
            score = self.run_epoch()
            print('Epoch {} accuracy: {:>6.1%}'.format(nEpoch,score))
            
            self.config.lossRate *= self.config.lossRateDecay
            
            if (score >= highestScore):
                nEpochWithoutImprovement = 0
                self.saver.save(self.sess, self.config.saveModelFile)
                highestScore = score
            else:
                nEpochWithoutImprovement += 1
                if nEpochWithoutImprovement >= self.config.nEpochsWithoutImprov:
                    print('Early stopping at epoch {} with {} epochs without improvement'.format(
                            nEpoch+1, nEpochWithoutImprovement))
                    break
        
        self.runPredict()
        
    def run_epoch(self):
        for index, (x,labels) in enumerate(self._getNextBatch(self.config.batchSize)):
            if (index % 100 == 0):
                summary, acc = self.sess.run(
                    [self.merged, self.accuracy], 
                    feed_dict={self.x : self.valData, 
                               self.labels : self.vallabels,
                               self.dropout : self.config.dropoutVal,
                               self.lr : self.config.lossRate})
                self.test_writer.add_summary(summary, index)
                print('Accuracy at batch {} : {:>6.1%}'.format(index, acc))
            _, summary, _ = self.sess.run(
                [self.y, self.merged, self.train_op], 
                feed_dict={self.x : x, 
                           self.labels : labels,
                           self.dropout : self.config.dropoutVal,
                           self.lr : self.config.lossRate})
            self.train_writer.add_summary(summary, index)
            
        summary, acc = self.sess.run(
                    [self.merged, self.accuracy], 
                    feed_dict={self.x : self.valData, 
                               self.labels : self.vallabels,
                               self.dropout : 1.0})
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
        
    def runPredict(self):
        #restore highest scoring model
        self.restoreModel()
        acc = self.sess.run([self.accuracy], 
                            feed_dict={self.x : self.testData, 
                                       self.labels : self.testlabels,
                                       self.dropout : 1.0})
        print('Model test accuracy: {}'.format(acc))
    
    def restoreModel(self):
        tf.reset_default_graph
        print('restoring model from {}'.format(self.config.restoreModel))
        self.sess = tf.Session()
        self.saver = tf.train.import_meta_graph('./yolo.meta')
        self.saver.restore(self.sess, tf.train.latest_checkpoint('./'))
        
        graph = tf.get_default_graph()
        self.dropout = graph.get_tensor_by_name('dropout:0')
        self.x = graph.get_tensor_by_name('img-input:0')
        self.labels = graph.get_tensor_by_name('label-input:0')
        self.accuracy = graph.get_tensor_by_name('Pred/accuracy:0')
        self.saver = tf.train.Saver()
        

def main():
    # Load training and eval data
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images  # Returns np.array
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images  # Returns np.array
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
    
    config = Config()
    model = CNNModel(config)
    model.constructModel()    
    model.train(train_data, train_labels)
    
    predictModel = CNNModel(config)
    predictModel.runPredict(eval_data, eval_labels)
    
if __name__ == '__main__':
    main()
    
    
    
    
    
    
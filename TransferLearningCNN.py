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
import caffe

class CNNTransferModel(object):
    '''
    Attempting transfer learning on mnist dataset
    With VGGnet pre-trained on imagenet
    '''
    def __init__(self, config, trainData, trainlabels, eval_data, eval_labels):
        self.config = config
        self.sess = None
        self.saver = None
        splitIndex = int(round(len(trainData)*self.config.trainValSplit))
        self.trainData = self._preprocessImages(trainData[:splitIndex])
        self.trainlabels = trainlabels[:splitIndex]
        self.valData = self._preprocessImages(trainData[splitIndex:])
        self.vallabels = trainlabels[splitIndex:]
        self.testData = self._preprocessImages(eval_data)
        self.testlabels =  eval_labels
    
    def constructModel(self):
        with tf.name_scope('input'):
            self.x = tf.placeholder(tf.float32, [None, 784], name='img-input')
            self.labels = tf.placeholder(tf.int64, [None], name='label-input')
            self.lr = tf.placeholder(dtype=tf.float32, shape=[], name="lr")
            self.dropout = tf.placeholder(dtype=tf.float32, shape=[], name="dropout")
            
        with tf.variable_scope('fc_layers'):
            fcLayer1 = tf.layers.dense(inputs=self.x, 
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
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
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
    
    def train(self):
        print('Starting training')
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
    
    def _preprocessImages(self, images):
        sys.path.insert(0, self.config.caffe_root + 'python')
        caffe.set_mode_gpu()
        net = caffe.Classifier(self.config.model_prototxt, self.config.model_trained,
                           mean=np.load(self.config.mean_path).mean(1).mean(1), #channel_swap=(2,1,0),
                           raw_scale=255,
                           image_dims=(28, 28))
        
        # Loading class labels
        with open(self.config.imagenet_labels) as f:
            labels = f.readlines()
            
        result = []
        count = 0
        print(np.asarray(images).shape)
        for imagevec in images:
            vec = [[imagevec], [imagevec], [imagevec]]
            print(np.asarray(vec).shape)
            prediction = net.predict([imagevec], oversample=False)
            msg = ('image {} : {} ( {} )'.format(count,
                                                 labels[prediction[0].argmax()].strip(), 
                                                 prediction[0][prediction[0].argmax()]))
            count = count + 1
            featureData = net.blobs[self.config.layer_name].data[0].reshape(1,-1).tolist()
            result.append(featureData)
            print(featureData)
            print(np.asarray(featureData).shape)
            if (count == 1):
                break
        
        
    def run_epoch(self):
        for index, (x,labels) in enumerate(self._getNextBatch(self.config.batchSize)):
            if (index % 10 == 0):
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
        print('Number of training data points: {}'.format(len(self.trainData)))
        print('Number of val data points: {}'.format(len(self.valData)))
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
        summary, acc = self.sess.run(
                    [self.merged, self.accuracy], 
                    feed_dict={self.x : self.testData, 
                               self.labels : self.testlabels,
                               self.dropout : 1.0})
        print('Model test accuracy: {}'.format(acc))
        self.test_writer.add_summary(summary)
    
    def restoreModel(self):
        print('restoring model from {}'.format(self.config.restoreModel))
        self.sess = tf.Session()
        self.saver = saver = tf.train.import_meta_graph(self.config.restoreModel)
        saver.restore(self.sess, tf.train.latest_checkpoint(self.config.restoreModelPath))
        
        graph = tf.get_default_graph()
        self.accuracy = graph.get_tensor_by_name('accuracy:0')
        self.x = graph.get_tensor_by_name('img-input:0')
        self.labels = graph.get_tensor_by_name('label-input:0')
        self.dropout = graph.get_tensor_by_name('dropout:0')
        self.saver = tf.train.Saver()
        
        
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
    
    caffe_root = '/home/joshua/caffe/'
    model_prototxt = caffe_root + 'models/211839e770f7b538e2d8/VGG_ILSVRC_19_layers_deploy.prototxt'
    model_trained = caffe_root + 'models/211839e770f7b538e2d8/VGG_ILSVRC_19_layers.caffemodel'
    imagenet_labels = caffe_root + 'data/ilsvrc12/synset_words.txt'
    mean_path = caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy'
    layer_name = 'fc7'
    
def main():
    # Load training and eval data
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images  # Returns np.array
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images  # Returns np.array
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
    
    config = Config()
    model = CNNTransferModel(config, train_data, train_labels, eval_data, eval_labels)
    
if __name__ == '__main__':
    main()
    
    
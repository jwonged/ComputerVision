'''
Created on 25 Feb 2018

@author: jwong
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import tensorflow as tf
import numpy as np
import caffe
import pickle

class CNNTransferModel(object):
    '''
    Attempting transfer learning on mnist dataset
    With VGGnet pre-trained on imagenet
    '''
    def __init__(self, config):
        self.config = config
        self.sess = None
        self.saver = None
    
    def constructModel(self):
        with tf.name_scope('input'):
            self.x = tf.placeholder(tf.float32, [None, 4096], name='img-input')
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
        print('Model constructed.')
            
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
    
    def train(self, trainData, trainlabels, testData, testlabels, load=False):
        if load:
            print('Reading ' + self.config.preprocessedFile)
            with open(self.config.preprocessedFile, 'rb') as f:
                data = pickle.load(f)
            self.trainData = data['trainData']
            self.trainlabels = data['trainlabels']
            self.valData = data['valData']
            self.vallabels = data['vallabels']
        else:
            print('Preprocessing images...')
            splitIndex = int(round(len(trainData)*self.config.trainValSplit))
            
            self.trainData = self._preprocessImages(trainData[:splitIndex])
            print('Feature extraction for training data completed.')
            self.trainlabels = trainlabels[:splitIndex]
            
            self.valData = self._preprocessImages(trainData[splitIndex:])
            print('Feature extraction for training data completed.')
            self.vallabels = trainlabels[splitIndex:]
            
            with open(self.config.preprocessedFile, 'wb') as f:
                data = {}
                data['trainData'] = self.trainData
                data['trainlabels'] = self.trainlabels
                data['valData'] = self.valData
                data['vallabels'] = self.vallabels 
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
                print('Saved to {}'.format(self.config.preprocessedFile))
        
        self.constructModel()
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
        
        self.runPredict(testData, testlabels, restore=False)
    
    def _preprocessImages(self, images):
        #Preprocess images into 4096 dimension vectors
        sys.path.insert(0, self.config.caffe_root + 'python')
        caffe.set_mode_gpu()
        net = caffe.Classifier(self.config.model_prototxt, self.config.model_trained, 
                               mean=np.load(self.config.mean_path).mean(1).mean(1), 
                           channel_swap=(2,1,0),
                           raw_scale=255,
                           image_dims=(224, 224))
        
        # Loading class labels
        with open(self.config.imagenet_labels) as f:
            labels = f.readlines()
            
        result = []
        count = 0
        for imagevec in images:
            imagevec = np.repeat(imagevec,3)
            imagevec = np.reshape(imagevec, [28,28,3])
            print(np.asarray(imagevec).shape)
            #print(np.asarray(input_image).shape)
            prediction = net.predict([imagevec], oversample=False)
            
            msg = ('image {} : {} ( {} )'.format(count,
                                                 labels[prediction[0].argmax()].strip(), 
                                                 prediction[0][prediction[0].argmax()]))
            print(msg)
            count = count + 1
            featureData = net.blobs[self.config.layer_name].data[0].reshape(1,-1).tolist()[0]
            print(np.asarray(featureData).shape)
            result.append(featureData)
        
        return result
        
        
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
        
    def runPredict(self, testData, testlabels, restore=False):
        testData = self._preprocessImages(testData)
        #restore highest scoring model
        if restore:
            self.restoreModel()
        acc = self.sess.run(self.accuracy, 
                            feed_dict={self.x : testData, 
                                       self.labels : testlabels,
                                       self.dropout : 1.0})
        print('Model test accuracy: {:>6.1%}'.format(acc))
    
    def restoreModel(self):
        print('restoring model from {}'.format(self.config.restoreModel))
        self.sess = tf.Session()
        self.saver = tf.train.import_meta_graph('./yolo.meta')
        self.saver.restore(self.sess, tf.train.latest_checkpoint('./'))
        
        graph = tf.get_default_graph()
        tf.reset_default_graph
        self.dropout = graph.get_tensor_by_name('dropout:0')
        self.x = graph.get_tensor_by_name('img-input:0')
        self.labels = graph.get_tensor_by_name('label-input:0')
        self.accuracy = graph.get_tensor_by_name('Pred/accuracy:0')
        self.saver = tf.train.Saver()
        
        
class Config():
    batchSize = 10
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
    preprocessedFile = './preprocessedDataStuff.pkl'
    
    caffe_root = '/home/joshua/caffe/'
    model_prototxt = caffe_root + 'models/211839e770f7b538e2d8/VGG_ILSVRC_19_layers_deploy.prototxt'
    model_trained = caffe_root + 'models/211839e770f7b538e2d8/VGG_ILSVRC_19_layers.caffemodel'
    imagenet_labels = caffe_root + 'data/ilsvrc12/synset_words.txt'
    mean_path = caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy'
    layer_name = 'fc7'

def main(load):
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images  
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images 
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
    
    config = Config()
    model = CNNTransferModel(config)
    model.train(train_data, train_labels, eval_data, eval_labels, load)

def predict():
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    eval_data = mnist.test.images 
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
    config = Config()
    model = CNNTransferModel(config)
    model.runPredict(eval_data, eval_labels, restore=True)
    
    
if __name__ == '__main__':
    if (len(sys.argv) < 1):
        print("Run with: python {} <option>\n <option>: '-train' or '-pred' or 'loadtrain'".format(
            sys.argv[0]))
        
    if (sys.argv[1] == '-pred'):
        predict()
    elif (sys.argv[1] == '-train'):
        main(False)
    elif (sys.argv[1] == '-loadtrain'):
        main(True)
    else:
        print('Run options: -pred or -train or -loadtrain')
    
    
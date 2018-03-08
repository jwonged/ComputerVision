# ComputerVision
Experimenting with CNN configurations on MNIST (Computer Vision Supervision 3)

### CNN_configurable.py
Main 4 Conv layer CNN with 2 fully-connected layers. Run with
```
python CNN_configurable.py -train
```
to train the network. This saves the best performing model to a meta file in the same directory.   
Quick configurations can be changed in the Config class.   
To run predict on MNIST test set using the saved model after training, use:
```
python CNN_configurable.py -pred
```
Gives 99.52% accuracy.
   
### TransferLearningCNN.py
Using VGGNet's CNN architecture pre-trained on imagenet. 
Extracting from layer fc7 and attaching 2 fully-connected layers to train on MNIST.
Requires caffe pre-installed. Change caffe_root accordingly in Config class.   
Use
```
python TransferLearningCNN.py -train
```
on the first run. This preprocesses the images into 4096-Dim vectors and 
saves it to a pickle file so the fully-connected layers can be trained separately the next time 
without pre-processing the whole datasete again, and can be loaded for training using:
```
python TransferLearningCNN.py -loadtrain
```
And prediction:
```
python TransferLearningCNN.py -pred
```
Gives ~98.01% accuracy

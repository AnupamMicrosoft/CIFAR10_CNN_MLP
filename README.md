# Image Classification - CNN and MLP
## Problem Description.
In this project I worked to implement fully-connected (MLP) and Convolutional Neural Networks (CNN) image classiﬁer on CIFAR10 dataset. In this task, my goal was to perform multi-class classiﬁcation.  I got a skeleton code for data loading and iterations of training data and then I implemented the rest of training in Pytorch code.

1) understand how to use Pytorch to build multi-class classiﬁers. 
2) understand the mechanism of convolution in image classiﬁcation. 
3) learn the power of non-linearity in modern neural networks. 
4) implement and apply a fully-connected multi-class image classiﬁer. 
5) implement and apply a Convolutional Neural Networks (CNN) classiﬁer

### Sub Tasks
1.Implement a 7 layers fully-connected neural networks with ReLu activation function. Target model accuracy should be around 50% percent. 
2. Then implement a 7 layers convolutional neural networks, 4 convolutional layers and 3 fullyconnected layers, with ReLu activation function. The input dimension of 1st fully-connected layer must be 4096. The model accuracy should be around 85% percent. 
3. In the end describe your 2 model structures including in channels, out channels, stride, kernel size, padding for CNN layer; input dim, out dim for fully connected layer. 
4.  For each of the model, report the (PB b=1PDb d=1 loss(labelb,d,fb(datab,d) Db )/B for each training epoch, where B is the total number of batches, fb is the model after updated by b-th batch and Db is the number of data points in b-th batch. An epoch is deﬁned as one iteration of all dataset. Essentially, during a training epoch, you record down the average training loss of that batch after you update the model, and then report the average of all such batch-averaged losses after one iteration of whole dataset. You could plot the results as a ﬁgure our simply list down. Please at least report 10 epochs. 
5. Report the ﬁnal testing accuracy of trained model. 
6. Compare results for 2 models (MLP and CNN)). 
7. Try neural network without non-linear activation functions and discuss your ﬁndings. 


## Resources
You can follow the setup instructions at https://pytorch.org/get-started/locally/. A useful tutorial on learning building CNN at https://pytorch.org/tutorials/beginner/blitz/cifar10_ tutorial.html. Convolutional functions could be found here: https://pytorch.org/docs/stable/nn.html#convolution-functions.


## Data
I used CIFAR10 classiﬁcation dataset. Pytorch/torchvision has provide a useful dataloader to automatically download and load the data into batches. I got the the same data loader to follow. 



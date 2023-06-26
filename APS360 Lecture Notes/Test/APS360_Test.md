---
title: APS360: Artificial Intelligence Fundamentals
author: Cairo Cristante
---

### Artifical Neuron Connectivity
A nerual network consists of three main parts. The first part are the netowkr inputs into the the system, the second is the hidden layer which creates connections between the input, and the last is the output layer.

<center>
<img src="\images/neural_network.png" width="500">
</center>

Each individual neuron is modelled by the following equation, and are they linked together via some function of conceations.

\[h = f(b+Wx)\]

where, 
- $x =$ a vector of the **neuron activations** at the **input layer**
- $h = $ a vector of the **neuron activations** at the **hidden layer**
- $W = $ a matrix of the **corresponding weights**
- $b = $ a vector of **biases** which represent the activation threshold/requirement of the neuron at the correspending $h$

### Activation Functions
The neural network is defined by some kind of activation function that computes the activation of the neuron based on the toal contribution from the neurons in the layer bellow it. This function **should** be non-linear (idk why lmao). The functions used are typically, the ReLU, Sigmoid, and Tanh functions.

<center>
<img src="\images/ReLU.png" width="300">
<img src="/images/Sigmoid.png" width="300">
<img src="\images/Tanh.png" width="300">
</center>

### Network Architechture
Networks can vary greatly in how they are structured. They are generally defined by the direction in which information flows, the type of connections present, and the number of hidden layers present.

Feed-Forward Networks
: networks in which information only flows form one later to the next layer

Fully-Connected Layer
: Neurons bewteen adjacet layers are fully pairwise connected

Number of Layers
: defined by the amount of hidden layers/sets of weights & biases

### Training Neural Networks
In order to train a neural network, a **loss function**, $L(\text{actual, predicted})$, is defined in order to evaluate the error in the networks reults when compared to the reality of the sample. The network is then trained in order to minimize this loss function, by determining the which neurons are effecting the resulting output. This is done by through optimization of the weights to minimize the loss function. The optomization technique typically used for the neural network application is always done using gradiaent decent. (I believe because newtons method is computationally taxing).

Training Set
: Data sets that are used to tune the neural network parameters

Test Set:
: Data sets that are used to measure the accuracy of the neural network

### Batch Training 




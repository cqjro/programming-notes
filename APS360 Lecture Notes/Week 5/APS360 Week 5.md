---
title: APS360 Week 5
author: Cairo Cristante
header-includes: 
        <script type="text/javascript"
            src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.0/MathJax.js?config=TeX-AMS_CHTML">
        </script>
        <script type="text/x-mathjax-config">
            MathJax.Hub.Config({
                tex2jax: {
                inlineMath: [['$','$'], ['\\(','\\)']],
                processEscapes: true},
                jax: ["input/TeX","input/MathML","input/AsciiMath","output/CommonHTML"],
                extensions: ["tex2jax.js","mml2jax.js","asciimath2jax.js","MathMenu.js","MathZoom.js","AssistiveMML.js", "[Contrib]/a11y/accessibility-menu.js"],
                TeX: {
                extensions: ["AMSmath.js","AMSsymbols.js","noErrors.js","noUndefined.js", "mhchem.js"],
                equationNumbers: {
                autoNumber: "AMS"
                }
                }
            });
        </script>
---

```math
% Math Reference

\newcommand{\topictitle}[1]{\noindent \textbf{\uline{#1}}} % New Topic Titles
\newcommand{\subtopic}[1]{\noindent \uline{#1}}
\newcommand{\supp}[1]{^{\text{#1}}} % faster text superscripts
\newcommand{\chemsub}[1]{_{\ce{#1}}} % chemical formula subscripts
\newcommand{\al}[2]{\text{#1} & \hspace{1cm} #2 &} % faster lines in display equations
\newcommand{\var}[3]{#1= & \hspace{1cm} \text{#2} \hspace{-4cm} & \text{#3} &} % easy display of variable definitions in align format
\newcommand{\note}{\noindent \textbf{Note: }} % making notes/warnings
\newcommand{\laplace}{\mathscr{L}} % laplace operator
\newcommand{\der}{\text{d}} % text derivative sign
\newcommand{\derivative}[2]{\frac{\der #1}{\der #2}}
\newcommand{\perivative}[2]{\frac{\partial #1}{\partial #2}}
\newcommand{\bvert}{\bigg\vert}
```

# APS360 Week 5 - Convolution Neural Networks Part II

## Visualizing Convolutional Filters

Below is an image representing the outputs of the various convolutional layers in a CNN. Each kernel within a CNN is used to learn different features of the image from specified kernels that are found using gradient descent iteration. These kernels are then joined through pooling methods and fed into a higher level convolutional layer. This subsequent layer is then able to combine the outputs of the low layer into higher level features not inheriently apparent from the base image. This process continute for however many convolution layers are present. This process is called **Hierachial Learning** and is believed to be the way in which our own neurons process information from our own vision.

<p align="center">
    <img src="/Week%205/cnn_kernels.png">
</p>

### Saliency Maps

If we want to understand what information the network is trying to process, we can use something called **Saliency Maps**. These will provide some sort of idea in the areas of focus that the CNN is paying attention to.

When generating a **Saliency Map** the following steps are conducted:

1. An image is feed into the CNN
2. Compute the gradient of the loss with respect to the changes in individual pixels
    - This is done in tandem with the gradients of the weights and biases
3. Take the maximum of the absolute values of the gradients
4. Visualize the map by assigning some sort of colour chanel value to gradient and overlaying it on the original input image

<p align="center">
    <img src="/Week%205/saliency.png">
</p>

> **Note:** Most of the time, these maps are not particulalry useful. They are generally used in order to provide intuiiton behind what the neural network is focusing on however, they can often times misrepresent what the CNN is actually doing.

## LeNet - The CNN Before Deep Learning

There are many versions of this network however, the commonly refered to is LeNet-5. It consists of 2 convolutional layers, 2 subsampling/pooling layers and 3 fully connected layers. The advantage of these networks is that they have translational, rotational, scale and noise invariance, meaning that it will be able to classify the image regardless the orientation, size or interuption of the subject.

<p align="center">
    <img src="/Week%205/lenet.png">
</p>

### Pytorch implementation

Since the LeNet-5 is a standard CNN, it can be easily implemented in Pytorch.

```python
class example(nn.Module):
    def __init__(self):
        self.conv1 = nn.Conv2d(
            in_channels = 1,
            out_channels = 6,
            kernel_size = 5,
            stride = 1,
            padding = 0
        )
        self.pool nn.AvgPool2d(
            kernel_size = 2, 
            stride = 2
        )
        self.conv2 = nn.Conv2d(
            in_channels = 6,
            out_channels = 16,
            kernel_size = 5
        )
        self.fc1 = nn.Linear(
            in_features = 5*5*16,
            out_features = 120
        )
        self.fc2 = nn.Linear(
            in_features = 120,
            out_features = 84
        )
        self.fc3 = nn.Linear(
            in_features = 84,
            out_features = 10
        )

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 5*5*16)
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = self.fc3(x)
        return x

```

## Improvements made in Mordern Architecture

Many of the improvements made to modern CNN architechture stem from the ANN best practices covered within Lecture 3 in addition to overal hardware advancements. However, the overall construction and hyperparameters of CNNs plays the largest role of how effcient we can make these networks.

### AlexNet

In 2010, there was a large dataset of with over 14 million entires and over 1000 classes with the goal of providing large amounts of data to help improve learning. This data was then used in a competition for models to be developed to improved upon the accuracy of the current models which usually saw increases of 1-2% per year. However, one such year saw the model **AlexNet** leading an accuracy improvement of roughly 10%.

<p align="center">
    <img src="/Week%205/alexnet.png">
</p>

It performed much better than base LeNet architechture due to the **size of the dataset avaliable**, increase in **computing power** due to higher quality GPUs and the use of **ReLU functions**, **SGD with momentum**, **weight decay** & **weight dropout** in addition to **Data Augmentation**.

#### Data Augmentation

Data Augmentation is the process of applying random transformations to the training data which increases the size of the training data but help generalize learning material and learn to classify inputs with different orientations.

<p align="center">
    <img src="/Week%205/augementation.png">
</p>

#### Generalization & Depth

It is generally found that the large the depth of the convolution portion of a CNN is, the more the generalization improved. However, there comes apoint in which the layer is so deep that gradients end up becoming extremely small or extremely large, this is an issue even more modern architechture solves. This is typically combatted by using **ReLUs**, **normalizing input data** and **Residual Connections??**.

### GoogLeNet

The goal of **GoogLeNet** was to take the network as deep as possible with as many convolution layers as possible. This network ends up using 22 convolution layers with only 4 million parameters compared to the 60 million used in **AlexNet**.

#### Inception Block

In order to reduce the number of paremeters despite the increase in convolution layers, GoogLeNet uses **Inception Blocks**. These blocks use parallel convolution with different sized kernels (typically 3x3, 5x5 & 7x7), reducing the need for hyperparameter optimization of these kernel sizes. After completing the parallel convolution, the channels are then be concatinated depth wise.

This is typically initially done with 1x1 point wise convolution in order to control the depth of the output, particaularry reducing the depth for things like RGB values.

<p align="center">
    <img src="/Week%205/inception_block.png">
</p>

#### Auxilary Loss

The use of inception block causes the network to be extremely deep which then causes the problem with shrinking gradients. In order to combat this, GoogLeNet uses **Intermediate Classifers** within the intermediate layers. The **Final Loss** of the entire CNN is the equal to the **Sum of Intermediate Losses** and the **Final Output Loss**.

### Visual Geometry Group (VGG)

This model created my Oxford had a very simply acrchitecture of stacked blocks with 11-19 layers with over 138 million paramters.

Despite the lage amount of paramters, the model showed that kerenls szies only need to be 3x3 and that stacked 3x3 kernels can approximate larger kernels more effciently.

Almost all modern CNNs use exlusively 3x3 kernels because of this paper.

### Residual Networks (ResNet)

**Residual networks** aims to again solve the problem of training extremely deep neural networks because despite the progress from all previously mention tricks, there is a threshhold of how deep a network can be before training the network fails again.

Residual Networks use **skip connections** to combat shrinking gradients. This allows a pathway for deeper layers to gain direct access to signals which would have been lost to vanishing gradients.

```python

# regular activation layer
next_activation = layer(activation)

# residual layer (input + output of activation of the layer)
next_activation = activation + layer(activation)
```

Additioanlly, **ResNet** used a stride of 2 within the convolution layers instead of max or average pooling however, they would use **Global Average Pooling**, which reduces the kernel size of the output of the final convolution layer into a single scalar.

> **Take Away:** In order for more effective **Deep Learning**, we need deeper and deeper models. Therefore any model that has tools to reduce the short comings of having a deeper model will drastically improve the accuracy.

## Transfer Learning

Humans are good at transfering the learning and knowledge that we gain from other tasks or experiences and applying it to learn something new. Since NNs are based on human biology it would also be useful to be able to transfer and adapt the models we create to fullfil different yet similar purposes. This transfer learning is the backbone of a large majority of AI based industries and is extremely important to creating an adaptable product.

### Embeddings Method

Embeddings are the intersection point between the convolution layers and the classifcation layers. The convolution layers up until this point work completely independent of the task trying to be achieved by the network. Because these layers learn extremely useful information, we can freeze the weights at the end of the convolution network, then add and train additional layers that are more suitable for the desired task.

<p align="center">
    <img src="/Week%205/embeddings.png">
</p>

### Fine-Tuning for New Task

After freezing weights, training the model for the new task with a lower learning rate can help adaupt the features to the new task.

### Pytorch Implemenation

Pytorch allows for the reproduction of existing models with either the pretrained weights or untrained models in which we are able to feed data into to train ourselves. This is what is recomended for when completing our projects.

```python
import torchvision.models
alexnet = torchvision.models.alexnet(pretrained=True)

Inception = torchvision.models.inception.inception_v3(pretrained=True)

vgg16 = torchvision.models.vgg.vgg16(pretrained=False)
vgg19 = torchvision.models.vgg.vgg19(pretrained=True)
resnet18 = torchvision.models.resnet.resnet18(pretrained=True)
resnet152 = torchvision.models.resnet.resnet152(pretrained=True)
feature_data = alexnet(image)
```
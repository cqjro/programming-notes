---
title: APS360 Week 4
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

# APS360 Week 4 - Convolutional Neural Networks

## Why convolutional neural networks?

There are many real world examples in which a neural networks training is completely useless because a different amount of inputs. One such example of this is something like the MNIST Dataset. Each image is 28 by 28 pixel in lenght, however its very realisitc to want to feed inputs into this network that are larger than 28 by 28 in order to receive the prediction and identification. In other to do this you can either retrain the model from scratch to account for the extra inputs OR you can pre-process the data to fit the correct number of inputs. In either case this is an undesirable amount of extra work to adjust for the neural network.

Additionally, networks with large input layers and hidden layers can result in to extremely large amount of parameters that make it near impossible to actually compute. For example, in a network with a 200x200 image inputs, 500 neuron hidden layer and 200 neuron second hidden layer, that makes for over 20 million parameters. This can lead to large **Computational Complexity** making it extremely hard to train, creates **Bad Inductive Bias** causing to network to ignore the underlying geometric of the image and makes the netowrk **Not Flexiable** which causes entirely new models to be creates for different size images.

## Convolution Operator - Better Inductive Bias

Convolution is a mathimatical operator on two functions f and g that expresses how the shape of one is modified by the other. It is commonly used in Electrical, Computer & Mechatronics Engineering in the field of signal processing.

```math
\newcommand{\al}[2]{\text{#1} & \hspace{1cm} #2 &} % faster lines in display equations

(f*g)[n]=\sum_{k=-\infty}^\infty f[k]g[n-k]
```

<p align="center">
    <img src="/Week%204/convolution.png">
</p>

### Convolution for 2D Images

Convolution of an image is effectively a given image and applying some sort of transformation onto it in the form of a kernel matrix. The application of this kernel matrix acts as a "filter" and outputs a modified version of the image (similar to applying a filter to photos on social media). This allows us to then extract meaningful data from the images which we can then use to train a neural network. Typically this is used to detect vertical lines, horizontal lines. Large kernel matricies are used to detect **"Blobs"** which are reigons that differ in properties such as brightness and colour compared to their surroundings.

<p align="center">
    <img src="/Week%204/image_convolution.png">
    <img src="/Week%204/image_convolution2.png" width="400">
</p>

The convolution of the image $I$ with filter kernel $K$.

1. Multiply each pixel in $I$ in range of the kernel by the corresponding element of kernel $K$ (elementwise multiplcation of the section of the image and the kernel)
2. Sum all these products and write to a new 2D Array
3. Slide kernel accross all area of the image until you reach the ends

```math
\newcommand{\al}[2]{\text{#1} & \hspace{1cm} #2 &} % faster lines in display equations

y[m, n] = I[m, n]*K[m, n]=\sum_{j=-\infty}^\infty \sum_{i=-\infty}^\infty I[i, j]\cdot K[m-i, n-j]
```

### Kernels

Since every image is different, they generally require their own kernels in order to determine the useful data from the images. A problem then arises for determining what kernel to use on these images. In the past these kernels were hand crafted for the specific scenario where a series of kernels were used extract useful data. However writing algorithms manually are extremely time consuming and inpercise, which is where deep learning comes in in order to accelorate the process.

---

## Convolutional Neural Networks (CNN)

CNN's were inspired by biological neurons responsible for vision. Individual neurons respond to stimuli only in a specific reigon of the visual field and the collection of such neurons overlap to create the visual field. Some of these neurons only react ot horizontal lines or other line orientation. The higher level neurons are based on the outputs of the neighbouring lower level neurons in order to construct higher level objects from these lower level features (Ex. veritcal and horizontal lines being combinded together to form a rectangle). This hierarchy spans many layers in order to combine the lower level features into higher level outputs.

Convolutional neural networks apply filters into the neural networks so that features do not have to be hand crafted. It does so through:

- **Locally Connected Layers**
  - Local Features in small regions of the image captured by the kernal space
- **Weight Sharing**
  - Detect the same local features across the entire image
  - This is redcues the overall number of parameters as the weights are shared by the regions in which te kernal matrix is applied.
- **Learned Kernel Values** 
  - The networks learn the kernel values analagous to weights removing the need for these handcrafted kernels

This solves the problem of computational complexity by reducing the overall parameters, solves the problem of losing the geomtric intuition by specifically extracting the geometric information in the form specific kernels and makes the network flexiable as the convolution can be applied to any size input.

In the network performs these actions by first passing the image into a series of locally connected convolution layers called an **Encoder** which learn the features of the image before then passing it into the classification layers which is affectively a ANN from before.

<p align="center">
    <img src="/Week%204/CNN.png">
</p>

---

## Convolution Layer Computation

1. When first intialized, the kernals are chosen at random. 
2. Within the convolution layer, during a forward pass call, the images are convolved using the current kernel.
3. During a backward pass, the kernels are updated with gradients treating each element of each kernel as its own weight.

<p align="center">
    <img src="/Week%204/kernel_weight.png">
</p>

### Determining the Size of the Output
For each dimension of an image, the corresponding output dimension of the convoluation can be determined using the following formula:

```math
o = \left[\frac{i+2p-k}{s}\right] + 1
```

```math
\newcommand{\al}[2]{\text{#1} & \hspace{1cm} #2 &} % faster lines in display equations
\newcommand{\var}[3]{#1: & \hspace{1cm} \text{#2} & \text{#3} &} % easy display of variable 

\begin{align*}
    \var{i}{Image Dimension}{} \\
    \var{k}{Kernal Size}{} \\
    \var{p}{Zero Padding Size}{} \\
    \var{s}{Stride Size}{} \\
\end{align*}
```

#### Zero Padding

During trainning, zeros are added around the border of the images in order to prevent the loss of information around the border as the kernel would only be considering the border pixels onces due to the geometric limitation creating a bias for the central pixels. Adding the zero border allows for the kernel to consider the border pixels within our analysis while not adding any additional information since the values are 0. Addtionally if we want to control the height and width of the image to keep it consistent with the previous layers, adding zero padding allows us to do so.

<p align="center">
    <img src="/Week%204/zero_padding.png">
</p>

#### Stride

Additionally, the distance between consecuative kernels can be modified in orer to control the resolution of the output image of the layer.

<p align="center">
    <img src="/Week%204/stride.png">
</p>


### CNNs & RGB

Generally, images have 3 colour channels which represent the brightness of a the red, green abd blue components of a pixel in order to envoke a specific colour when seen. This means that the images actually becomes a 3D tensor because of the colour channels at each pixel. This subsequently means that the kernel must also be a 3D tensor in order to traverse the image space.

In this case however, the depth of the kernal **MUST** match the depth of the **input** in order to be able to traverse the convolute all of the RGB values into a single scaler. This means that the ouput of the RBG convolutions will always result in a depth of 1.

### Feature Maps

When gathering features from the image, it is better to use multiple kernels that gather different information about the image working in parallel in order to feed more valuable information into the classifer for better predictions.

When using multiple kernels within teh same layer, the depth of the output then becomes the amount of kernels that are used added together. The outputs are effectively overlayed on top of eachother to create this depth.

---

## Pooling Operators & Consolidation

In fully connected neural networks, we consolidate the information with in the hidden layers in order to reduce the number of units we are considering and remove information not useful to the task of the neural network. We want do so the same thing in CNNs as well before feeding the convolution layer outputs into the classiciation layers. The three main ways of doing so are using **Max Pooling**, **Average Pooling** and **Strided Convolutions**.

### Max Pooling

This methods usings a specified "kernal" filter matrix and stride value accross the output matrix and only returns the maximum value within the kernal space on the image. The dimension of the output from max pooling uses the following formula:

```math
o = \left[\frac{i-k}{s}\right] + 1
```

<p align="center">
    <img src="/Week%204/max_pooling.png">
</p>

### Average Pooling

This method, similar to max pooling, uses a kernel fileter matrix area and a stride but only returns the average of all the elements within the filter space on the image. The output resolution is calculated the same way as Max Pooling.

```math
o = \left[\frac{i-k}{s}\right] + 1
```

<p align="center">
    <img src="/Week%204/average_pooling.png">
</p>

### Strided Convolution

In the most recent arcitechture, people have been moving towards using strided convolutions instead of using pooling operations, since this already maps the image space to a smaller output volume and is less computationally heavy.

<p align="center">
    <img src="/Week%204/strided_convolution.png">
</p>

---

## CNN Construction

Generally, the convolution layers will always have the same steps:

1. Convolution
2. Linear Rectification (ReLU)
3. Pooling (Max Pooling or Average Pooling)

The process is then repeated as many times as wanted.

<p align="center">
    <img src="/Week%204/cnn_structure1.png">
</p>

The reason for this structure is because of the biological analogy. Intially, the convolution depth tends to be low and the network begins learning very low level features (Ex. vertical & hortizontal lines). The depth is lower initially because there are not a lot of low level features for the network to learn. Addtionally, lower level features tend to be small enough to be captured by smaller kernels where has higher level features, such as things like faces, will not fit in these small kernels. Rather than increasing the size of the kernels to fit higher level features, we use pooling to make the resolution of the outputs smaller which enables the learning of higher level features by fitting them into the kernels. More kernels are introducesd to learn these higher level features because there are more of them thus increasing the convolution depth.


<p align="center">
    <img src="/Week%204/cnn_structure2.png">
</p>

### CNNs in Pytorch

#### Defining Layers

```python
# Fully Connected Layer for the Classifier
self.fullyconnected = nn.Linear(n, 30)

# Convolution Layer Definition
self.conv1 = nn.Conv2d(
    in_channels = 3,
    out_channels = 7, 
    kernel_size = 5, # defined as int or tuple
    stride = 1, # default 1, int or tuple
    padding = 1 # default 0, int or tuple
)
```

#### Example CNN
```python
class LargeNet(nn.Module):
    def __init__(self):
        super(LargeNet, self).__init__()
        self.name = "large"
        self.conv1 = nn.Conv2d(3, 5, 5)
        self.pool = nn.MaxPool(2, 2) # defines pooling method
        self.conv2 = nn.Conv2d(5, 10, 5)
        self.fc1 = nn.Linear(10*5*5, 32)
        self.fc2 = nn.Linear(32, 10)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) # runs the first convolution, activation & pooling
        x = self.pool(F.relu(self.conv2(x))) # runs the second convolution, activation & pooling
        x = x.view(-1, 10 * 5 * 5) # flattens the out the output of convolution layer for input into fc
        x = F.relu(self.fc1(x)) # runs through fc1
        x = self.fc2(x) # runs through to final layer
        return x


```
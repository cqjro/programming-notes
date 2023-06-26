---
title: APS360 Week 6
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

# APS360 Week 6 - Unsupervised Learning

## Motivation for Unsupervised Learning

All deep learning models that we have covered up until this point in the course use **Supervised Learning** which involes the confirmations and iteration of model parameters based on **Groud Truth Labels**.

The problem with **Supervised Learning** is that in order to train it, it **REQUIRES** large amounts of the labelled ground truth data which is expensive both time wise an montarily. Addtionally, unlabeled data is widely avaliable, accessiable and realistic in most real world scenarios requiring deep learning.

Supervised Neural networks are inheriently based on biology, but however, do not replicated it compleltely. Infants tend to learn things very easy and most importantly sometimes **Unsupervised**. This means that babies can learn things like language, movement, patterns without explicit communcation (because of a lack there of). In order to truely replicate the biological inspiration for learning, the next logical step would be to replicate the unsupervised learning found in babies.

We can think of unsupervised learning as intially recognizing inherit patterns in data. A peson would be able to differentiate animals like cats and dogs by simply using sight alone and recognizing the inherient patterns in their appearance without ever knowning what either animal is called. We can apply this concept to unsupervised networks and allow them to learn these inherit patterns and classify the data as such and then convert the network into a supervised task by applying labels to the classes.

A lot of time this unsueprvised learning is represented in feature clusters which serve to identify data points based on their inherient patterns which can then be classified later.

<p align="center">
    <img src="/Week%206/feature-clusters.png">
</p>

### Types of Learning

Unsupervised Learning
: Learning Patterns from data without groud truth labels.

Supervised Learning
: Learning patterns from data with ground truth labels and iterating paramters based on the associated prediction error.

Self-Supervised Learning
: Sucesss of supervised learning without relying on human provided supervision. This can be in the form of masking part of an input and making predictionns based on the masked information allowing the network to get a deeper understanding of unerlying features.

Semi-Supervised Learning
: Learning from data that mostly contains unlabelled samples, with only a small amount of the data containing manually labeled data. This effectively allows us to generalize the results from data that we do not have labels for.

## Autoencoders

**Autoencoders** find representations of the input data that can be used to reconstruct it. This done using two components:

Encoders
: Convolution Layers, ReLU and Pooling techniques used for feature identificiation and dimensional reduction

Decoders
: Replacement for a classifer in a CNN
Converts the internal representation of the input and reconstructs it as an output using a generative network

<p align="center">
    <img src="/Week%206/autoencoder.png">
    <img src="/Week%206/autoencoder2.png">
</p>

The autoencoder network is structures in a somewhat hourglass shape. The input data is compressed to a lower dimensional space by the encoder from which the decoder generates the image from the represented compressed data with the same dimensionality as the input image.

This architechture when trained forces the optimiation towards parameters which have the highest probability of compressing the input data such that the learned representation can be decoded to generate an image that is identical to the input. This is obviously not always possible, however is the ideal scenario of these networks.

In similer terms, the autoencoder forces the network to learn the **Most Important Features** of the input data when compressed such that the decoder will be able to replicate the image from those features alone.

> Note: It is important to consider that even if the visualized output is nearly identical to the input image, this does not rule out the possibility of overfitting in the model.

### Applications

Autoencoders, because of their architecture encoding data and generating data, can be used in many different applications.

1. Feature Extraction
2. Unsupervised Pre-training
3. Dimensionality Reduction (encoding/compressing)
4. Generation of New Dat
   - used for generative AI applications
5. Anomaly Detection
   - Autoencoders tend to be bad at reconstucting outliers, thus outliers can be detected through inaccurate image generation

### Pytorch Implementaion

```python
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        encoding_dim = 32
        self.encoder = nn.Linear(28*28, encoding_dim)
        self.decoder = nn.Linear(encoding_dim, 28*28)
    def forward(self, img)
        flattend = img.view(-1, 28*28)
        x = self.encoder(flattened)
        x.F.sigmoid(self.decoder(x))
        # Sigmoid function is used to scale the output between 0 & 1
        return x

    criteration = nn.MSELoss()
    # computes the error between the x and the input img
    # allows for the autoencoder to be optimized to minimize error 
    # between the input and the generated output
```

### Stacked Autoencoders 

In order to make the learning Deeper and more effective, we always want to add more layers to the process. For this reason, autoencoders tend to be stacked with multiple layers but are symmetrical around the encoded data layer to ensure that the model is able to generate an image that has the same dimensionality of the input.

<p align="center">
    <img src="/Week%206/stacked-encoders.png">
</p>

### Denoising Autoencoders

Noise can be added to the input images of the autoencoder to force the model to learn more important features. The autoencoder will then also be trained to generate and recover images from the applied noise. This important as it prevents the model from simply copying the inputs outright and forces it to replicate the inputs based on its inherient features.

<p align="center">
    <img src="/Week%206/noise.png">
</p>

### Pytorch Implementation (Guassian Noise/Salt & Pepper Noise)

```python
# how much noise to add to images
nf = 0.4 # noise factor

# add random noise to the input images
# img + noise * random normal distribution of img.shape
noisy_img = img + nf * torch.randn(*img.shape)

# Clip the images to be between 0 and 1
noisy_img = np.clip(noisy_img, 0., 1.)

# compute predicted outputs using noisy_img
outputs = model(noisy_img)

# compute error compared to the origninal input image
loss = criterion(outputs, img)
```

### Image Generation

The autoencoders drastically reduce the dimensionality of the images and in doing so creates some kind of structure of the features called the **embedding space**. 

After training, the **embedding space** will store all the most important features of the input images in order save space, the network will map similar images with similar embeddings. 

We can exploit the mapping of similar images to then generate new types of images with similar features.

### Generation by Interpolation

One of the main ways we can generate images is interpolating the results of their embeddings. This way we get a middle ground of their features creating a unique image. We can use weighted interpolation in order to control the degree to which image is has its features priortized.

<p align="center">
    <img src="/Week%206/image-interpolation.png">
</p>

> The weight on the bottom row decreases from left to right, showing transition from an interpolated image similar to Image 2 to an interpolated image similar to Image 1.

#### Steps for Image Interpolation

```math
\newcommand{\var}[3]{#1: & \hspace{1cm} \text{#2}} % easy display of variable definitions in align format

\begin{align*}
\var{i}{Input Image}{} \\
\var{z}{Embedded Image}{}
\end{align*}
```

1. First, two images will have their embeddings computed by being run through the encoder

```math
z_1 = \text{encoder}(i_1) \\
z_2 = \text{encoder}(i_2)
```

2. Compute a weighted interpolation of the two embedded images

```math
z_{gen} = (1-w)z_1 + wz_2
```

3. Decode the interpolated embedding to produce the generate image

```math
i_{gen} = \text{decoder}(z_{gen})
```

## Variational Autoencoders (VAEs)

Variational Autoencoders have two main attributes that differs from the base autoencoders:

Probabilisitic
: Output of VAEs are partly determined by chance regardless of training

Generative
: Generate new instances that look like and reflect the instances in the training set

These properties make the network non-determininistc and allow for randomness to change the output if given the same input over and over, making this type of autoencoder ideal for generative usecases.

<p align="center">
    <img src="/Week%206/VAE.png">
</p>

The encoder portion of VAEs generate a normal distribution with mean $\mu$ and standard deviation $\sigma$ as apposed to the fixed embedding method used in base encoders. This effectively means that the encoder is learning the mean and standard deviation required to create a distribution to generate an appropriate output.

An embedding is then sampled from the distribution which is then decoded and restructed by the decoder through the following expression.

```math
z = \mu + \sigma\cdot\epsilon \\
\epsilon \sim \mathcal{N}(0, I) \\
\text{$\epsilon$ sampled from the specific normal distribution to simulate randomness}
```

### VAE Distribution

In order to ensure that the encoder distrbution of $q_{\phi}(z|z)=\mathcal{N}(\mu, \sigma)$ is a close as possible to $p(z) = \mathcal{N}(0, I)$, we use something called Kullback-Leibler Divergence (KL Divergance) to measure the difference between the two distributions.

```math
D_{KL}(P||Q) = \sum_{x\epsilon X} p(x)\log\left(\frac{p(x)}{q(x)}\right)
```

When plugging in the specific encoder distributions, we get the following relationship:

```math

D_{KL}(p|q) = \frac12\sum_{i=1}^N \left[ \mu_i^2+\sigma_i^2-\left(1+\log(\sigma_i^2)\right) \right]

```

This term is then used as an additional loss term along with the MSE to push the $\mu$ and $\sigma$ towards being similar to $\mathcal{N}(0, I)$.

<p align="center">
    <img src="/Week%206/VAE-training.png">
</p>

## Convolutional Autoencoders

Convolutional are excactly as the name implies, they are Autoencoders with Convolutional layers that learn the visual features of images and compresses them into embeddings. 

<p align="center">
    <img src="/Week%206/convolutional-autoencoder.png">
</p>

### Transposed Convolution

Since the encoder will be using convolutional layers, it follow that the decoder must use the same in order to upscale the features to the originial input size. This process is called **Transposed Convolution**. 

**Transposed Convolution** maps 1 pxiel to a image of nxm pixels. The kerenls are optimized and learned the same as the regular convolution.

#### Steps of Transposed Convolution

 1. Multiply each pixel of the input image by the kernel matrix to get weighted kernel matrix
 2. Insert the weighted kernel matrix into the output matrix
 3. Sum together any overlapping inserted values

<p align="center">
    <img src="/Week%206/transposed-convolution.png">
    <img src="/Week%206/transposed-convolution2.png">
</p>

#### Output Dimension

```math
o = s(i-1) + (k-1)-2p_i + p_o +1
```

```math
\newcommand{\var}[3]{#1: & \hspace{1cm} \text{#2}} % easy display of variable definitions in align format

\begin{align*}
\var{o}{Output Dimension}{} \\
\var{i}{Input Dimension}{} \\
\var{s}{Stide}{} \\
\var{k}{Kernel Size}{} \\
\var{p_i}{Input Padding}{} \\
\var{p_o}{Output Padding}{}
\end{align*}
```

#### Padding

**Padding** within transposed convolutional layers works in the opposite way to regular padding. The padding in this case specifies the perimeter around the image in which to truncate from the output.

<p align="center">
    <img src="/Week%206/transpose-padding.png">
</p>

### Output Padding

Regardless of the size of the input image, a regular convolutional layer will return the sames size output matrix (Ex. both 7x7 and 8x8 => 3x3). When the ouput matrix is upscaled, there is no information to inidicate the dimension of the output image (Ex. 3x3 => 7x7 ? or 8x8 ?). Output padding increases the resulting output shape on one side to keep the outputs consistent. 

> Note: This does not actually add a 0-padding on the perimeter but effects the calculation to reshape the output

#### Stides

Like padding, **Strides** in transpose layers again has the opposite effect to that of regular convolutional layers. Increasing the stride increases something call the **unsampling effect**. The stride effectively defintes the position of the output each kernel/input prouct is placed.

<p align="center">
    <img src="/Week%206/transpose-stride.png">
</p>

### Pytorch Implementation

```python
# Normal Convolution Layer
conv = nn.Conv2d(
    in_channels=8,
    out_channels=8,
    kernel_size=5
)

x = torch.randn(2, 8, 64, 64)
y = conv(x)
y.shape
 >>>>>> torch.Size([2, 6, 60, 60])
```

```python
convt = nn.CovTranspose2d(
    in_channels=8,
    out_channels=8,
    kernel_size=5,
    padding=2,
    stride=2,
    output_padding=1
)

x = torch.randn(2, 8, 64, 64)
y = conv(x)
y.shape


# output without padding, stride, or output padding
>>>>>> torch.Size([2, 8, 64, 64])

# output with padding 
>>>>>> torch.Size([32, 8, 64, 64])

# output with padding & stride
>>>>>> torch.Size([32, 8, 127, 127])

# output with padding, stride & output padding
>>>>>> torch.size([32, 8, 128, 128])
```

```python
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Sequential puts a bunch of layers together and treats it as a single layer
        self.encoder = nn.Sequential(  
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 7)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1,output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def embed(self, x)
        return self.encoder(x)

    def decode(self, e):
        return self.decode(e)
```

## Pre-training with Autoencoders

This is an alternative to **Transfer Learning**. Transfer learning involved transfering the learned feautres of a convolutional layer to differently purposed tasks. Autoencoders can fulfill a similar task because they can be trained on unlablled data, meaning that these networks can be applicable to any task.

Once the Bottleneck hidden layer of the autoencoder is reached, the network results can be transfered to a different task such as classification.

<p align="center">
    <img src="/Week%206/pretrain.png">
</p>

## Self-Supervised Learning

The motivation behind self-supervised learning is to be able to turn unsupervised learning into supervised learning.

The core goal is to develop a method to generate labels automatically and solve the problem of someone having to understand the model content.

The strategy in order to achieve this goal involves being very clever with designing the puzzle for the network to solve as will be seen the following examples.

### RotNet

RotNet is an example of a network which uses its clever design in order to achieve self-supervision. The network involes rotating images 0º, 90º, 180º, and 270º and then having the network classify the degree of rotation. This network then becomes unsupervised because the rotation classes will then be generated automatically without needing knowledge on the originial dataset.

This is extremely useful. Not only does the model generate its own classes, but in order to identify these classes the network has to understand the features within the image to classify the degree of rotation making it easy to transfer these learned features to other scenarios.

RotNet is a great example of coming up with a clever puzzle to generate classes automatically. The problem is that coming up with puzzles similar to this that generate their own classes is extremely difficult.

<p align="center">
    <img src="/Week%206/RotNet.png">
</p>

### Contrastive Learning

Contrastive methods are a newer approach to self-supervised learning. Similar to RotNet, constrastive learning involves augmenting an input image with two seperate augmentations (i.e two images are produced) then having the network train to determine whether the two augmented images comes from the same source. Training in such a way equal performane to training with supervised training sets.

<p align="center">
    <img src="/Week%206/contrastive-learning.png">
    <img src="/Week%206/SimCLR.png">
</p>


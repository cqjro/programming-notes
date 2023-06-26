---
title: APS360 Week 7
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

# APS360 Week 7 - Generative Adversarial Networks

## Generative Models

Generative models, as discussed in the unsupervised learning section do not require a **Discriminative Model** and **Ground Truth Labels** and instead encode inherient features of the model into embeddings. These embedded features can then be used in conjuction with interpolation and probablititic techniques can be used to create unique images. There is still a loss function within these models based on an auxiliary task that these models serve to fulfill.

**Unconditional Generative Models**
: Generative models which only recieve random noise as inputs with no control over the categories they generate

**Conditional Generative Models**
: Recieve one-hot encoding of the target category in addition to random noise or masking OR embeddings generated from external models
Provide enhanced user control over what the model will generate

### Types of Generative Models

- Autoregressive Models
- Variational Autoencoders
- Generative Adversarial Networks (GANs)
- Flow-Based Generative Models (outside of course scope)
- Diffusion Models (outside of course scope)

## Genrative Adversarial Networks (GANs)

The motivation behind GANs comes from the cons of using autoencoders. Basic autoencoders have to compress the images into their most important features. This encourages the autoencoders to reduce the MSE loss by predicting the average pixel. As a result, the output image of these auto encoders tends to be extremely blurred.

<p align="center">
    <img src="/Week%207/blur.png" height="400px">
</p>

GANs aim to improve upon the loss system by encouraging the images to reflect those from which they are derived from. To do this, GANs train two models:

**Generator**
: Generates images based on input data with the goal of looking as real as possible

**Discriminator**
: Attempt to distinguish between real world images and the fake ones produced by the **Generator**

This setup effectively allows the discriminator to act as the loss function. If the discriminator is unable to distinguish between real world images, it means the generative images are realistic.

This effectively means that the **generator** will be trained in order to maximize the probability that the discriminator labels the generated image as real, while the **discriminator** will be trained to maximize identifying real vs fake images. Since these models loss functions are so interconnected, training of each model will alternate between the two under the following:

<p align="center">
    <img src="/Week%207/GAN-training.png" height="400px">
</p>

### Problem Training GANs

GANs, like all other architecture have draw backs and difficulties within the training process.

#### Vanishing Gradients

If the discriminator is very good an predicting the real vs fake images, meaning small changes in generator weights do not have an effect on the discriminator, it will be impossible for the generator to learn how to fool it. Since small weight changes do not effect the discriminator output, incremental chagnes to the generator will not be possible therefore creating a vanishing gradient.

#### Mode Collapse

In most use cases, we ideally want to procuce a variaity of unique outputs. If a generator continously produces the same output or set of outputs, the discriminator would ideally reject these outputs as been fake. If, however, the discriminator is trapped in a local optimum, it will not be able to adapt to adapt to changes in the generator and produce only one kind of output regardless the geneator output.

#### Failing to Converge

GANs take a very long time to train due to the optimization process and it is very difficult to gauge the progress of the training process.

Some of the stragies to reduce training time include using `LeakyReLU` instead of `ReLU`, using `Batch Normalization` and `regularization of discriminator weights` and `noise to discriminator inputs`.

### Pytorch Implementation

```python
# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(28*28, 300),
            nn.LeakyReLU(300, 100)
            nn.Linear(300, 100)
            nn.LeakyReLU(0.2),
            nn.Linear(100, 1)
        )
    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = self.model(x)
        return out.view(x.size(0))


def train_discriminator(discriminator, generator, images):
    batch_size = images.size(0)
    noise = torch.randn(batch_size, 100)
    fake_images = generator(noise)
    inputs = torch.cat([images, fake_images])

    # Label the images vs the fake images as 0 and 1
    labels = torch.cat([torch.zero(batch_size), torch.ones(batch_size)])

    outputs = discriminator(inputs)
    loss = criterion(outputs, labels)
    return outputs, loss
```

```python
# Generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(100, 300)
            nn.LeakyRelu(0.2)
            nn.Linear(300, 28*28)
            nn.Sigmoid()
        )
    def forward(self, x):
        out = self.model(x).view(x.size(0), 1, 28, 28)
        return out.view(x.size(0))

def train_generator(discriminator, generator, batch_size):
    batch_size = images.size(0)
    noise = torch.randn(batch_size, 100)
    fake_images = generator(noise)
    outputs = discriminator(fake_images)
    # test the fake images and reward generator if the disciminator was fooled

    label = torch.zeros(batch_size)
    loss = criterion(output, labels)
    return fake_images, loss
```

## Application of GANs

- Gray Scale to colour
  - This can be used to restore colour to old black and white images
- Conditional Generation
  - Taylor the image generation to only output specific image categories at request
- Style Transfer
  - Apply stylistic elements of one image to another
  - Ex. Converting a painting into a realistic image or viceversa
  - Ex. Applying zebra stripes to a horse or viceversa

## Adversarial Attacks
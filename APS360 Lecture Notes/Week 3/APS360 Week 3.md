---
title: APS360 Week 3 - Training Neural Networks
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

# APS360 Week 3 - Training of Neural Networks


## Hyper Parameters

When trainiing a neural network, there are two types of parameters that are optimized for. The weights of the inputs into the network and the **Hyperparameters**. The hyperparameters function as the "settings" of the neural network and are the parameters are not optimized using gradient descent. These include things like:

- Batch Size
- Number of Layers
- Layer Size
- Type of Activation Function
- Learning Rate
- etc

Neural Networks work to optimize the weights through gradient descent through the **inner loop** of optimization. In contrast, **Hyperparameters** are optimized through the **outer loop** of optimization. 

This basically means that the **weights** and **biases** are optimized based on **training metrics** where as the **hyperparatemers** are optimized based on the **test metrics**.

<p align="center">
    <img src="/Week%203/hyperparameters.png">
</p>

### Tuning Hyperparameters

Hyperparameters are tuned using searches that opimizes the model accuracy based on important and unimportant parameters. This basically means you are exploring the structure space and minimize the average validation error.

Grid Search
: A hyperparameter tuning method that is exhaustive and explores all methods possible in the hyperparameter space

Random Search
: Randomly sample n-sample set of parameters and train them then use the model with the lowest validation error. (PREFERED METHOD DUE TO BEING LESS EXPENSIVE)

<p align="center">
    <img src="/Week%203/hyperparameter_search.png">
</p>

---

## Optimizers

Learning problems when it comes to neural networks are really just optimization problems, in which we are trying to **minimize the error** from a given **loss function** by determine the change in the **weight**.

An **optimizer** determines how the **weights** should change based on the value of the **loss function**. Which effectively solves the **credit assignment problem** and assigns "blame" to various parameters. This is typically represented through the use of gradient descent.

### Stochasic Gradient Descent (SGD)
This method takes a training sample from the training dataset at random. The SGD method tends to me more erratic and random and infact may not be the fastest way to optimize. However, the randomness increases the ability to optimize and helps with learning.

<p align="center">
    <img src="/Week%203/SGD.png">
</p>

This is typically done in something called **Mini-Batch Gradient Descent**. In which a batch on n-samples from the dataset are selected at random in order to generate the random noise while also increasing the speed of training since only taking single samples in very ineffcient.

Batch Size
: Training examples size per optimization step
(this is chosen by fitting the gpu memeory to 100%, can be done using nvida-smi using an nvida gpu)

Iteration 
: one step of samples taken and optimization performed

Epoch
: The times all the train data is used once to update the parameters (if 1000 samples, batch size = 10, then 1 epoch = 100 iterations)

### Momentum
Gradient decent tends to happen with millions or billions of dimentiosn in neural networks. Most points in which the graidents are 0 are saddle points. This means that in some dimensions this space is a minimum and in others this may be a saddle point. This creates areas such as **Ravines** which have curves more steep in one dimenion than another.SGD has trouble navigating through ravines due to the oscilations. 

**Momentum** helps the SGD accelerate the speed in the direction of the minima and dampen the oscilations.

```math
\newcommand{\al}[2]{\text{#1} & \hspace{1cm} #2 &} % faster lines in display equations
\newcommand{\var}[3]{#1= & \hspace{1cm} \text{#2} \hspace{-4cm} & \text{#3} &} % easy 

\begin{align*}
    \al{}{\begin{cases}
    v_{ji}^t=\lambda v_{ji}^{t-1}-\gamma\frac{\partial E}{\partial w_{ji}} \\
    w_{ij}^{t+1} = w_{ji}^t+v_{ji}^t
    \end{cases}}
\end{align*}
```

### Adaptive Moment Estimation (Adam)
This problem with regular moment defintion is that the learning rate is the same for all weights. But not all weights are the same. Adam Optimizer allows you to have learning rates with each individual weight for better learning.

vt is a moving average for each weight

```math
\newcommand{\al}[2]{\text{#1} & \hspace{1cm} #2 &} % faster lines in display equations
\newcommand{\var}[3]{#1= & \hspace{1cm} \text{#2} \hspace{-4cm} & \text{#3} &}

\begin{align*}
\al{Momentum?}{m_t=\beta_1m_{t-1}+(1-\beta_1)\left(\frac{\partial E}{\partial w_{ij}}\right)} \\ 
\al{Velocity?}{v_t = \beta_2v_{t-1} + (1-\beta_2)\left(\frac{\partial E}{\partial w_{ij}}\right)^2} \\
\\
\al{change in weights?}{w_{ji}^{t+1} = w^t_{ji} - \frac{\gamma}{\sqrt{v_t}+\epsilon}m_t}
\end{align*}
```

```python
# Pytorch Implementation
torch.optim.Adam(model.parameters(), lr=0.001)
```

---

## Learning Rate

The learning determines the size of the step that the optimizer takes during each interation of the of gradient descent. The larger the step size, the larger the change in the weights per iteration. It is important to control this learning rate in order to ensure the learning is both effcient but will not over learn and miss the minima.

The factors that should influence the learning rate are the 

1. **Type of Problem**
2. The **Optimizer**
3. The **Batch size**
    - larger size $\rightarrow$ larger rate
4. The **stage of the training**
   - The learning rate is typically reduced after a certain number of epochs
   - This can be done coresponding to different functions but is typically done in either a step decay or exponential decay

---

## Normalization

It tends to be the case that inputs are multifacited and can therefore be of different orders of magintude. Because of this, inputs may influce a model much more than other inputs despite their importance to prediction being the same. Therefore, by normalizing the input values such that the input magnitude is not a factor on the network output. The normalization must be consitent from layer to layer. The main methods used to normalize data is **Batch Normalization** and **Layer Normalization**.

### Batch Normalization

As the name implies, the **Batch Normalization** normalizes the activations batchwise per **feature** (type of input) of the examples in the batch. This is run throughout the training and is recorded as a moving average. This is then used as a inference time after training during implementation of the model. This method has a higher learning rate, regularizes the model and is less sensitive to intiatialization but is highly dependent on the batch size and is not compatiable with SGD.

<p align="center">
    <img src="/Week%203/batch_normalization.png">
    <img src="/Week%203/inference_time.png">
</p>

### Layer Normalization
Layer normalization is the oppostive of batch normalization. This normalizes the the data on a per example basis accorss all features. This means that the normalization is not dependant on the on the batch size and is much simplier to implement.

<p align="center">
    <img src="/Week%203/layer_normalization.png">
</p>

---

## Regularization

Reguarlization of a neural network is the process of prventing overfitting of the data in order to improve the accuracy of the model. This is an imparetive best practice in deep learning in order to ensure the accuacy of the models.

### Dropout

The **Dropout Method** inheriently makes the task of the neural network "harder" and therefore learn more. This is done by outright removing certain neurons to envoke the learning of more robust features. This is meant to mimic biological events in which neurons are removed do to environmental events such as a loss of limba and animals are able to learn and bounce back from this scenario.

During the **training phase**, the activations of certain neurons will be set to 0 (or dropped) with some probability $p$ of being dropped.

During the **inference phase**, the weights are multipled by $(1-p)$ in order to maintain the same distribution as in training.

<p align="center">
    <img src="/Week%203/dropout.png">
</p>

### Weight Decay (L2?)

The **Weight Decay Method** prevents the weights from growing too much causing over fitting and lowers the overall variance.

```math
\newcommand{\al}[2]{\text{#1} & \hspace{1cm} #2 &} % faster lines in display equations
\newcommand{\var}[3]{#1= & \hspace{1cm} \text{#2} \hspace{-4cm} & \text{#3} &} % easy 


\begin{align*}
\al{}{E(W,y,t) = E(W,y,t)+\frac{\alpha}{2}||W||^2_2 \rightarrow\frac{\partial E}{\partial W}=\frac{\partial E}{\partial W} + \alpha W} \\

\al{}{W^{t+1} = W^t-\gamma\left(\alpha W_t+\frac{\partial E}{\partial W}\right)}
\end{align*}
```

### Early Stopping &  Early Stopping with Patience

In each training iteration, the valiadtion loss is observed. When the validation loss starts to increase a counter is started. If the validation loss decreases, the counter is reset. The training can then be stopped at a point to minimize the validation loss.

This can be implemented with patiance, where after the counter has started, the model is run for a speficied amount of iterations (or patiance) before the training is stopped.

---

## Pytorch Implementation

> ALWAYS SET THE RANDOM SEED SO THAT THEREFORE ALL THE RANDOM NUMBERS WILL BE THE SAME IN ORDER TO OBSERVE REPRODUCABILITY THAT ARE NOT ONLY BASED ON RANDOM RESULTS
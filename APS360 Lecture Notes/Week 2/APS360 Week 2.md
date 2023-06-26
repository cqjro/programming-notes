---
title: APS360 Notes Week 2
author: Cairo Cristante
header-inclues:
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
        \(
            \newcommand{\topictitle}[1]{\noindent \textbf{\uline{#1}}} % New Topic Titles
            \newcommand{\subtopic}[1]{\noindent \uline{#1}}
            \newcommand{\sub}[1]{_{\text{#1}}} % faster text subscripts
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
            \newcommand{\image}[2]{\begin{center} \includegraphics[width= #2 cm]{#1} \end{center}}
        )/

---


# APS360 Week 2

---

## Biological Neuron Analogy

Machine Learning Neural Networks are inheritantly inspired by the design of biological neural networks. These involve the use of neurons that contain various structures that help the flow of information.

<p align="center">
    <img src="/Week%202/neuron.png">
</p>

Dentrites
: recieve information from other neruons (information input)

Cell body
: consolidates information from the dendrites (information processor)

Axon
: passes the information on to the other neurons (information output)

Synapse
: the area in which the axon and dendrites of seperate neurons connet in order to transfer infromation

A neuron will **fire** when it is recieves some form of causing it to produce some kind of output.

---

## Artificial Neuron
After observing the biological neuron and its behaviour, we can create a mathimatical model that mimics its behaviour in order to use its functionality.

\[ y=f\left(\sum_iw_ix_i+b \right)\]

```math
\newcommand{\var}[3]{#1: & \hspace{1cm} \text{#2} \hspace{-4cm} & \text{#3} &}

\begin{align*}
\var{x_i}{the input into the network (axon)}{} \\
\var{w_i}{the weight of the inputs $x_i$ that are learnt for a particular input(training)}{} \\
\var{b}{the bias are ``activation thresholds" that are learnt without the use of input}{} \\
\var{f}{the activation function that determines how the output is changed based on the weighted-inputs}{} \\
\var{y}{the output of the network}{}
\end{align*}
```

---

## Activation Functions
These are the functions that rationalize and determine the appropriate output for a given set of weighted inputs. There are many different functions that can be used in order to get a result, each with their own advantages and disadvantages. 

Early on in artificial neurons, sign functions, also known as Heaviside or Unit Step Functions where used to define the decision boundary.
```math
f(x)=\text{sign}(x) \\

f(x)= \begin{cases} 
      0, & x<0 \\
      1, & x\geq 0
   \end{cases}
```

The problem with these models is that they are not continous, and are therefore not differentiable. Because of this, these models are unable to learn because methods of optimization such as gradient decent are not applicable. Thus activation functions must be of a continous nature.

### Linear Activation Function
This is achieved by multipling the input of $\sum_i w_ix_i+b$ input by a constant, which is typically 1. The activation function determines the output of the network based on its position relative to the **Decison Boundry**. If the output is to the right of the line this is a "positive" output, meaning for something like a dog image detector that a dog was detected. If the output is to the right of the line, that means that the output is a "negative" output, meaning a dog is not detected. The function $y=x\cdot w+b$ is a generalized function for any n-dimension space known as a **Hyperplane** which splits the n-dimensional inputs into two.
\[ y=\sum_i w_ix_i+b\]
```math
\text{Generalized Function:} \hspace{0.5cm} y = x\cdot w + b
```

<p align="center">
    <img src="/Week%202/linear.png">
</p>

Most real world data does **NOT** have a linear relationships and cannot be seperated by a straight line. Instead, non-linear activation functions or  a series of non-linear transformations typically used in order to seperate/categorize the data.
<p align="center">
    <img src="/Week%202/non-linear.png">
</p>

### Sigmoid Activation Function
Sigmoid functions used to be very common because they are smooth an differentiable in addition to having a range between [-1, 1] or [0, 1]. There are a lot of different sigmoid functions however the most commonly used ones are the **Hyperbolic Tanget** and the **Logistic Function**.
<p align="center">
    <img src="/Week%202/tanh.png" width="400">
    <img src="/Week%202/sigmoid.png" width="400">
</p>

```math
\newcommand{\al}[2]{\text{#1} & \hspace{1cm} #2 &}

\begin{align*}
    \al{Hyperbolic Tangent:}{f(x) = \text{tanh}(x)} \\
    \al{Logistic Function:}{f(x) = \frac{1}{1+e^{-x}}}
\end{align*}
```

Despite the advantages of these functions, their derivatives have only a very small area in which the gradient is non-zero. $\displaystyle \lim_{x\to\infty}\nabla f$ or $\displaystyle \lim_{x\to-\infty}\nabla f$ both approach 0 therefore making any attempt at learning almost disappear because of gradients that are effectively 0.
<p align="center">
    <img src="/Week%202/sigmoidderivative.png" width="400">
</p>

### Rectified Linear Unit (ReLU)
This is the modern activation method used today. It is effectively the maximum between 0 and whatever is inputed into the system, but can be expressed piecewise. The reason these functions are used is because they are very easy to differentiate, being equal to either 0 or 1.

```math
\newcommand{\al}[2]{\text{#1} & \hspace{1cm} #2 &}

\begin{align*}
    \al{ReLU Function:}{\text{ReLU}(x) = \text{max}(0, x) } \\
    \al{LeakyReLU:}{\text{LeakyReLU}(x) = \begin{cases} 
      x, & x\geq0 \\
      -ax, & \text{otherwise}
   \end{cases}} \\
    \al{Parametric ReLU}{\text{PReLU}(x) = \begin{cases} 
      x, & x\geq0 \\
      ax, & \text{otherwise}
   \end{cases}}
\end{align*}
```

#### Continous ReLU Approximations
In order to make these functions continous for differentiation and consideration of all cases, continous approximations of the ReLU functions are also used. These work just as well or better than the orignal functions.

```math
\newcommand{\al}[2]{\text{#1} & \hspace{1cm} #2 &}

\begin{align*}
    \al{Swish/Sigmoid ReLU:}{\text{SiLU(x)}=x\cdot\sigma(x)=\frac{x}{1+e^{-x}}} \\
    \al{SoftPlus:}{\text{SoftPlus(x)}=\frac{1}{\beta}\log(1+e^{\beta x})}
\end{align*}
```

 >For the purposes of our course, 95% of applications will work fine in Pytorch using the default ReLU function.

---

## Traning a Neural Network
When we train a neural network, we are optimizing the weights of the inputs in order to minimize the error of the output.
1. **Validated data** in which the output of the respective $x$ input is known, is used to get an output from the model
2. The error of the prediction is then compueted in a **Loss Function** which compares the prediction to the ground true. 
3. The weights are then going to be adjusted in order to minimize the loss function using optimization methods, specifically **Gradient Descent**. 
4. This process is they repeated for all data in the training set until a **resonable error margin** is achieved.

### Softmax Function
The problem that is run into when using a neural network of multiple neurons with their respective weights is that the outputs can be on orders that are in no way compariable. These uncomparable output values are called **logits**. In order to compare these values, we use something called a **Softmax Function** which normalizes the logits into a categorical probability distribution over all posisble categories. This means that the outputs will then all be formatted as a probability of the outcome denoted by a value between 0 and 1.

```math
\newcommand{\al}[2]{\text{#1} & \hspace{1cm} #2 &}

\begin{align*}
    \text{Softmax}(x)_i=p_i=\frac{e^{x_i}}{\displaystyle\sum_{i=1}^{K}e^{x_i}}, \hspace{1cm} \text{where $0 \leq p_i \leq 1$ and $\displaystyle \sum_{i=1}^{K} p_i = 1$}
\end{align*}
```

### One-hot Encoding
One hot encoding is the method used to handle the **Ground Truth** data. This is done by mapping the categorical data to be represented as a matrix. For example, if there are three neurons in a network for identifiying animals, there are three categorical outputs. They probability distributions for the validated data would contain one category with a value of 1 (100% probability) and two with a value of 0 (0% probabilities).

```math
\begin{bmatrix}
\text{Cat} \\
\text{Dog} \\
\text{Sheep}
\end{bmatrix} =
\begin{bmatrix}
(1 & 0 & 0) \\
(0 & 1 & 0) \\
(0 & 0 & 1)
\end{bmatrix}
```

### Loss Functions
The loss function computes the error of perdictions on a case by case basis, however to train the entire model, we compuete the average error over all training samples.

```math 
\newcommand{\al}[2]{\text{#1} & \hspace{1cm} #2 &}

\begin{align*}
    E=\text{Loss}(y, t) \\
\end{align*}

\newline

\text{where $y$ is the neural network output and $t$ is the known output}
```

#### Loss Function 1: Mean Squared Error (MSE)
This is generally used for regression problems in statistically analysis.

```math
\newcommand{\al}[2]{\text{#1} & \hspace{1cm} #2 &}

\begin{align*}
\text{MSE} = \frac1N\sum_{n=1}^N(y_n-t_n)^2
\end{align*}
```

#### Loss Function 2: Cross Entropy (CE)
This is used for classification problems. The reason this is used is because it provides a larger penalty for predicting an outcome incorrectly due to the log term on around the predicted value. For smaller values of $y_{n, k}$ the larger the error value and will then be adjusted more drastically. This forumla can be used to $\log_{10}$ but historically is used with $\log_{2}$ because this formula comes from information theory

```math
\newcommand{\al}[2]{\text{#1} & \hspace{1cm} #2 &}

\begin{align*}
    \text{CE} = -\frac1N\sum_{n=1}^N\sum_{k=1}^Kt_{n, k}\log(y_{n, k})
\end{align*}
```

##### Binary Cross Entropy (BCE)
This is similar to the Cross entropy function but is used in the cases in which there are only two categories. This is analogous to a **Bernouli distribution** in which the two categories represent the probability of **success** ($p$) and **failure** $(1-p)$.

```math
\newcommand{\al}[2]{\text{#1} & \hspace{1cm} #2 &}

\begin{align*}
    \text{BCE} = -\frac1N\sum_{n=1}^N\left[t_n\log(y_n)+(1-t_n)\log(y_n)\right]
\end{align*}
```


---
### Single Layer NN Training: Delta Rule
When training data we want to observe the change in error with respect to the change in weight, aka the derivative of the loss function. We can then optimize (minimize) the error by finding the minima by setting the dertivative of the loss function eqaul to zero and determining the weighs that minimize the error. This is done using numerical methods for optimization, most commonly, gradient decent becausse it is less computationally expensive than other methods that require you to take the inverse of a matrix.

```math
\newcommand{\al}[2]{\text{#1} & \hspace{1cm} #2 &}

\frac{\text{d}E}{\text{d}w_{p}} =\left(\frac{\text{d}E}{\text{d}y}\right)\left(\frac{\text{d}y}{\text{d}a}\right)\left(\frac{\text{d}y}{\text{d}w_p}\right), \hspace{1cm} \text{where $a = \sum_p (w_px_p + b)$ }

\newline

\begin{align*}
    \al{Single Variable Gradient Descent:}{w_{ij}^{k+1}=w_{ij}^k - \gamma\frac{\partial E}{\partial w_{ij}}}\\
    \al{Multiple Variable Gradient Descent:}{W^{k+1}=W^k-\gamma\nabla E}\\
\end{align*}

\newline

\gamma = \text{Learning Rate (Step Size)}

\newline

W = \begin{bmatrix}
        w_{i1} \\
        w_{i2} \\
        \vdots \\
        w_{ij}
    \end{bmatrix}, \hspace{1cm} \text{where $i$ is the category number and $j$ is the input number}
```

These will be trained based on the which category is the group truth and thus the weights will be updated one category at a time.

---

### Multi-layer NN Training
For most problems, having only one decion boundry aka a single layer of neurons is not suffcient. For example, XOR problems require two descion bounderies. 
<p align="center">
    <img src="/Week%202/XOR.png">
</p>

In order to solve problems that require multiple decision boundries, a hidden layer that lies in between the input and the neuron is required. In fact any neural network with at least one hidden layer is a **Universal Function Approximator** and can approximate any function. The weights between the input layer and the hidden layer, and the hidden layer and the output layer are all optimized using gradient decesnt in order to induce learning of the model. The movement of weighting optimization from the weights at the output working towards the input is called **Back Passing**, and the opposite of which is called **Forward Pass**.

The idea behind the muliple layer is based on the fact that the input data will not be linearly seperable. What the hidden layers allow the network to do is learn how to project the inputs into different dimensions where they are linearly seperable in order to consolidate them into a single output.

<p align="center">
    <img src="/Week%202/hiddenlayer.png">
    <img src="/Week%202/sepeartion.png">
</p>

---

## Network Architecture

Feed-Forward Networks
: Networks in which information flows forward from one layer to a later layer from the input to the output

Fully-Connected Networks
: Neurons between adjacent layers are fully connected

Number of Layers
: Number of Hidden Layers + Output Layer



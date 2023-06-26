# Week 1 Notes:

## Project Tips:
1. Choose project with interest and knowledge in
2. Has access to suffcient data (not too much data)
   1. Papers with codes
   2. Kaggle
   3. Research papers
   4. Other publically avaliable datasets 
3. If taking peoples projects much identifiy that the project has been pulled from that section

## Approches to AI
How can we make computers intellegence?

### Symbolic Approach
- Aims to transfer the knowledge that a human would have to computers
- The computer can then use that knowledge to simulate intellegence 
- Create well defined symbols (cats, dogs) and then create well defined algotrithms that use this information
- **This was not applicable to real-world applications**

### Connectionist Approach
- Provides data into computer and have an algorithm and learns and forms connections based on this data

## Sub-Fields of Machine Learning
- Machine Learning
- Computer Vision
- Natural Language Processing

**All of thsse are now dominated by Deep Learning***

### Machine Learning
We focus on mearhcin learning, specifically Deep Learning, aka Neural Networks. This involves the computer learning from data and positive examples so it can then generalize for data it has not seen.

We need machine learning because these will be able to more accurately determine if there is a goat in the image in cases that you have not considered when hard coding an example.

Additionally, for things like images, theese are 30k by 30k matrcies of pixels that a person cannot possible adjust for.


## Formal Defintion of ML
A program that learns from **experience (E)** with respect to some class of **task (T)** and some measure of **performance (P)**. The performace at **T**, as measured by **P**, improves with experience **E**.

## Deep Learning
Deep Learning is the the use of Artifical Neural Networks (ANN). This has the highest perforamce per data provided.

### Formal Definition
This is a subset of machine learning that allows **multiple levels of representation** obtained by composition simple **non-linear modules** that transform the prepresentation at one level **(starting with raw input)** into a representation at a higher **abstract level**.

If the model is linear, it is likely that this model will not learn very well.

## Types of Machine Learning

### Supervised Learning
- Regression or Categortical Classification
  - Ex. Image Classification of Animals in an Image
  - Ex. Polynomial Plot fitting in Regression
- Requires data on the ground-truth label/output

### Unsupervised Learning
- Self-supervised Learning, Semi-supervised Learning
- Requries observations without human annotation?

### Reinforcement Learning
- Sparce rewards from environment (Ex. win/ lose)
  - Think Code Bullet youtube videos for training a player to beat a level
  - Encourage by points, distance tranveled, winning, etc
- Actions will effect the environment

## Generalization: Overfitting vs Underfitting
Machine learning models will be trained on training data. As the data becomes more complex with the training data, the error will become less and less. 

When validated with a validation data set, there will be a point in which the complexity becomes so complex that it will "perfectly" match the data points, which then creates overfitting error. 

Underfitting is when the models is so simple, that even with training data, the model cannot make an accurate prediction and is thus underfitting.

>if error increases with validation data => overfitting
>if error decreases with validation data and then remains constant => underfitting



# DO NOT USE THE TESTING DATA FROM A RESEARCH PAPER AND ONLY FEED THE MODEL THE TESTING DATA ONCE
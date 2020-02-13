# Bayesian-project
This repository contains all the code made by us during the project on "Bayesian deep learning" for the course in "Bayesian statistics".
Disclaimer: Everything should compile, but most of the content consists of experiments that failed to perform as hoped, but that is.
The repository contains:

-A moderately successful attempt at implementing Radford Neal Hamiltonian Montecarlo based Neural Network using Pystan, concerning these files: 
*The file "Neal_initialize" should be run first as it compiles the model and dumps it on the directory (to set manually) through pikel. 
*The file "Neal_Network" contains a class, which unpikels the model upon initialization, and contains a method to train (through pystan) given a dataset, to test using the weights it has sampled, to save the weights, and to measure the goodness of fit.
*The file "Neal_1D" is a main implementing a 1 dimensional regression problem using the network and plotting the results.
*The file "robot_arm_problem" implements the multidimensional robot arm problem as described in Neal's book, and displays in the form of an animation of the arm 

-A failed attempt to implement Neural Processes using Pytorch concerning these files:
*"Attentive_Neural_Processes" contains the implementation of the eponymous algorythm which proved in the Garnelo implementation to be the best type of NP. It fails, though, in many ways, being way slower than the Deepmind implementation was, and way less efficient, when it works at all. The structure of the algorythm upon repeated checks seems to be the same as that of the blueprint, so the fallacy probably falls down to our poor experience with Pytorch, which may have resulted in us unwittingly implementing some uncorrect procedure (such as with the optimizer or the custom loss function).
*"ANP_main" tests the class on a simple regression problem for a family of random sinusoidal functions, with underwhelming results

-A spectacularly bad attempt at having one or both the networks play the italian game of cards of Briscola, involving these files:
*"briscola_tree" implements a decision tree to play briscola, it succeeds at playing at a basic level, every turn it runs simulations where it draws randomly a hand for the opponent from the cards still at play and plays additional hypothetic turns according to the parameter "depth".
*"Neal_briscola_train" trains Neal's network based on the matches between two trees of chosen parameters, each card is represented by two features as "number of cards still in the deck stronger than this card"/"number of cards still in the deck" and "value of the card"/"points still available" which we thought would synthesize aptly the information required, but the NN fails to win a single match, so maybe it just isn't cut out for reinforcement learning.
*"briscola" implements a charming user interface to test the game, but given the failures, all it does test are the decision trees.
*"cards" is the folder with the pictures of the cards the directory where you set it is to be written at the start of the "briscola" file


ENVIRONMENT
all files were run on a Conda environment using Python 3.6.9., As an editor Pycharm was used.
All libraries were installed through Pycharm by File-Settings-Project-Project interpreter and clicking on the plus icon to browse for them, except for a couple which have specific instructions.
Here's a list:
-Numpy
-Matplotlib
-Pytorch
-Tensorflow version 1.14: To install it, from Anaconda powershell "conda activate name_environment"  "pip install tensorflow==1.14"
-Pygame: This one too to download using pip from the Anaconda Powershell as explained for Tensorflow.
-Pystan: follow instructions at "https://pystan.readthedocs.io/en/latest/getting_started.html", especially download all the libraries mentioned (for Windows libpython and m2w64-toolchain)




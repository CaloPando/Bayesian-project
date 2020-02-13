# Bayesian-project
This repository contains all the code made by us during the project on "Bayesian deep learning" for the course in "Bayesian statistics".
The repository contains:


1. A moderately successful attempt at implementing Radford Neal Hamiltonian Montecarlo based Neural Network using Pystan, concerning these files:  <br/> 

-*The file "Neal_initialize" should be run first as it compiles the model and dumps it on the directory (to set manually)                      through pikel.*  <br/>

-*The file "Neal_Network" contains a class, which unpikels the model upon initialization, and contains a method to train (through pystan) given a dataset, to test using the weights it has sampled, to save the weights, and to measure the goodness of fit. Disclaimer: The model is really slow, a clear sign that Stan isn't optimized for these type of models, we persisted with the approach anyway for intellectual purposes.* <br/>

-*The file "Neal_1D" is a main implementing a 1 dimensional regression problem using the network and plotting the results.* <\br>

-*The file "robot_arm_problem" implements the multidimensional robot arm problem as described in Neal's book, and displays in the form of an animation of the arm* <br/>


-An implementation of Neural Processes using Pytorch concerning these files:

-*"Attentive_Neural_Processes" contains the implementation of the eponymous algorythm which proved in the Garnelo implementation to be the best type of NP. It is however twice slower than Deepmind's implementation. The structure of the algorythm upon repeated checks seems to be the same as that of the blueprint, so the fallacy probably falls down to our poor experience with Pytorch, which may have resulted in us unwittingly implementing some incorrect procedure (especially with the custom loss function).* <br/>

-*"ANP_main" tests the class on a simple regression problem for a family of random sinusoidal functions* <br/>

-An attempt at having one or both the networks play the italian game of cards of Briscola, involving these files (COMING SOON): <br/>
*"briscola_tree" implements a decision tree to play briscola, it succeeds at playing at a basic level, every turn it runs simulations where it draws randomly a hand for the opponent from the cards still at play and plays additional hypothetic turns according to the parameter "depth".* <br/>
*"Neal_briscola_train" trains Neal's network based on the matches between two trees of chosen parameters, each card is represented by two features as "number of cards still in the deck stronger than this card"/"number of cards still in the deck" and "value of the card"/"points still available" which we thought would synthesize aptly the information required, the NN does win some matches, but due to time constraints we couldn't focus on improving, plus it is still up for debate wether this experimental NN is actually cut out for reinforcement learning alltogether.* <br/>
*"briscola" implements a charming user interface to test the AI.* <br/>
*"cards" is the folder with the pictures of the cards the directory where you set it is to be written at the start of the "briscola" file* <br/>


ENVIRONMENT:
all files were run on a Conda environment using Python 3.6.9., As an editor, Pycharm was used.
All libraries were installed through Pycharm by File-Settings-Project-Project_interpreter and clicking on the plus icon to browse for them, except for a couple which have specific instructions.
Here's a list:
-Numpy
-Matplotlib
-Pytorch
-Tensorflow version 1.14: To install it, from Anaconda powershell "conda activate name_environment"  "pip install tensorflow==1.14"
-Pygame: This one too to download using pip from the Anaconda Powershell as explained for Tensorflow.
-Pystan: follow instructions at "https://pystan.readthedocs.io/en/latest/getting_started.html", especially download all the libraries mentioned (for Windows libpython and m2w64-toolchain)

Also used was ffmpeg to make an animation in "robot_arm_problem", it is downloadable at this link: "https://www.ffmpeg.org/download.html"
Beware that almost all files require to manually set the directories where to save the results




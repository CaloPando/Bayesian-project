# Bayesian-project
This repository contains all the code made by us during the project on "Bayesian deep learning" for the course in "Bayesian statistics".
The repository contains:


1. A moderately successful attempt at implementing Radford Neal Hamiltonian Montecarlo based Neural Network using Pystan, concerning these files:  <br/> 

    -*The file "Neal_initialize" should be run first as it compiles the model and dumps it on the directory (to set manually)                      through pikel.*  <br/>

    -*The file "Neal_Network" contains a class, which unpikels the model upon initialization, and contains a method to train (through pystan) given a dataset, to test using the weights it has sampled, to save the weights, and to measure the goodness of fit. Disclaimer: The model is really slow, a clear sign that Stan isn't optimized for these type of models, we persisted with the approach anyway for intellectual purposes.* <br/>

    -*The file "Neal_1D" is a main implementing a 1 dimensional regression problem using the network and plotting the results.* <br/>

    -*The file "robot_arm_problem" implements the multidimensional robot arm problem as described in Neal's book, and displays in the form of an animation of the arm* <br/>
    
    -*The file "Neal_classification" implements a simple classification problem where random points are generated on a plane and those that fall into a circle at the center are labelled as positives. To succeed the adaptive delta has to be lowered. This applies to anytime the network has to fit a steep function (in this case a step).*<br/>


2. An attempt at implementing of Neural Processes using Pytorch concerning these files:

      -*"Attentive_Neural_Processes" contains the implementation of the eponymous algorythm which proved in the Garnelo implementation to be the best type of NP. It is however twice slower than Deepmind's implementation, and it yields subpar results and only by tweaking the cross-attention. The structure of the algorythm upon repeated checks seems to be the same as that of the blueprint, so the fault probably lies on our poor experience with Pytorch, which may have resulted in us unwittingly implementing some incorrect procedure (especially with the custom loss function).* <br/>

      -*"ANP_main" tests the class on a simple regression problem for a family of random sinusoidal functions, eventually it leanrs to approximate the functions* <br/>

3. Deepmind's implementation, utilized for most of the tests as it was more efficient, WE DON'T OWN ANYTHING ALL CREDITS GO TO THEM, we just modified the input to feed it the aforementioned example, and the MNIST, even if this didn't converge for us.<br/>
Theses files are involved:<br/>
      -*"true_neural_processes" contains the Garnelo implementation plus two functions we implemented to feed the examples of the family of random sinusoids and the same classification problem of "Neal_classification" only with random radius *<br/>
      -*"neural_processes_main" implements the example of the random sinusoids*<br/>
      -*"neural_processes_mnist" feeds the MNIST dataset to the Neural Processes*<br/>
      -*"np_classification_main" implements the classification example*<br/>
      

4. A framework for training Neal's Network to play the italian game of Briscola (COMING SOON): <br/>

      -*"briscola_tree" implements a decision tree to play briscola, it succeeds at playing at a basic level, every turn it runs simulations where it draws randomly a hand for the opponent from the cards still at play and plays additional hypothetic turns according to the parameter "depth".* <br/>

      -*"Neal_briscola_train" trains Neal's network based on the matches between two trees of chosen parameters, each card is represented by two features as "number of cards still in the deck stronger than this card"/"number of cards still in the deck" and "value of the card"/"points still available" which we thought would synthesize aptly the information required, the NN does win some matches, but due to time constraints we couldn't focus on properly training it. the framework could be adapted to test other more conventional Deep Learning methods though (ours was an experiment after all).* <br/>

      -*"briscola" implements a charming user interface to test the AI.* <br/>

      -*"cards" is the folder with the pictures of the cards. The directory where you put it is to be set at the start of the "briscola" file* <br/>
      
In the "images" folder are some plots, in the "trained_models" some trained models that can be loaded.


ENVIRONMENT:<br/> 
---------------
all files were run on a Conda environment using Python 3.6.9.<br/> 
As an editor, Pycharm was used.
All libraries were installed through Pycharm by File-Settings-Project-Project_interpreter and clicking on the plus icon to browse for them, except for a couple which have specific instructions.<br/> 
Here's a list:<br/> 

-*Numpy*<br/> 
-*Matplotlib*<br/> 
-*Pytorch*<br/> 
-*Tensorflow version 1.14: To install it, from Anaconda powershell "conda activate name_environment"  "pip install tensorflow==1.14"*<br/> 
-*Pygame: This one too to download using pip from the Anaconda Powershell as explained for Tensorflow.*<br/> 
-*Pystan: follow instructions at "https://pystan.readthedocs.io/en/latest/getting_started.html", especially download all the libraries mentioned (for Windows libpython and m2w64-toolchain)*<br/> 
-*Also used was ffmpeg to save an animation in "robot_arm_problem", it is downloadable at this link: "https://www.ffmpeg.org/download.html"*<br/> 

Beware that almost all files require to manually set the directories where to save the results<br/> 


RESULTS:<br/>
-------------
Our Stan based Neural Network succeeds at approximating functions where it has training points, and the variance swells where it has none to indicate its uncertainty.

![alt text](https://github.com/CaloPando/Bayesian-project/blob/master/images/train_on_range2.png)

It has more difficulty estimating random noise as it just regularizes the function. Therefore it estimates epistemic uncertainty more than that due to measurements (when the latter is excessive, that is)

![alt_text](https://github.com/CaloPando/Bayesian-project/blob/master/images/regression_perturbed_high_noise.png)

Sudden climbs constitute a challenge

![alt_text](https://github.com/CaloPando/Bayesian-project/blob/master/images/delta_099.png)

But lowering the adaptive delta to 0.8 it gets the job done

![alt_text](https://github.com/CaloPando/Bayesian-project/blob/master/images/xcubo_delta08.png)

Likewise the delta needs to be lowered further to 0.5, to be able to perform classification (because of the jump discontinuity from 0 to 1).

![alt_text](https://github.com/CaloPando/Bayesian-project/blob/master/images/Neal_classification.png)

It also performs nicely at a multidimensional problem, such as the robot arm problem, which was present in Neal's book:<br/>
The network was trained on 20 points randomly sampled, the blue ellipse is the variance and it increases when close to the edges of the training interval as expected. It's the trained model "robot_arm_sloppy" (also "robot_arm_perfect" is trained with 80 points). 

![alt_text](https://github.com/CaloPando/Bayesian-project/blob/master/images/frame_robot_arm.PNG)




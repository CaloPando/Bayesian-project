# Bayesian-project
This repository contains all the code made by us during the project on "Bayesian deep learning" for the course in "Bayesian statistics".
The repository contains:


1. A moderately successful attempt at implementing Radford Neal Hamiltonian Montecarlo based Neural Network using Pystan, concerning these files:  <br/> 

    -*The file "Neal_init" should be run first as it compiles the model and dumps it on the directory (to set manually)                      through pikel.*  <br/>

    -*The file "Neal_NN" contains a class, which unpikels the model upon initialization, and contains a method to train (through pystan) given a dataset, to test using the weights it has sampled, to save the weights, and to measure the goodness of fit. Disclaimer: The model is really slow, a clear sign that Stan isn't optimized for these type of models, we persisted with the approach anyway for intellectual purposes.* <br/>

    -*The file "Neal_1D_regression" is a main implementing a 1 dimensional regression problem using the network and plotting the results.* <br/>

    -*The file "robot_arm_problem" implements the multidimensional robot arm problem as described in Neal's book, and displays in the form of an animation of the arm* <br/>
    
    -*The file "Neal_classification" implements a simple classification problem where random points are generated on a plane and those that fall into a circle at the center are labelled as positives. To succeed the adaptive delta has to be lowered. This applies to anytime the network has to fit a steep function (in this case a step).*<br/>


2. An attempt at implementing of Neural Processes using Pytorch concerning these files:

      -*"Attentive_Neural_Processes" contains the implementation of the eponymous algorythm which proved in the Garnelo implementation to be the best type of NP. It is however twice slower than Deepmind's implementation, and it yields subpar results and only by tweaking the cross-attention. The structure of the algorythm upon repeated checks seems to be the same as that of the blueprint, so the fault probably lies on our poor experience with Pytorch, which may have resulted in us unwittingly implementing some incorrect procedure (especially with the custom loss function).* <br/>

      -*"NP_torch_main" tests the class on a simple regression problem for a family of random sinusoidal functions, eventually it leanrs to approximate the functions* <br/>

3. Deepmind's implementation, utilized for most of the tests as it was more efficient, WE DON'T OWN ANYTHING ALL CREDITS GO TO THEM, we just modified the input to feed it a couple of example, and the MNIST, and did other minor changes, even if this didn't converge for us.<br/>
the original can be found at "https://github.com/deepmind/neural-processes/blob/master/attentive_neural_process.ipynb".<br/>
Theses files are involved:<br/>
      -*"true_neural_processes" contains the Garnelo implementation plus two functions we implemented to feed the examples of the family of random sinusoids and the same classification problem of "Neal_classification" only with random radius *<br/>
      -*"neural_processes_main" implements the example of the random sinusoids*<br/>
      -*"neural_processes_mnist" feeds the MNIST dataset to the Neural Processes*<br/>
      -*"np_classification_main" implements the classification example*<br/>
      

4. A framework for training Neal's Network to play the italian game of Briscola (COMING SOON): <br/>

      -*"briscola_tree" implements a decision tree to play briscola, it succeeds at playing at a basic level, every turn it runs simulations where it draws randomly a hand for the opponent from the cards still at play and plays additional hypothetic turns according to the parameter "depth".* <br/>

      -*"Neal_briscola" trains Neal's network based on the matches between two trees of chosen parameters, each card is represented by two features as "number of cards still in the deck stronger than this card"/"number of cards still in the deck" and "value of the card"/"points still available" which we thought would synthesize aptly the information required.<br/> 
      Two networks are trained, one for palying first , and one for playing in response (as this one takes 2 cards input, 4 features), the latter is trickier, as the Network needs to know wether he'll win the hand or not by playing a card, so if he'd lose it the features of the two cards played are multiplied by -1. <br/> 
      It's an admittedly conceited setup but we couldn't afford to have two many features or it would have been too slow, and we believe that with more training and a more powerful algorythm very good results could have been achieved, but again the goal was just to show the potential of the model, not the briscola per se.<br/>
      The NN does win some matches, but due to time constraints we couldn't focus on properly training it. the framework could be adapted to test other more conventional Deep Learning methods though (ours was an experiment after all).* <br/>

      -*"briscola" implements a charming user interface to test the AI.* <br/>
      
      -*"briscola_features" contains the functions that create the features for training*<br/>

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

The Network has no problem also dealing with fast increasing functions.

![alt_text](https://github.com/CaloPando/Bayesian-project/blob/master/images/xcubo_delta08.png)

This also applies to classification problems where it has to fit a step function in a circle (model trained in "trained_models").

![alt_text](https://github.com/CaloPando/Bayesian-project/blob/master/images/Neal_classification.png)

It also performs nicely at a multidimensional problem, such as the robot arm problem, which was present in Neal's book:<br/>
The network was trained on 20 points randomly sampled, the blue ellipse is the variance and it increases when close to the edges of the training interval as expected. It's the trained model "robot_arm_sloppy" (also "robot_arm_perfect" is trained with 80 points). 

![alt_text](https://github.com/CaloPando/Bayesian-project/blob/master/images/frame_robot_arm.PNG)



We also laid the foundations for the briscola problem, "briscola_bot" plays a very superficial briscola, it managed to collect a winning streak against the decision tree at a low simulations level. It was supposed to be the first step but time constraints didn't allow us to improve it by the deadline. Moreover the procedure of utilizing the sample mean as the parameter for the priors of the subsequent samples doesn't work really well. These Stan based networks work better if run on the whole dataset instead than updated gradually on chunks of it.<br/>
The "briscola_bot" was just trained on 20 matches with just 80 iterations and is kind of promising, if often erratic, which points out that this framework could be used to test more optimized versions of NNs with better results in far less time.

![alt_text](https://github.com/CaloPando/Bayesian-project/blob/master/images/briscola_UI.PNG)

As for Neural Processes, the use of attention improves the performance dramatically: The first plot is from non attentive after 10000 iterations, the second one is from ANP after just 2000

![alt_text](https://github.com/CaloPando/Bayesian-project/blob/master/images/NP_nonattentive_10000.png)

![alt_text](https://github.com/CaloPando/Bayesian-project/blob/master/images/ANP_2000.png)






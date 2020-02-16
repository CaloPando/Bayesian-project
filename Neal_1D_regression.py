import pystan
import numpy as np
from math import pi
import matplotlib.pyplot as plt
from Neal_NN import NN
from scipy.stats import norm
'''
#This script implements a simple 1D regression problem and plots the result
the number of epochs is se to 1. By running multiple loops the model uses the sample mean
as mean of the priors for the Gaussians in the next sampling sessions, a procedure that has given mixed results,
but was an attempt to have the weights update themselves gradually 
'''

num_iterations=150
num_warmup=100
num_train=60
D = 100
I = 1
O = 1
L = 2
H = 4
x = np.linspace(0, 2 * pi, D)
'''''
With centered=False the uncenterd model is run, this was suggested by Stan documentation as an improvement on some models
but it doesn't change much in our case
'''
model=NN(L=L, H=H, I=I, O=O,centered=True)

y=np.exp(np.sin(x))+np.random.normal(0,0.05,D)

#The model accepts an input of dimension [x_dim,dataset_dim], so I add 1 dimension
x= np.expand_dims(x, 0)
y = np.expand_dims(y, 0)


num_epochs=1
for epoch in range(num_epochs):
    #This line is necessary because there is apparently a flaw with how python multiprocessing was implemented in Windows
    #Without, it is impossible to run multiple cores
    if __name__ == '__main__':
        context = np.random.choice(list(range(D)), num_train)
        context.sort()
        model.train(x=x[:,context],y=y[:,context],iter=num_iterations, warmup=num_warmup,max_treedepth=10, thin=2,chains=4,cores=4)
        #model.save_parameters('network_5000_20')
        print(model.fit)
        mu, sigma = model.predict(x_pred=x)
        
        #This computes the log likelihood
        prob = 0
        for d in range(D):
            prob += np.log(norm(mu[0, d], sigma[0, d]).pdf(y[0, d])) 
        prob=prob/D
        print("Log likelihood:%f" % prob)
        
        #This is an estimate of the std of the random noise of the original function, we notice, that while good at capturing
        #epistemic uncertainty, it has a regularizing effect, so tends to disregard high levels of noise and fit the real function
        print('estimated standard deviation ' + str(np.mean(sigma[0,:])))
        
        '''
        Notice the quantiles of the normal, no Bonferroni correction was applied, Deepmind's code of Neural Processes
        disregards them completely and just adds the variance 
        '''
        plt.fill_between(x[0, :], mu[0,:] - 1.96 * sigma[0,:], mu[0,:] + 1.96 * sigma[0,:], facecolor=(0, 1, 1))
        context_plot, =plt.plot(x[0, context], y[0, context], 'ro')
        prediction_plot, =plt.plot(x[0, :], mu[0,:], 'b')
        real_plot, =plt.plot(x[0, :], y[0, :], 'r')
        plt.legend([context_plot, prediction_plot, real_plot], ["context", "Prediction", "Real"])
        plt.show()

        plt.show()





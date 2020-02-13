import pystan
import numpy as np
from math import sin
from math import exp
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

y = []

for i in range(len(x)):
    y.append(exp(sin(x[i])) + np.random.normal(0, 0.05))

y = np.asarray(y)
x = np.asarray(x)

y=np.exp(sin(x))+np.random.normal(0,0.05,D)

#The model accepts an input of dimension [x_dim,dataset_dim], so I add 1 dimension
x= np.expand_dims(x, 0)
y = np.expand_dims(y, 0)


num_epochs=1
for epoch in range(num_epochs):
    #This line is necessary because there is apparently a flaw with how python multiprocessing was implemented in Windows
    #Without it, is impossible to run multiple cores
    if __name__ == '__main__':
        context = np.random.choice(list(range(D)), 30)
        context.sort()
        model.train(x=x[:,context],y=y[:,context],iter=50,warmup=30,max_treedepth=10, thin=2,chains=4,cores=4)
        #model.save_parameters('network_5000_20')
        print(model.fit)
        mu, sigma = model.predict(x_pred=x)
        prob = 0
        for d in range(D):
            prob += norm(mu[0, d], sigma[0, d]).pdf(y[0, d])
        prob=prob/D
        print("Average probability of real value:%f" % prob)
        output_variance = np.mean(model.output_variance)
        print(output_variance)
        '''
        Notice the quantiles of the normal, no Bonferroni correction was applied, Deepmind's code of Neural Processes
        disregarded them completely and just added the variance 
        '''
        plt.fill_between(x[0, :], mu[0,:] - 1.96 * sigma[0,:], mu[0,:] + 1.96 * sigma[0,:], facecolor=(0, 1, 1))
        plt.plot(x[0, context], y[0, context], 'ro')
        plt.plot(x[0, :], mu[0,:], 'b')
        plt.plot(x[0, :], y[0, :], 'r')
        plt.show()





from Neal_NN import NN
import numpy as np
import matplotlib.pyplot as plt

'''
This is a simulation of a classification problem where we generate a set of random points on a plane and set
those that fall inside a circle to 1, to 0 otherwise, to divide them in two categories.
The network has to than classify them. It only works by lowering the adaptive delta, and with the proper training
size. I used two outputs for the two classes for the network, instead of one because it perfomed better
'''

size_plain=2
size_total=200
size_train=50
num_iterations=200
num_warmup=150

data_x=np.random.rand(size_total,2)*size_plain-size_plain/2

data_y=np.zeros((size_total,2))
radius=0.8
def f(x):
    return np.sqrt(radius**2-x**2)

true_positives=[index for index, value in enumerate(data_x) if value[1]**2+value[0]**2<radius**2]
true_negatives=[index for index, value in enumerate(data_x) if value[1]**2+value[0]**2>radius**2]
data_y[true_positives,1]=1
data_y[true_negatives,0]=1

x = np.linspace(-radius,radius,100)
y_up = f(x)
y_down=-f(x)



train = np.random.choice(list(range(size_total)), size_train )
train.sort()
model=NN(L=3, H=4, I=2, O=2, centered=True, normalized=1)

if __name__ == '__main__':
    model.train(data_x[train,:].transpose(), data_y[train,:].transpose(), iter=num_iterations, warmup=num_warmup,
                thin=2, delta=0.5, chains=4, cores=4)
    #model.load_parameters('Neal_classification')
    mu, sigma = model.predict(x_pred=data_x.transpose())
    mu=mu.transpose()
    positives=[index for index, value in enumerate(mu) if value[1]>value[0]]
    negatives=[index for index, value in enumerate(mu) if value[1]<value[0]]
    plt.plot(x, y_up, 'k--')
    plt.plot(x, y_down, 'k--')
    plt.fill_between(x, y_down, y_up, color='#539ecd')
    positives_plot =plt.scatter(data_x[positives, 0], data_x[positives, 1], marker='x',c='k')
    negatives_plot =plt.scatter(data_x[negatives, 0], data_x[negatives, 1], marker='o',c='r')
    plt.legend([positives_plot, negatives_plot], ["1","0"])
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    plt.show()
    
    
    
    
    



















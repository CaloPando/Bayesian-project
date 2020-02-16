import torch
from Attentive_Neural_Processes import neural_process
from math import pi
import random
import matplotlib.pyplot as plt

iterations=100000
num_points=200
max_context=50
num_batches=1
test_interval=100

neur_proc=neural_process(input_size=1, output_size=1, hidden_size_encoder=[128]*4, hidden_size_decoder=[128]*4
                 ,latent_size=32, hidden_size_normal=32)

for it in range(iterations):
    A=random.uniform(-1,1)
    k=random.uniform(0, pi)
    #A=1
    #k=0

    x_target=torch.sort(torch.FloatTensor(num_batches, num_points, 1).uniform_(0, 2*pi), 1).values
    y_target = A*torch.sin(x_target+k)
    num_context=random.choice(list(range(max_context)))+1
    idx=random.sample(list(range(num_points)),num_context)
    x_context=x_target[:,idx,:]
    y_context = y_target[:, idx, :]

    if it%test_interval!=0:
        neur_proc.train(x_context, y_context, x_target, y_target)
    else:
        if it>2500:
            test_interval=200

        mu, sigma=neur_proc.test(x_context[0,:,:], y_context[0,:,:], x_target[0,:,:])

        #plot test
        x=x_target[0,:,0].detach().squeeze().numpy()
        y=y_target[0,:,0].detach().squeeze().numpy()
        m=mu[:,0].detach().squeeze().numpy()
        s=sigma[:,0].detach().squeeze().numpy()
        x_cont=x_context[0,:,0].detach().squeeze().numpy()
        y_cont=y_context[0,:,0].detach().squeeze().numpy()

        plt.plot(x, y, 'r', linewidth=2)
        plt.plot(x, m, 'b', linewidth=2)
        plt.plot(x_cont, y_cont, 'ro', markersize=10)
        plt.fill_between(
            x, m - s, m + s,
            alpha=0.2,
            facecolor='#65c9f7',
            interpolate=True)

        # Make the plot pretty
        plt.ylim([-1, 1])
        # plt.grid('off')
        ax = plt.gca()
        plt.show()














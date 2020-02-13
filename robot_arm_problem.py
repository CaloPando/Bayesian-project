from Neal_NN import NN
import numpy as np
from math import sin
from math import cos
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import rcParams

''''
To produce the animation you need to download ffmpeg at https://www.ffmpeg.org/download.html, than write here the directory 
where ffmpeg.exe is located, also write the directory where you want the animation to be saved
'''
directory_ffmpeg=r'set here the directory to ffmpeg.exe'+r'\\'
directory_animation=r'set here the directory in which to save the animation'+r'\\'

num_train=20
L1=2
L2=1.3
theta1=0
theta2=2*np.pi
phi1=-np.pi/2
phi2=np.pi/2

x1_train=np.random.uniform(low=theta1,high=theta2,size=num_train)
x2_train=np.random.uniform(low=phi1,high=phi2,size=num_train)


y1_train=[]
y2_train=[]

for i in range(num_train):
    y1_train.append(L1*cos(x1_train[i])+L2*cos(x1_train[i]+x2_train[i])+np.random.normal(0, 0.05))
    y2_train.append(L1 * sin(x1_train[i]) + L2 * sin(x1_train[i] + x2_train[i])+np.random.normal(0, 0.05))



I = 2
O = 2
L = 3
H = 8

model=NN(L=L,H=H,I=I,O=O,normalized=0)

if __name__ == '__main__':
    '''
    You can load the model already trained on the range we set (either the overtrained version
    or the undertrained sloppier version) or train a new model with a different range, comment
    and uncomment the following lines accordingly
    '''
    #model.load_parameters('robot_arm3')
    model.train(x=np.array([x1_train, x2_train]), y=np.array([y1_train, y2_train]), iter=100, warmup=50, thin=2,
              chains=4, cores=4)
    print(model.fit)

    #here test is simply done by having the arm sweep the range, different combinations can be attempted
    #to have the arm move in different ways
    num_test = 200
    x1_test = np.linspace(theta1, theta2, num_test)
    x2_test = np.linspace(phi1, phi2, num_test)

    '''
    Yes, I know, it could have been done more neatly using numpy arrays and not appending a list,
    but since it doesn't cause any problem, I left it this way
    '''
    y1_test = []
    y2_test = []

    for i in range(num_test):
        y1_test.append(L1 * cos(x1_test[i]) + L2 * cos(x2_test[i] + x1_test[i]))
        y2_test.append(L1 * sin(x1_test[i]) + L2 * sin(x2_test[i] + x1_test[i]))

    mu, sigma = model.predict(x_pred=np.asarray([x1_test, x2_test]))


    rcParams['animation.ffmpeg_path'] =directory_ffmpeg+'ffmpeg.exe'


    fig = plt.figure()
    ims = []
    plt.axis([-L1 - L2, L1 + L2, -L1 - L2, L1 + L2])

    '''
    I draw the ellipse manually, paradoxically it was the quickest way, Matplotlib is just that counterintuitive
    when it comes to making animations.
    I also get a ton of warnings because of this, but nothing which impedes the program
    '''
    for i in range(num_test):
        y0 = mu[1, i]
        y1 = sigma[1, i]
        x0 = mu[0, i]
        x1 = sigma[0, i]
        x_ellipse = np.linspace(x0 - x1, x0 + x1, 100)
        y_up = []
        y_down = []
        for d in range(100):
            y_up.append(y0 + y1 * np.sqrt(1 - (x_ellipse[d] - x0) ** 2 / x1 ** 2))
            y_down.append(y0 - y1 * np.sqrt(1 - (x_ellipse[d] - x0) ** 2 / x1 ** 2))

        ims.append(plt.plot([0, L1 * cos(x1_test[i]), y1_test[i]], [0, L1 * sin(x1_test[i]), y2_test[i]], '-o',
                            color='dimgrey', markerfacecolor='r', markersize=12, linewidth=7.0) + plt.plot(mu[0, i],
                                                                                                           mu[1, i],
                                                                                                           'og') +
                   plt.plot(x_ellipse[:], y_up[:], 'b') + plt.plot(x_ellipse[:], y_down[:], 'b'), )

    im_ani = animation.ArtistAnimation(fig, ims, interval=100, repeat_delay=3000,
                                       blit=True)

    plt.show()

    im_ani.save(directory_animation+'robot_arm.mp4', writer='ffmpeg')







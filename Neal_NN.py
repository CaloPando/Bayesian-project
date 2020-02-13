import pystan
import pickle
import numpy as np
import random
from scipy.stats import norm
#set here your directory
DIRECTORY=r'your directory'

class NN:
    def __init__(self, L, H, I, O, normalized=0, alpha=0.5, beta=1, centered=True):

        self.centered=centered
        self.L = L
        self.H = H
        self.I=I
        self.O=O
        self.N=0
        self.alpha=alpha
        self.beta=beta
        self.model = self.create_model()
        self.normalized = normalized
        self.weights=0
        self.weights_input=0
        self.weights_output=0
        self.biases=0
        self.biases_output=0
        self.offsets=0
        self.offsets_input=0
        self.output_variance = 0
        self.output_net = 0

        self.m_w_i = np.zeros((H, I))
        self.m_w=np.zeros((L-1,H,H))
        self.m_w_o = np.zeros((O, H))
        self.m_b=np.zeros((H,L))
        self.m_b_o = np.zeros((O))
        self.m_t = np.zeros((L))
        self.m_t_i = 0
        
        self.la=[]
        self.fit=[]

    def create_model(self):
        if self.centered:
            with open(DIRECTORY+r'\\'+'model.pkl', 'rb') as f:
                sm = pickle.load(f)
        else:
            print('creating uncentered model')
            with open(DIRECTORY+r'\\'+'model_uncentered', 'rb') as f:
                sm = pickle.load(f)
        return sm

    def train(self, x, y, iter=100, warmup=50, thin=5, delta=0.99, max_treedepth=10, chains=1, cores=1):
        #This is the training method it takes in input the training query and the
        #and the training target, also, parameters concerning the sampler can be set on call
        D=x.shape[-1]
        train_data = {'x': x,
                      'y': y,
                      'D': D, 'L': self.L, 'H': self.H, 'I': self.I, 'O': self.O, 'alpha': self.alpha, 'beta': self.beta, 'm_w_i':self.m_w_i,
        'm_w':self.m_w, 'm_w_o':self.m_w_o, 'm_b':self.m_b, 'm_b_o':self.m_b_o, 'm_t':self.m_t, 'm_t_i':self.m_t_i,
                      'normalized':self.normalized}

        fit = self.model.sampling(data=train_data, iter=iter, warmup=warmup, thin=thin, chains=chains, n_jobs=cores,
                          control={'max_treedepth' : max_treedepth, 'adapt_delta': delta})
        la=fit.extract(permuted=True)
        self.fit=fit
        self.la=la
        
        self.weights_input = la['w_I']
        self.weights = la['w']
        self.weights_output = la['w_O']
        self.biases = la['b']
        self.offsets = la['t']
        self.biases_output = la['b_O']
        self.offsets_input = la['t_I']
        self.output_variance = la['s_O']
        self.output_net = la['v_O']

        self.m_w=np.mean(self.weights,0)
        self.m_w_i = np.mean(self.weights_input, 0)
        self.m_w_o = np.mean(self.weights_output, 0)
        self.m_b=np.mean(self.biases,0)
        self.m_b_o = np.mean(self.biases_output, 0)
        self.m_t = np.mean(self.offsets, 0)
        self.m_t_i = np.mean(np.mean(self.offsets_input, 0))

        self.N=self.weights.shape[0]

        lpml=self.LPML(x=x,y=y)
        print("LPML:%f" %lpml)

    def save_parameters(self,filename):
        #This method seves the weights through pikle, first set here the directory
        #where you want to save them
        a={'H':self.H, 'L':self.L, 'I':self.I, 'O':self.O, 'N':self.N, 'weights':self.weights, 'weights_input':self.weights_input,
           'weights_output':self.weights_output, 'biases':self.biases, 'biases_output':self.biases_output,
           'offsets':self.offsets, 'offsets_input':self.offsets_input, 'output_variance': self.output_variance,
           'output_net': self.output_net,'normalized':self.normalized, 'fit':self.fit}

        filename=DIRECTORY+r'\\'+filename+'.pkl'
        with open(filename, 'wb') as handle:
            pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)


    def load_parameters(self,filename):
        #This is to load them
        filename=DIRECTORY+r'\\'+filename+'.pkl'
        with open(filename, 'rb') as handle:
            la = pickle.load(handle)
        self.fit=la['fit']
        self.H=la['H']
        self.L=la['L']
        self.I=la['I']
        self.O=la['O']
        self.N=la['N']
        self.normalized=la['normalized']

        self.weights_input = la['weights_input']
        self.weights = la['weights']
        self.weights_output = la['weights_output']
        self.biases = la['biases']
        self.offsets = la['offsets']
        self.biases_output = la['biases_output']
        self.offsets_input = la['offsets_input']
        self.output_variance=la['output_variance']
        self.output_net=la['output_net']

        self.m_w = np.mean(self.weights, 0)
        self.m_w_i = np.mean(self.weights_input, 0)
        self.m_w_o = np.mean(self.weights_output, 0)
        self.m_b = np.mean(self.biases, 0)
        self.m_b_o = np.mean(self.biases_output, 0)
        self.m_t = np.mean(self.offsets, 0)
        self.m_t_i = np.mean(self.offsets_input, 0)


    def predict(self,x_pred,num_samples=None):
        #If num_samples is set to None all samples are utilized for prediction
        if num_samples is None:
            num_samples=self.N

        D=x_pred.shape[-1]
        idx_samples = list(range(self.N - 1))
        random.shuffle(idx_samples)
        idx_samples = idx_samples[0:(num_samples - 1)]
        mu=[]
        sigma=[]

        for d in range(D):
            y_vec=[]
            for n in idx_samples:
                y_pred = x_pred[:, d]
                y_pred = np.tanh(np.dot(self.weights_input[n, :, :], y_pred + self.offsets_input[n]) + self.biases[n, :, 0])
                for l in range(self.L - 1):
                    y_pred = np.tanh(np.dot(self.weights[n, l, :, :], y_pred + self.offsets[n, l]) + self.biases[n, :, l + 1])
                if self.normalized:
                    y_pred = np.tanh(np.dot(self.weights_output[n, :, :], y_pred + self.offsets[n, self.L - 1]) + self.biases_output[n, :])
                else:
                    y_pred = np.dot(self.weights_output[n, :, :], y_pred + self.offsets[n, self.L - 1]) + self.biases_output[n, :]
                y_vec.append(y_pred)

            mu_temp=list(np.mean(y_vec,0))
            mu.append(mu_temp)
            sigma_temp = list(np.sqrt(np.var(y_vec,0)))
            sigma.append(sigma_temp)
        mu = np.array(mu)
        sigma = np.array(sigma)
        mu=mu.transpose()
        sigma=sigma.transpose()

        return mu, sigma


    def LPML(self,x,y):
        #this is to compute the lpml
        D=y.shape[1]
        cpos=[]
        for d in range(D):
            cpo=0
            for n in range(self.N):
                for o in range(self.O):
                    cpo+=1/norm(self.output_net[n,d,o],self.output_variance[n]).pdf(y[o,d])
            cpo=self.N/cpo
            cpos.append(cpo)

        lpml=np.sum(np.log(cpos))
        return lpml

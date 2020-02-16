import torch
from torch import nn

'''
This is a tentative implementation of Neural Processes, I don't know why it's so slow to converge,
and doesn't exhibit the interpolating qualities that the Deepmind's implementation has.
The structure should be the same, except some alterations, which we point out, without which it simply just
outputted constants 
'''


# This implements the lapalcian attention
def laplace(k, q, v, scale=1):
    num_target = q.shape[0]
    num_context = k.shape[0]
    k = k.view(1, num_context)
    k = k.repeat((num_target, 1))
    q = q.repeat((1, num_context))
    weights = -torch.abs(k - q) / scale
    '''
    If I use the softmax function like in the deepmind code it only outputs flat lines.
    The issue seems to be that this way the weights have higher values and influence more the ouput
    but I don't know why it works anyway on their code
    '''
    weights = torch.exp(weights)

    rep = torch.mm(weights, v)

    return rep


# This can be both decoder and encoder
class DeterministicEncoder(nn.Module):
    def __init__(self, input_size=1, output_size=1, hidden_size=[16, 16]):
        super().__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.input_size = input_size
        # define the layers
        self.layers = self.create_net()

    def create_net(self):
        modules = []
        modules.append(nn.Linear(self.input_size, self.hidden_size[0]))
        modules.append(nn.ReLU())
        for h in range(len(self.hidden_size) - 1):
            modules.append(nn.Linear(self.hidden_size[h], self.hidden_size[h + 1]))
            modules.append(nn.ReLU())
        # final layer
        modules.append(nn.Linear(self.hidden_size[-1], self.output_size))

        sequential = nn.Sequential(*modules)

        return sequential

    def forward(self, x_context, y_context, x_target):
        # forward pass
        context = torch.cat((x_context, y_context), -1)
        v = self.layers(context)
        q = x_target
        k = x_context
        r = laplace(k, q, v)

        return r








# This can be both decoder and encoder
class LatentEncoder(nn.Module):
    def __init__(self, input_size=1, latent_size=1, hidden_size=[16, 16]):
        super().__init__()

        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.input_size = input_size
        # define the layers
        self.layers = self.create_net()

        self.penultimate_layer=nn.Sequential(
          nn.Linear(self.latent_size, self.latent_size),
          nn.ReLU()
        )

        self.mean_layer=nn.Linear(self.latent_size, self.latent_size)

        self.sigma_layer=nn.Linear(self.latent_size, self.latent_size)

    def create_net(self):
        modules = []
        modules.append(nn.Linear(self.input_size, self.hidden_size[0]))
        modules.append(nn.ReLU())
        for h in range(len(self.hidden_size) - 1):
            modules.append(nn.Linear(self.hidden_size[h], self.hidden_size[h + 1]))
            modules.append(nn.ReLU())
        # final layer
        modules.append(nn.Linear(self.hidden_size[-1], self.latent_size))

        return nn.Sequential(*modules)

    def forward(self, x_context, y_context):
        # forward pass
        context=torch.cat((x_context, y_context), -1)
        z = self.layers(context)
        zc=torch.mean(z, 0)
        common=self.penultimate_layer(zc)
        mu=self.mean_layer(common)
        sigma = 0.1 + 0.9 * torch.sigmoid(self.sigma_layer(common))
        dist = torch.distributions.Normal(loc=mu, scale=sigma)

        return dist



# This can be both decoder and encoder
class Decoder(nn.Module):
    def __init__(self, input_size=1, output_size=1, hidden_size=[16, 16]):
        super().__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.input_size = input_size
        # define the layers
        self.layers = self.create_net()

    def create_net(self):
        modules = []
        modules.append(nn.Linear(self.input_size, self.hidden_size[0]))
        modules.append(nn.ReLU())
        for h in range(len(self.hidden_size) - 1):
            modules.append(nn.Linear(self.hidden_size[h], self.hidden_size[h + 1]))
            modules.append(nn.ReLU())
        # final layer
        modules.append(nn.Linear(self.hidden_size[-1], self.output_size*2))

        return nn.Sequential(*modules)

    def forward(self, x):
        # forward pass
        y = self.layers(x)

        mu, log_sigma = torch.split(y, self.output_size, dim=-1)

        sigma = 0.1 + 0.9 * torch.nn.functional.softplus(log_sigma)
        dist = torch.distributions.Normal(loc=mu, scale=sigma)

        return mu, sigma, dist








class neural_process():
    def __init__(self, input_size=1, output_size=1, hidden_size_encoder=[128]*4,
                 hidden_size_decoder=[128]*4
                 , latent_size=32, hidden_size_normal=16):

        self.deterministic_encoder = DeterministicEncoder(input_size + output_size, latent_size, hidden_size_encoder)
        self.latent_encoder = LatentEncoder(input_size + output_size, latent_size, hidden_size_encoder)
        self.decoder = Decoder(input_size + latent_size * 2, output_size, hidden_size_decoder)
        
        self.optimizer = self.create_optimizer()

    def create_optimizer(self):
        params = list(self.deterministic_encoder.parameters()) + \
                 list(self.latent_encoder.parameters()) + list(self.decoder.parameters())

        return torch.optim.Adam(params, lr=0.0001)

    def train(self, x_context, y_context, x_target, y_target, num_epochs=1):
        # I expect the data to come with the shape [num_batches, data_size, x_size]
        num_batches = x_target.shape[0]
        for epoch in range(num_epochs):
            for batch in range(num_batches):
                self.train_inner(x_context[batch, :, :], y_context[batch, :, :], x_target[batch, :, :],
                                 y_target[batch, :, :])

    def train_inner(self, x_context, y_context, x_target, y_target):
        self.optimizer.zero_grad()  # zero the gradient buffers
        data_size = x_target.shape[0]

        # I compute latent representation
        prior = self.latent_encoder(x_context, y_context)

        # For training, when target_y is available, use targets for latent encoder
        # as Deepmind does
        posterior=self.latent_encoder(x_target, y_target)
        latent_rep = posterior.sample()

        # Deterministic path like latent but without Normal encoding and with attention
        deterministic_r = self.deterministic_encoder(x_context, y_context, x_target)

        latent_r = latent_rep.repeat((data_size, 1))

        input_decoder = torch.cat((x_target, latent_r, deterministic_r), -1)

        mu, _, predicted_distribution = self.decoder(input_decoder)
        print(deterministic_r)

        # Now I compute the loss, the Elbo, the formula on paper
        log_p = predicted_distribution.log_prob(y_target)
        kl = torch.distributions.kl_divergence(prior, posterior)
        kl = kl.repeat(data_size)
        loss = -torch.mean(log_p - kl / data_size)

        loss.backward()
        self.optimizer.step()

    def test(self, x_context, y_context, x_target):
        data_size = x_target.shape[0]

        prior = self.latent_encoder(x_context, y_context)

        latent_r = prior.sample()

        # Deterministic path like latent but without Normal encoding and with attention
        deterministic_r = self.deterministic_encoder(x_context, y_context, x_target)

        latent_r = latent_r.repeat((data_size, 1))

        input_decoder = torch.cat((x_target, latent_r, deterministic_r), -1)

        mu, sigma,_ = self.decoder(input_decoder)

        return mu, sigma

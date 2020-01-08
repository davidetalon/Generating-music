# -*- coding: utf-8 -*-

from torch import nn
import numpy as np
import time
import torch
# from random import randint
import random



class ReplayMemory(object):

    def __init__(self, capacity = 512):
        self.capacity = capacity

        self.memory = []
        self.position = 0

    def push(self, batch):
        split = torch.split(batch, 1, dim=0)
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position:self.position+batch.shape[0]] = split
        self.position = (self.position + batch.shape[0]) % self.capacity

    def sample(self, batch_size):
        sampled = random.sample(self.memory, batch_size)
        concatenated = torch.cat(sampled, dim=0)
        # concatenated = torch.unsqueeze(concatenated, dim=1)
        return concatenated

def weights_init(m):
    """
        Initialize the network: convolutional and batchnorm layers are initialized with 
        values coming from a Normal distribution with mean 0 and variance 0.02. Biases are set to 0.
        Args:
            m: layer to initialize.
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    # elif classname.find('BatchNorm') != -1:
    #     nn.init.normal_(m.weight.data, 1.0, 0.02)
    #     nn.init.constant_(m.bias.data, 0)

class Generative(nn.Module):
    def __init__(self, nz=256 , ng=256, ngf=64, latent_dim=100):

        super(Generative, self).__init__()
        
        self.linear = nn.Linear(latent_dim, 256*ngf)
        self.ngf = ngf

        self.main = nn.Sequential(

            nn.ConvTranspose1d( 16 * ngf, ngf * 8, kernel_size=25, stride=4, padding=11, output_padding =1, bias=True),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.5),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose1d(ngf * 8, ngf * 4, kernel_size=25, stride=4, padding=11, output_padding =1, bias=True),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.5),

            # # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose1d( ngf * 4, ngf * 2, kernel_size=25, stride=4, padding=11, output_padding =1, bias=True),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.5),

            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose1d( ngf * 2, ngf, kernel_size=25, stride=4, padding=11, output_padding =1, bias=True),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.5),

            # state size. (ngf) x 32 x 32
            nn.ConvTranspose1d( ngf, ng, kernel_size=25, stride=4, padding=11, output_padding =1, bias=True),
            nn.Tanh(),

            # nn.Softmax(dim=1)
            # state size. (nc) x 64 x 64
        )

    
    def forward(self, x):
        x = self.linear(x)
        x = nn.ReLU()(x)
        x = x.view(x.shape[0], 16 * self.ngf, 16)
        x = self.main(x)

        return x
        

class Discriminative(nn.Module):
    
    def __init__(self, ng=256, ndf=64):

        super(Discriminative, self).__init__()

        self.ndf = ndf

        self.main = nn.Sequential(

            nn.Conv1d(ng, ndf, 25, 4, 11, bias=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(ndf, ndf * 2, 25, 4, 11, bias=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(ndf * 2, ndf * 4, 25, 4, 11, bias=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(ndf * 4, ndf * 8, 25, 4, 11, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv1d(ndf * 8, ndf*16, 25, 4, 11, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(ndf*256, 1)
        )


    def forward(self, x):
        x = self.main(x)

        return x
        



def train_batch(gen, disc, batch, loss_fn, disc_optimizer, gen_optimizer, latent_dim, device, replay_memory):

    
    # (batch_size, channel, seq_len)

    # batch = torch.transpose(batch, 0, 1)
    # batch = torch.transpose(batch, 1, 2)

    batch_size = batch.shape[0]
    # # print(target_real_data.shape)
    
    # ############################
    # # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
    # ###########################
    disc_optimizer.zero_grad()

    # flipped labels and smoothing
    real = torch.empty((batch_size,1), device=device).uniform_(0, 0.1)
    fake = torch.empty((batch_size,1), device=device).uniform_(0.9, 1.0)

    # noisy labels
    noisy = torch.empty((batch_size,1), device=device).uniform_(0.9, 1.0)
    random = torch.rand(*real.shape, device=device)
    real = torch.where(random <= 0.05, noisy, real)

    noisy = torch.empty((batch_size,1), device=device).uniform_(0, 0.1)
    random = torch.rand(*fake.shape, device=device)
    fake = torch.where(random <= 0.05, noisy, fake)


    # computing the loss
    output = disc(batch)
    real_loss = loss_fn(output, real)
    D_x = output.mean().item()
    start = time.time()
    real_loss.backward()
    end = time.time()

    rnd_assgn = torch.randn((batch_size, 1, latent_dim), device=device)
    fake_batch = gen(rnd_assgn)


    # adding to replay memory
    # replay_memory.push(fake_batch.detach())
    # experience = replay_memory.sample(batch_size)
    experience = fake_batch.detach()


    output = disc(experience)
    fake_loss = loss_fn(output, fake)
    start = time.time()
    fake_loss.backward()
    end = time.time()
    D_G_z1 = output.mean().item()

    disc_top = disc.main[0].weight.grad.norm()
    disc_bottom = disc.linear[-1].weight.grad.norm()

    disc_loss = (real_loss + fake_loss)/2

    disc_optimizer.step()

 
    ############################
    # (2) Update G network: maximize log(D(G(z)))
    ###########################
    gen_optimizer.zero_grad()


    output = disc(fake_batch)
    gen_loss = loss_fn(output, real)


    D_G_z2 = output.mean().item()
    gen_loss.backward()

    gen_top = gen.linear.weight.grad.norm()
    gen_bottom = gen.main[-2].weight.grad.norm()

    gen_optimizer.step()

    return gen_loss.item(), real_loss.item(), fake_loss.item(), disc_loss.item(), D_x, D_G_z1, D_G_z2, disc_top.item(), disc_bottom.item(), gen_top.item(), gen_bottom.item()

def train_disc(gen, disc, batch, lmbda, disc_optimizer, latent_dim, device):


    ############################
    # (2) Update D network
    ###########################

    disc.zero_grad()

    # getting the batch size
    batch_size = batch.shape[0] 

    # vectors of 1's and -1's
    one = torch.tensor(1, dtype=torch.float, device=device, requires_grad=True)
    n_one = -1 * one
    
    # sampling random variables
    epsilon = torch.rand((batch_size, 1, 1), device=device, requires_grad=True)
    epsilon = epsilon.expand(epsilon.size())
    rnd_assgn = torch.empty((batch_size,1, latent_dim), device=device).uniform_(-1, 1)

    # computing the batch
    fake_batch = gen(rnd_assgn)
    
    
    # computing the critic of real samples
    D_real = disc(batch)
    D_real = D_real.mean()
    # D_real.backward(n_one)

    # computing the critic of fake samples
    D_fake = disc(fake_batch)
    D_fake = D_fake.mean()
    # D_fake.backward(one)

    # computing the gp loss
    interpolation = epsilon * batch + (1-epsilon) * fake_batch
    D_interpolation = disc(interpolation)
    gradients = torch.autograd.grad(outputs=D_interpolation, inputs=interpolation,
                              grad_outputs=torch.ones_like(D_interpolation, device=device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    # gradients = torch.randn((64, 1, 16384), device=device, requires_grad=True)
    gradients = gradients.view(gradients.size(0),  -1)
    gradient_norm = gradients.norm(2, dim=1)
    gp = ((gradient_norm - 1)**2)
    D_interpolation = D_interpolation.mean()
    gp = lmbda * gp.mean()

    # gp.backward(one)



    # grad = torch.autograd.grad(midpoint_critic, u, grad_outputs= torch.ones_like(midpoint_critic).to(device))
    # grad_norm = torch.norm(grad[0], 2, dim=1)
    
    # gp = torch.pow(grad_norm - 1, 2)
    # midpoint_critic = midpoint_critic.mean()
    # gp = gp.mean() 
    loss = D_fake - D_real + gp
    loss.backward()

    # gathering the loss and updating
    disc_optimizer.step()
        
    # renaming
    D_x = D_real
    D_G_z1 = D_fake
    D_G_z2 = gp
    
    disc_top = disc.main[0].weight.grad.norm()
    disc_bottom = disc.main[-1].weight.grad.norm()

    return loss, D_x, D_G_z1, D_G_z2, disc_top, disc_bottom

#     # def optimize_disc(true_loss, fake_loss, gp_loss, disc_optimizer):
#     #     true_loss = true_loss.mean()
#     #     fake_loss = fake_loss.mean()
#     #     gp_loss = lamb * gp_loss.mean()

#     #     disc_loss = fake_loss - true_loss + gp_loss
#     #     disc_optimizer.step()

# def train_disc(gen, disc, batch, lmbda, disc_optimizer, latent_dim, device):

#     ############################
#     # (2) Update D network
#     ###########################

#     disc_optimizer.zero_grad()

#     # getting the batch size
#     batch_size = batch.shape[0] 

#     # vectors of 1's and -1's
#     one = torch.tensor(1).float().to(device)
#     n_one = -1 * one
    
#     # sampling random variables
#     epsilon = torch.rand((batch_size, 1, 1), device=device)
#     rnd_assgn = torch.randn((batch_size, 1, latent_dim), device=device)

#     # computing the batch
#     fake_batch = gen(rnd_assgn)
    
#     interpolation = epsilon * batch + (1-epsilon) * fake_batch

#     # computing the critic of real samples
#     D_real = disc(batch)
#     D_real = D_real.mean()
#     D_real.backward(n_one)

#     # computing the critic of fake samples
#     D_fake = disc(fake_batch)
#     D_fake = D_fake.mean()
#     D_fake.backward(one) 

#     # computing the gp loss
#     D_interpolation = disc(interpolation)
#     gradients = torch.autograd.grad(outputs=D_interpolation, inputs=interpolation,
#                               grad_outputs=torch.ones_like(D_interpolation).to(device),
#                               create_graph=True, retain_graph=True, only_inputs=True)[0]
#     gradients = gradients.view(gradients.size(0),  -1)
#     gradient_norm = gradients.norm(2, dim=1)
#     gp = ((gradient_norm - 1)**2)
#     D_interpolation = D_interpolation.mean()
#     gp = lmbda * gp.mean()

#     gp.backward(one)


    

#     # grad = torch.autograd.grad(midpoint_critic, u, grad_outputs= torch.ones_like(midpoint_critic).to(device))
#     # grad_norm = torch.norm(grad[0], 2, dim=1)
    
#     # gp = torch.pow(grad_norm - 1, 2)
#     # midpoint_critic = midpoint_critic.mean()
#     # gp = gp.mean() 
#     loss = D_fake - D_real + gp
#     # loss.backward()
#     # gathering the loss and updating
#     disc_optimizer.step()
        
#     # renaming
#     D_x = D_real
#     D_G_z1 = D_fake
#     D_G_z2 = gp
    
#     disc_top = disc.main[0].weight.grad.norm()
#     disc_bottom = disc.main[-1].weight.grad.norm()

#     return loss, D_x, D_G_z1, D_G_z2, disc_top, disc_bottom


def train_gen(gen, disc, batch, gen_optimizer, latent_dim, device):

    for p in disc.parameters():
        p.requires_grad = False

    # zero the gradient
    gen.zero_grad()

    batch_size = batch.shape[0]

    n_ones = -1 * torch.tensor(1, dtype=torch.float, device=device)
    # sampling a batch of latent variables
    rnd_assgn = torch.empty((batch_size,1, latent_dim), device=device).uniform_(-1, 1)
    fake_batch = gen(rnd_assgn)

    # computing the critic
    G = disc(fake_batch)
    G = G.mean()

    G.backward(n_ones)
    G_loss = -G

    gen_optimizer.step()

    gen_top = gen.linear.weight.grad.norm()
    gen_bottom = gen.main[-2].weight.grad.norm()

    return G_loss, gen_top, gen_bottom
    

if __name__=='__main__':

    from dataset import MusicDataset, ToTensor, collate, ToMulaw, OneHotEncoding
    from torch.utils.data import Dataset, DataLoader
    import torch
    from torchvision import transforms

    # generative model params
    nz = 1
    ngf = 16

    # discriminative model params
    ng = 256
    ndf = 64

    # Check device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
  
    # set up the generator network
    gen = Generative(nz, ng, ngf)
    gen.to(device)
    # set up the discriminative models
    disc = Discriminative(ng, ndf)
    disc.to(device)


    seq_len = 16000 * 5
    normalize = True
    # trans = ToMulaw()

    # subseq_len = 65536
    trans = transforms.Compose([ToMulaw(),
                                OneHotEncoding()
                                ])

    
    # load data
    dataset = MusicDataset("/Users/davidetalon/Desktop/Dev/Generating-music/dataset/maestro_mono",
                                                        seq_len = seq_len,
                                                        normalize = normalize,
                                                        transform=trans)
                        
    dataloader = DataLoader(dataset, batch_size=5, collate_fn=collate(), shuffle=True)

    # test training
    gen_optimizer = torch.optim.Adam(gen.parameters())
    disc_optimizer = torch.optim.Adam(disc.parameters())

    adversarial_loss = torch.nn.BCELoss()

    # Test the network output
    replay_memory = ReplayMemory(capacity=512)

    for i, batch_sample in enumerate(dataloader):

        batch = batch_sample.to(device)
        gen_loss, real_loss, fake_loss, discr_loss, D_x, D_G_z1, D_G_z2, discr_top, discr_bottom, gen_top, gen_bottom = train_batch(gen, disc, batch, adversarial_loss, disc_optimizer, gen_optimizer, device, replay_memory)

        if i == 3:
            break
    

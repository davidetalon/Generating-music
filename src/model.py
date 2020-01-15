#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torch import nn
import numpy as np
import time
import torch
import random

class Transpose1dLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=11, upsample=None, output_padding=1):
        super(Transpose1dLayer, self).__init__()
        self.upsample = upsample

        self.upsample_layer = torch.nn.Upsample(scale_factor=upsample)
        reflection_pad = kernel_size // 2
        self.reflection_pad = nn.ConstantPad1d(reflection_pad, value=0)
        self.conv1d = torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride)
        self.Conv1dTrans = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding, output_padding)

    def forward(self, x):
        if self.upsample:
            return self.conv1d(self.reflection_pad(self.upsample_layer(x)))
        else:
            return self.Conv1dTrans(x)

class AttentionLayer(nn.Module):
    """Attention layer of the SAGAN model.

    Args:
        in_dim (int): number of channels of the input
    """

    def __init__(self, in_dim):

        super(AttentionLayer, self).__init__()

        self.value_conv = nn.Conv1d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv1d(in_channels = in_dim, out_channels = in_dim//8, kernel_size= 1)
        self.query_conv = nn.Conv1d(in_channels = in_dim, out_channels = in_dim//8 , kernel_size= 1)
        self.conv = nn.Conv1d(in_channels = in_dim//8, out_channels = in_dim , kernel_size= 1)
        
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):

        fxi = self.value_conv(x)
        gxj = self.key_conv(x)

        sij = torch.bmm(fxi.permute(0,2,1), gxj)
        betas = self.softmax(sij).permute(0,2,1)

        hidden = self.query_conv(x)

        output = torch.bmm(hidden, betas)
        output = self.conv(output)

        output = self.gamma * output + x

        return output


class ReplayMemory(object):
    """Replay memory for storing latest-generated samples. Samples are stored with a FIFO policy.
    A random sampling is performing when retrieving from it.
    
    Args:
        capacity (int): capacity of the memory. Default: 512.
    """

    def __init__(self, capacity = 512):
        self.capacity = capacity

        self.memory = []
        self.position = 0

    def push(self, batch):
        """Push a batch into the memory

        Args:
            batch (tensor): tensor to push into the memory

        """
        split = torch.split(batch, 1, dim=0)
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position:self.position+batch.shape[0]] = split
        self.position = (self.position + batch.shape[0]) % self.capacity

    def sample(self, batch_size):
        """Randomly draw a batch_size of samples from the memory.

        Args:
            batch_size (int): number of sample to draw.

        Returns:
            Drawn samples.
        """ 
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
    """Generative model which maps from noise in the latent space to samples in the data space.

    Args:
        ng (int): number of channels of the data space (generated samples). Default: 1.
        ngf (int): dimensionality factor of the generator. Default: 64.
        extended_seq (bool): set if extended sequences are required. Default: ``False``.
        latent_dim (int): number of channels of the latent space. Default: 100.
        post_proc (bool): set if the post processing is required. Default: ``True``.
        attention (bool): set if apply attention. Default: ``False``.
    """

    def __init__(self, ng=1, ngf=64, extended_seq=False, latent_dim=100, post_proc=True, attention=False):

        super(Generative, self).__init__()
        self.ngf = ngf
        self.extended_seq = extended_seq
        self.post_proc = post_proc
        self.attention = attention

        if self.extended_seq:
            self.linear = nn.Linear(latent_dim, 256*2*ngf)
        else:
            self.linear = nn.Linear(latent_dim, 256*ngf)

        main = [

            # nn.ConvTranspose1d( 16 * ngf, ngf * 8, kernel_size=25, stride=4, padding=11, output_padding=1, bias=True),
            Transpose1dLayer(16 * ngf, 8 * ngf, 25, 1, upsample=4),
            nn.ReLU(inplace=True),
            
            # state size. (ngf*8) x 4 x 4
            # nn.ConvTranspose1d(ngf * 8, ngf * 4, kernel_size=25, stride=4, padding=11, output_padding=1, bias=True),
            Transpose1dLayer(8 * ngf, 4 * ngf, 25, 1, upsample=4),
            nn.ReLU(inplace=True)
        ]
        
        block2 = [
            # nn.ConvTranspose1d( ngf * 4, ngf * 2, kernel_size=25, stride=4, padding=11, output_padding=1, bias=True),
            Transpose1dLayer(ngf * 4, 2 * ngf, 25, 1, upsample=4),
            nn.ReLU(inplace=True),
        ]

        if attention:
            main +=  [AttentionLayer(ngf * 4)]
            main += block2
            main += [AttentionLayer(ngf * 2)]
        else:
            main += block2

        main += [
            # state size. (ngf*2) x 16 x 16
            # nn.ConvTranspose1d( ngf * 2, ngf, kernel_size=25, stride=4, padding=11, output_padding=1, bias=True),
            Transpose1dLayer(ngf * 2, ngf, 25, 1, upsample=4),
            nn.ReLU(inplace=True),
            
            # state size. (ngf) x 32 x 32
            # nn.ConvTranspose1d( ngf, ng, kernel_size=25, stride=4, padding=11, output_padding=1, bias=True),
            Transpose1dLayer(ngf, ng, 25, 1, upsample=4),
            nn.Tanh(),
        ]

        if extended_seq:
            extra_layer = [
                nn.ConvTranspose1d( 32 * ngf, ngf * 16, kernel_size=25, stride=4, padding=11, output_padding=1, bias=True),
                nn.ReLU(inplace=True),
            ]
            
            main = extra_layer + main

        # instantiate the model
        self.main = nn.Sequential(*main)

        if post_proc:
            self.post_proc_filter_len = 512
            self.post_proc_layer = nn.Conv1d(ng, ng, self.post_proc_filter_len)


    
    def forward(self, x):
        x = self.linear(x)
        x = nn.ReLU()(x)
        if self.extended_seq:
            x = x.view(x.shape[0], 2 * 16 * self.ngf, 16)
        else:
            x = x.view(x.shape[0], 16 * self.ngf, 16)

        x = self.main(x)

        if self.post_proc:

            if (self.post_proc_filter_len % 2) == 0:
                pad_left = self.post_proc_filter_len // 2
                pad_right = pad_left - 1
            else:
                pad_left = (self.post_proc_filter_len - 1) // 2
                pad_right = pad_left

            x = nn.functional.pad(x, (pad_left, pad_right))
            x = self.post_proc_layer(x)

        return x

class PhaseShuffle(nn.Module):
    """Phase Shuffle layer as described by https://arxiv.org/pdf/1802.04208.pdf

    Args:
        shift_factor: absolute value of the maximum shift allowed. Default: 2.
    """

    def __init__(self, shift_factor=2):

        super(PhaseShuffle, self).__init__()
        self.shift_factor = shift_factor

    def forward(self, x):

        seq_len = x.shape[-1]
        random_shift = torch.randint(low = 0, high= 3, size=(x.shape[0],))
  
        abs_shift = torch.abs(random_shift)

        shifted_batch = torch.empty(x.size())
        for idx, sample in enumerate(torch.split(x, 1, dim=0)):

            current_shift = abs_shift[idx]
            # sample = torch.unsqueeze(sample, dim=0)
 
            if (abs_shift[idx] == 0):
                shifted = sample
            elif (abs_shift[idx] > 0):
                # shifted = torch.empty(sample.size(), device=torch.device("cuda"))
                padded = torch.nn.functional.pad(sample, (current_shift, 0), mode='circular')
                shifted = torch.narrow(padded, dim=-1, start=0, length=seq_len)
            else:
                padded = torch.nn.functional.pad(sample, (0, current_shift), mode='circular')
                shifted = torch.narrow(padded, dim=-1, start=x.shape[-1] - seq_len, length=seq_len)
 
            shifted_batch[idx] = shifted

        # # x = torch.cat(shifted_batch, dim=0)
        # x = shifted_batch

        return x

        

class Discriminative(nn.Module):
    """Discriminative model of the gan: could act as critic or catch fake samples depending on the training algorithm.

    Args:
        ng (int): number of channels of the data space (generated samples). Default: 1.
        ndf (ndf): dimensionality factor of the discriminator. Default: 64.
        extended_seq (bool): extended_seq (bool): set if extended sequences are required. Default: ``False``.
        wgan (bool): set if wgan is used as training algorithm. Default: ``False``.
        attention (bool): set if apply attention. Default: ``False``.
    """

    def __init__(self, ng=1, ndf=64, extended_seq=False, wgan=False, attention=False):

        super(Discriminative, self).__init__()

        self.ng = ng
        self.ndf = ndf
        self.extended_seq = extended_seq
        self.wgan = wgan
        self.attention = attention

        main = [

            nn.Conv1d(ng, ndf, 25, 4, 11, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            PhaseShuffle(shift_factor=2),

            nn.Conv1d(ndf, ndf * 2, 25, 4, 11, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            PhaseShuffle(shift_factor=2)
        ]

        block2 = [
            nn.Conv1d(ndf * 2, ndf * 4, 25, 4, 11, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            PhaseShuffle(shift_factor=2),
        ]
        
        if attention:
            main += [AttentionLayer(ndf * 2)]
            main += block2
            main += [AttentionLayer(ndf * 4)]
        else:
            main += block2
        
        main += [
            nn.Conv1d(ndf * 4, ndf * 8, 25, 4, 11, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            PhaseShuffle(shift_factor=2),

            # state size. (ndf*8) x 4 x 4
            nn.Conv1d(ndf * 8, ndf*16, 25, 4, 11, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        if self.extended_seq:
            extra_block = [
            # state size. (ndf*8) x 4 x 4
            PhaseShuffle(shift_factor=2),
            nn.Conv1d(ndf * 16, ndf*32, 25, 4, 11, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            ]

            main += extra_block


        final_block = [
            nn.Flatten(),
            nn.Linear(ndf*(512 if self.extended_seq else 256), 1)
        ]
        main += final_block

        self.main = nn.Sequential(*main)

        self.squashing_layer = nn.Sigmoid()

    
    def forward(self, x):
        x = self.main(x)
        # if self.wgan:
        #     x = self.squashing_layer(x)

        return x
        



def train_batch(gen, disc, batch, loss_fn, disc_optimizer, gen_optimizer, latent_dim, device, replay_memory):
    """Train the models as a vanilla GAN.

    Args:
        gen (Generative): generator of the GAN.
        disc (disc): discriminator of the GAN.
        batch (tensor): batch to use for training.
        loss_fn: loss function used for the optimization process.
        disc_optimizer (Optimizer): optimizer of the discriminator.
        gen_optimizer (Optimizer): optimizer of the generator.
        latent_dim (int): number of channels of the latent space.
        device (torch.device): device where to store tensors.
        replay_memory (ReplayMemory): replay memory used to store new generated samples.

    Returns:
        gen_loss: loss of the generator model.
        real_loss: loss obtained by the discriminator with the data sampled batch.
        fake_loss: loss obtained by the discriminator with the fake generated batch.
        disc_loss: total loss obtained by the discriminator.
        D_x: guess of the discriminator for the data sampled batch.
        D_G_z1: guess of the discriminator for the fake generated batch while updating the discriminator.
        D_G_z2: guess of the discriminator for the fake generated batch while updating the generator.
        disc_top: absolute value of the gradient at the top level of the discriminator.
        disc_bottom: absolute value of the gradient at the bottom level of the discriminator.
        gen_top: absolute value of the gradient at the top level of the generator.
        gen_bottom: absolute value of the gradient at the bottom level of the generator.

    """

    
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


    # adding to replay memory and then sample from it
    replay_memory.push(fake_batch.detach())
    experience = replay_memory.sample(batch_size)
    # experience = fake_batch.detach()


    output = disc(experience)
    fake_loss = loss_fn(output, fake)
    start = time.time()
    fake_loss.backward()
    end = time.time()
    D_G_z1 = output.mean().item()

    disc_top = disc.main[0].weight.grad.norm()
    disc_bottom = disc.main[-1].weight.grad.norm()

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
    """Train the discriminator in the WGAN setting.

    Args:
        gen (Generative): generator of the WGAN.
        disc (disc): critic of the WGAN.
        batch (tensor): batch to use for training.
        lmbda (int): lambda parameter to weight the gradient penaly.
        disc_optimizer (Optimizer): optimizer of the critic.
        latent_dim (int): number of channels of the latent space.
        device (torch.device): device where to store tensors.
    
    Returns:
        loss: loss of the discriminator model
        D_real: critic of the data sampled batch.
        D_fake: critic of the fake generated batch.
        gp: gradient penalty.
        disc_top: absolute value of the gradient at the top level of the discriminator.
        disc_bottom: absolute value of the gradient at the bottom level of the discriminator.
    """


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
    D_real.backward(n_one)

    # computing the critic of fake samples
    D_fake = disc(fake_batch.detach())
    D_fake = D_fake.mean()
    D_fake.backward(one)

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
    gp = lmbda * gp.mean()
    gp.backward(one)

    loss = D_fake - D_real + gp
    # loss.backward()

    # gathering the loss and updating
    disc_optimizer.step()
        
    
    disc_top = disc.main[0].weight.grad.norm()
    disc_bottom = disc.main[-1].weight.grad.norm()

    return loss, D_real, D_fake, gp, disc_top, disc_bottom



def train_gen(gen, disc, batch, gen_optimizer, latent_dim, device):
    """Train the generator in the WGAN setting.

    Args:
        gen (Generative): generator of the WGAN.
        disc (Discriminative): critic of the WGAN.
        batch (tensor): batch to use for training.
        gen_optimizer (Optimizer): optimizer of the generator.
        latent_dim (int): number of channels of the latent space.
        device (torch.device): device where to store tensors.
    
    Returns:
        G_loss: loss of the generator model
        gen_top: absolute value of the gradient at the top level of the generator.
        gen_bottom: absolute value of the gradient at the bottom level of the generator.
    """

    # for p in disc.parameters():
    #     p.requires_grad = False

    # for p in gen.parameters():
    #     p.requires_grad = True

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
    gen_bottom = 0
    # gen_bottom = gen.main[-2].weight.grad.norm()

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
    

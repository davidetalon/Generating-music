#!/usr/bin/env python
# -*- coding: utf-8 -*-

# TODO: check if the testing part is working
# TODO: Phase shuffle tensor creation device
from torch import nn
import numpy as np
import time
import torch
import random


class Transpose1dLayer(nn.Module):
    """Transposed 1D convolution. It applies a fractionally strided convolution.

    Args:
        in_channels (int): number of channels of the input.
        out_channels (int): number of channels produced by the convolution.
        kernel_size (int): size of the convolving kernel.
        stride (int): Stride of the convolution.
        upsample (int): Upsample of the data before convolution. Default: 512.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=4):
        super(Transpose1dLayer, self).__init__()

        self.upsample = upsample
        self.reflection_pad = kernel_size // 2

        self.conv1d = torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride)


    def forward(self, x):

        # apply nearest neighbour upsampling
        x = torch.nn.functional.interpolate(x, scale_factor=self.upsample, mode='nearest')

        # pad to mantain current shape
        x = torch.nn.functional.pad(x, (self.reflection_pad, self.reflection_pad), mode='constant', value=0)

        x = self.conv1d(x)

        return x



class AttentionLayer(nn.Module):
    """Attention layer of the SAGAN model.

    Args:
        in_dim (int): number of channels of the input.
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

        # map on another space
        fxi = self.value_conv(x)
        gxj = self.key_conv(x)

        # compute betas
        sij = torch.bmm(fxi.permute(0,2,1), gxj)
        betas = self.softmax(sij).permute(0,2,1)

        hidden = self.query_conv(x)

        # apply attention
        output = torch.bmm(hidden, betas)
        output = self.conv(output)

        # multiply and sum back
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
        
        # add the batch to the circular buffer and get the pointer
        self.memory[self.position:self.position+batch.shape[0]] = split
        self.position = (self.position + batch.shape[0]) % self.capacity

    def sample(self, batch_size):
        """Randomly draw a batch_size of samples from the memory.

        Args:
            batch_size (int): number of sample to draw.

        Returns:
            Drawn samples.
        """ 

        # draw random samples
        sampled = random.sample(self.memory, batch_size)
        concatenated = torch.cat(sampled, dim=0)

        return concatenated

def weights_init(m):
    """
        Initialize the network: convolutional and linear layers are initialized with 
        values coming from a Kaiming normal distribution.

        Args:
            m: layer to initialize.
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight.data)

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
        self.post_proc_filter_len = 512

        if self.extended_seq:
            self.linear = nn.Linear(latent_dim, 256*2*ngf)
        else:
            self.linear = nn.Linear(latent_dim, 256*ngf)

        self.conv1 = Transpose1dLayer(16 * ngf, 8 * ngf, 25, 1, upsample=4)
        self.conv2 = Transpose1dLayer(8 * ngf, 4 * ngf, 25, 1, upsample=4)
        self.conv3 = Transpose1dLayer(4 * ngf, 2 * ngf, 25, 1, upsample=4)
        self.conv4 = Transpose1dLayer(2 * ngf, ngf, 25, 1, upsample=4)
        self.conv5 = Transpose1dLayer(ngf, ng, 25, 1, upsample=4)

        if self.attention:
            self.att1 = AttentionLayer(ngf * 4)
            self.att2 = AttentionLayer(ngf * 2) 

        if self.extended_seq:
            self.extended = Transpose1dLayer(32 * ngf, ngf * 16, 25, 1, upsample=4)

        if self.post_proc:
            self.post_proc_layer = nn.Conv1d(ng, ng, self.post_proc_filter_len)

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)

    def forward(self, x):
        x = self.linear(x)
        
        if self.extended_seq:
            x = x.view(x.shape[0], 2 * 16 * self.ngf, 16)
        else:
            x = x.view(x.shape[0], 16 * self.ngf, 16)

        x = nn.ReLU(inplace=True)(x)

        if self.extended_seq:
            x = self.extended(x)
            x = nn.ReLU(inplace=True)(x)

        x = self.conv1(x)
        x = nn.ReLU(inplace=True)(x)

        x = self.conv2(x)
        x = nn.ReLU(inplace=True)(x)

        # check for attention
        if self.attention:
            x = self.att1(x)

        x = self.conv3(x)
        x = nn.ReLU(inplace=True)(x)

        # check for attention
        if self.attention:
            x = self.att2(x)

        x = self.conv4(x)
        x = nn.ReLU(inplace=True)(x)

        x = self.conv5(x)
        x = nn.Tanh()(x)

        
        if self.post_proc:

            # compute pad so as to mantain same shape
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

        # get the random shift
        random_shift = torch.randint(low = -self.shift_factor, high= self.shift_factor + 1, size=(x.shape[0],), device=x.device)
        abs_shift = torch.abs(random_shift)
        

        shifted_batch = []
        for idx, sample in enumerate(torch.split(x, 1, dim=0)):

            current_shift = abs_shift[idx]

            if (abs_shift[idx] == 0):
                
                shifted = sample
            elif (abs_shift[idx] > 0):

                # circular shift: pad circularly and get needed part
                padded = torch.nn.functional.pad(sample, (current_shift, 0), mode='circular')
                shifted = torch.narrow(padded, dim=-1, start=0, length=seq_len)
            else:

                # circular shift: pad circularly and get needed part
                padded = torch.nn.functional.pad(sample, (0, current_shift), mode='circular')
                shifted = torch.narrow(padded, dim=-1, start=x.shape[-1] - seq_len, length=seq_len)
 
            
            shifted_batch.append(shifted)

        # gather the batch
        x = torch.cat(shifted_batch, dim=0)

        return x
       

class Discriminative(nn.Module):
    """Discriminative model of the gan: could act as critic or catch fake samples depending on the training algorithm.

    Args:
        ng (int): number of channels of the data space (generated samples). Default: 1.
        ndf (ndf): dimensionality factor of the discriminator. Default: 64.
        extended_seq (bool): extended_seq (bool): set if extended sequences are required. Default: ``False``.
        wgan (bool): set if wgan is used as training algorithm. Default: ``False``.
        attention (bool): set if apply attention. Default: ``False``.
        phase_shift (int): choose the maximum circular shift possible. Default: 2.
    """

    def __init__(self, ng=1, ndf=64, extended_seq=False, wgan=False, attention=False, phase_shift=2):

        super(Discriminative, self).__init__()

        self.ng = ng
        self.ndf = ndf
        self.extended_seq = extended_seq
        self.wgan = wgan
        self.attention = attention
        self.shift_factor = phase_shift

        self.conv1 = nn.Conv1d(ng, ndf, 25, 4, 11, bias=True)
        self.conv2 = nn.Conv1d(ndf, ndf * 2, 25, 4, 11, bias=True)
        self.conv3 = nn.Conv1d(ndf * 2, ndf * 4, 25, 4, 11, bias=True)
        self.conv4 = nn.Conv1d(ndf * 4, ndf * 8, 25, 4, 11, bias=True)
        self.conv5 = nn.Conv1d(ndf * 8, ndf*16, 25, 4, 11, bias=True)

        self.linear = nn.Linear(ndf*(512 if self.extended_seq else 256), 1)

        if self.extended_seq:
            self.extra_layer = nn.Conv1d(ndf * 16, ndf*32, 25, 4, 11, bias=True)

        if self.attention:
            self.att1 = AttentionLayer(ndf * 4) 
            self.att2 = AttentionLayer(ndf * 8)
        
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)

      
    def forward(self, x):

        x = self.conv1(x)
        x = nn.LeakyReLU(0.2, inplace=True)(x)
        x = PhaseShuffle(shift_factor=self.shift_factor)(x)

        x = self.conv2(x)
        x = nn.LeakyReLU(0.2, inplace=True)(x)
        x = PhaseShuffle(shift_factor=self.shift_factor)(x)

        x = self.conv3(x)
        x = nn.LeakyReLU(0.2, inplace=True)(x)
        x = PhaseShuffle(shift_factor=self.shift_factor)(x)

        if self.attention:
            x = self.att1(x)
        
        x = self.conv4(x)
        x = nn.LeakyReLU(0.2, inplace=True)(x)
        x = PhaseShuffle(shift_factor=self.shift_factor)(x)

        if self.attention:
            x = self.att2(x)
        
        x = self.conv5(x)
        x = nn.LeakyReLU(0.2, inplace=True)(x)
        
        if self.extended_seq:
            x = PhaseShuffle(shift_factor=self.shift_factor)(x)
            x = self.extra_layer(x)
        
        x = nn.Flatten()(x)
        x = self.linear(x)
        
        if not self.wgan:
            x = nn.Sigmoid()(x)

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

    batch_size = batch.shape[0]

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


    # computing the loss on the real data
    output = disc(batch)
    real_loss = loss_fn(output, real)
    D_x = output.mean().item()
    # backprop
    real_loss.backward()

    
    # generate a fake batch
    rnd_assgn = torch.randn((batch_size, 1, latent_dim), device=device)
    fake_batch = gen(rnd_assgn)


    # add the fake batch to the replay memory and then sample from it
    replay_memory.push(fake_batch.detach())
    experience = replay_memory.sample(batch_size)
    experience = fake_batch.detach()

    # compute the loss on the fake batch
    output = disc(experience)
    fake_loss = loss_fn(output, fake)
    D_G_z1 = output.mean().item()
    # backprop
    fake_loss.backward()

    
    # get top and bottom layers grads
    disc_top = disc.conv1.weight.grad.norm()
    disc_bottom = disc.linear.weight.norm()

    # get the total loss
    disc_loss = (real_loss + fake_loss)/2

    # update params
    disc_optimizer.step()

 
    ############################
    # (2) Update G network: maximize log(D(G(z)))
    ###########################
    gen_optimizer.zero_grad()


    # compute the generator loss on the fake batch
    output = disc(fake_batch)
    gen_loss = loss_fn(output, real)
    D_G_z2 = output.mean().item()
    # backprop
    gen_loss.backward()

    # get top and bottom layers grads
    gen_top = gen.linear.weight.grad.norm()
    gen_bottom = gen.conv5.conv1d.weight.grad.norm()

    # update params
    gen_optimizer.step()

    return gen_loss.item(), real_loss.item(), fake_loss.item(), disc_loss.item(), D_x, D_G_z1, D_G_z2, disc_top.item(), disc_bottom.item(), gen_top.item(), gen_bottom.item()

def train_disc(gen, disc, batch, lmbda, disc_optimizer, latent_dim, requires_grad, device):
    """Train the discriminator in the WGAN setting.

    Args:
        gen (Generative): generator of the WGAN.
        disc (disc): critic of the WGAN.
        batch (tensor): batch to use for training.
        lmbda (int): lambda parameter to weight the gradient penaly.
        disc_optimizer (Optimizer): optimizer of the critic.
        latent_dim (int): number of channels of the latent space.
        requires_grad (bool): set if backpropagation is needed.
        device (torch.device): device where to store tensors.
    
    Returns:
        loss: loss of the discriminator model
        D_real: critic of the data sampled batch.
        D_fake: critic of the fake generated batch.
        gp: gradient penalty.
        W_loss: Wasserstain loss between data and learnt distribution.
        disc_top: absolute value of the gradient at the top level of the discriminator.
        disc_bottom: absolute value of the gradient at the bottom level of the discriminator.
        ave_grads: average gradients of discriminator layers.
    """

    ############################
    # (2) Update D network
    ###########################

    disc.zero_grad()

    # getting the batch size
    batch_size = batch.shape[0]
    batch.requires_grad_(False)

    # vectors of 1's and -1's
    one = torch.tensor(1, dtype=torch.float, device=device)
    n_one = -1 * one
    
    # sampling random variables
    epsilon = torch.rand((batch_size, 1, 1), device=device)
    epsilon = epsilon.expand(batch.size())
    rnd_assgn = torch.empty((batch_size,1, latent_dim), device=device, requires_grad=False).uniform_(-1, 1)

    # computing the batch
    fake_batch = gen(rnd_assgn)
    
    # computing the critic of real samples
    D_real = disc(batch)
    D_real = D_real.mean()
    if requires_grad:
        D_real.backward(n_one)

    # computing the critic of fake samples
    D_fake = disc(fake_batch.detach())
    D_fake = D_fake.mean()
    if requires_grad:
        D_fake.backward(one)

    # computing the gp penalty
    interpolation = epsilon * batch.detach() + (1-epsilon) * fake_batch.detach()
    interpolation.requires_grad_(True)
    D_interpolation = disc(interpolation)
    gradients = torch.autograd.grad(outputs=D_interpolation, inputs=interpolation,
                              grad_outputs=torch.ones_like(D_interpolation, device=device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    
    gradients = gradients.view(gradients.size(0),  -1)

    gp = lmbda * ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    if requires_grad:
        gp.backward(one)

    # compute the losses
    loss = D_fake - D_real + gp
    W_loss = D_real - D_fake

    # gather average gradients information
    ave_grads = []
    if requires_grad:
        for n, p in disc.named_parameters():
            if(p.requires_grad) and ("bias" not in n):
                ave_grads.append(p.grad.abs().mean().item())

    # update params
    if requires_grad:
        disc_optimizer.step()
        
    # get top and bottom layers grads
    disc_top = disc.conv1.weight.grad.norm()
    disc_bottom = disc.linear.weight.norm()

    return loss.item(), D_real.item(), D_fake.item(), gp.item(), W_loss.item(), disc_top.item(), disc_bottom.item(), ave_grads

    
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
        G_loss: loss of the generator model.
        gen_top: absolute value of the gradient at the top level of the generator.
        gen_bottom: absolute value of the gradient at the bottom level of the generator.
        ave_grads: average gradients of discriminator layers.
    """

    gen.zero_grad()

    batch_size = batch.shape[0]
    batch.requires_grad_(False)

    n_ones = -1 * torch.tensor(1, dtype=torch.float, device=device)
    
    # sampling a batch of latent variables
    rnd_assgn = torch.empty((batch_size,1, latent_dim), device=device).uniform_(-1, 1)
    fake_batch = gen(rnd_assgn)

    # computing the critic on the fake batch
    G = disc(fake_batch)
    G = G.mean()

    G.backward(n_ones)
    G_loss = -G

    # gather average gradients information
    ave_grads = []
    for n, p in gen.named_parameters():
        if(p.requires_grad) and ("bias" not in n):
            ave_grads.append(p.grad.abs().mean().item())

    # update params
    gen_optimizer.step()

    # get top and bottom layers grads
    gen_top = gen.linear.weight.grad.norm()
    gen_bottom = gen.conv5.conv1d.weight.grad.norm()
    

    return G_loss.item(), gen_top.item(), gen_bottom.item(), ave_grads
    
if __name__=='__main__':

    from dataset import MusicDataset
    from torch.utils.data import Dataset, DataLoader
    import torch

    # generative model params
    nz = 1
    ngf = 64

    # discriminative model params
    ng = 1
    ndf = 64

    # Check device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
  
    # set up the generator network
    gen = Generative(ng, ngf, extended_seq=False, latent_dim=100, post_proc=False, attention=False)
    gen.to(device)
    # set up the discriminative model
    disc = Discriminative(ng, ndf, extended_seq=False, wgan=False, attention=False, phase_shift=2)
    disc.to(device)


    seq_len = 16384
    normalize = True
    
    dataset = MusicDataset("dataset/piano_f32le/training", seq_len=seq_len, hop=seq_len/2, normalize=normalize, transform=None)
    dataloader = DataLoader(dataset, batch_size=10)

    # test training
    gen_optimizer = torch.optim.Adam(gen.parameters())
    disc_optimizer = torch.optim.Adam(disc.parameters())

    adversarial_loss = torch.nn.BCELoss()

    # Test the network output
    replay_memory = ReplayMemory(capacity=512)

    for i, batch_sample in enumerate(dataloader):

        batch = batch_sample.to(device)
        gen_loss, D_real, D_fake, disc_loss, D_x, D_G_z1, D_G_z2, disc_top, disc_bottom, gen_top, gen_bottom = train_batch(gen, disc, \
                batch, adversarial_loss, disc_optimizer, gen_optimizer, 100, device, replay_memory)
        disc_loss, D_real, D_fake, gp, W_loss, disc_top, disc_bottom, ave_grads = train_disc(gen, disc, batch, 10, disc_optimizer, 100, True, device)
        gen_loss, gen_top, gen_bottom, ave_grads = train_gen(gen, disc, batch, gen_optimizer, 100, device)

        if i == 3:
            break
    

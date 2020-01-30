#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torch import nn
import numpy as np
import time
import torch
import random

class Transpose1dLayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=4):
        super(Transpose1dLayer, self).__init__()

        self.upsample = upsample

        self.reflection_pad = kernel_size // 2
        # self.reflection_pad = nn.ConstantPad1d(reflection_pad, value=0)

        self.conv1d = torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride)
        # self.Conv1dTrans = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding, output_padding)

    def forward(self, x):

        x = torch.nn.functional.interpolate(x, scale_factor=self.upsample, mode='nearest')

        x = torch.nn.functional.pad(x, (self.reflection_pad, self.reflection_pad), mode='constant', value=0)

        x = self.conv1d(x)

        return x
        # return self.conv1d(self.reflection_pad(self.upsample_layer(x)))


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
        Initialize the network: convolutional and linear layers are initialized with 
        values coming from a Kaiming normal distribution.

        Args:
            m: layer to initialize.
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight.data)
    # elif classname.find('BatchNorm') != -1:
    #     nn.init.normal_(m.weight.data, 1.0, 0.02)
    #     nn.init.constant_(m.bias.data, 0)

# class Generative(nn.Module):
#     """Generative model which maps from noise in the latent space to samples in the data space.

#     Args:
#         ng (int): number of channels of the data space (generated samples). Default: 1.
#         ngf (int): dimensionality factor of the generator. Default: 64.
#         extended_seq (bool): set if extended sequences are required. Default: ``False``.
#         latent_dim (int): number of channels of the latent space. Default: 100.
#         post_proc (bool): set if the post processing is required. Default: ``True``.
#         attention (bool): set if apply attention. Default: ``False``.
#     """


#     def __init__(self, ng=1, ngf=64, extended_seq=False, latent_dim=100, post_proc=True, attention=False):

#         super(Generative, self).__init__()
#         self.ngf = ngf
#         self.extended_seq = extended_seq
#         self.post_proc = post_proc
#         self.attention = attention
#         self.post_proc_filter_len = 512

#         if self.extended_seq:
#             self.linear = nn.Linear(latent_dim, 256*2*ngf)
#         else:
#             self.linear = nn.Linear(latent_dim, 256*ngf)

#         self.conv1 = Transpose1dLayer(16 * ngf, 8 * ngf, 25, 1, upsample=4)
#         self.conv2 = Transpose1dLayer(8 * ngf, 4 * ngf, 25, 1, upsample=4)
#         self.conv3 = Transpose1dLayer(4 * ngf, 2 * ngf, 25, 1, upsample=4)
#         self.conv4 = Transpose1dLayer(2 * ngf, ngf, 25, 1, upsample=4)
#         self.conv5 = Transpose1dLayer(ngf, ng, 25, 1, upsample=4)

#         if self.attention:
#             self.att1 = AttentionLayer(ngf * 4)
#             self.att2 = AttentionLayer(ngf * 2) 

#         if self.extended_seq:
#             self.extended = Transpose1dLayer(32 * ngf, ngf * 16, 25, 1, upsample=4)

#         if self.post_proc:
#             self.post_proc_layer = nn.Conv1d(ng, ng, self.post_proc_filter_len)

#         for m in self.modules():
#             if isinstance(m, nn.ConvTranspose1d) or isinstance(m, nn.Linear):
#                 nn.init.kaiming_normal_(m.weight.data)

#     def forward(self, x):
#         x = self.linear(x)
#         x = nn.ReLU(inplace=True)(x)
#         # x = nn.ReLU()(x)
#         if self.extended_seq:
#             x = x.view(x.shape[0], 2 * 16 * self.ngf, 16)
#         else:
#             x = x.view(x.shape[0], 16 * self.ngf, 16)


#         if self.extended_seq:
#             x = self.extended(x)
#             x = nn.ReLU(inplace=True)(x)

#         x = self.conv1(x)
#         x = nn.ReLU(inplace=True)(x)

#         x = self.conv2(x)
#         x = nn.ReLU(inplace=True)(x)

#         # check for attention
#         if self.attention:
#             x = self.att1(x)

#         x = self.conv3(x)
#         x = nn.ReLU(inplace=True)(x)

#         # check for attention
#         if self.attention:
#             x = self.att2(x)

#         x = self.conv4(x)
#         x = nn.ReLU(inplace=True)(x)

#         x = self.conv5(x)
#         x = nn.Tanh()(x)

        

#         if self.post_proc:

#             if (self.post_proc_filter_len % 2) == 0:
#                 pad_left = self.post_proc_filter_len // 2
#                 pad_right = pad_left - 1
#             else:
#                 pad_left = (self.post_proc_filter_len - 1) // 2
#                 pad_right = pad_left

#             x = nn.functional.pad(x, (pad_left, pad_right))
#             x = self.post_proc_layer(x)

#         return x

    

class Generative(nn.Module):
    def __init__(self, ng=1, ngf=64, extended_seq=False, latent_dim=100, post_proc=True, attention=False): 
        super(Generative, self).__init__()
        

        self.model_size = ngf  # d
        self.num_channels = ng  # c
        self.latent_di = latent_dim
        self.post_proc_filt_len = 512
        # "Dense" is the same meaning as fully connection.
        self.linear = nn.Linear(latent_dim, 256 * self.model_size)

   
        stride = 1
        upsample = 4
        self.conv1 = Transpose1dLayer(16 * self.model_size, 8 * self.model_size, 25, stride, upsample=upsample)
        self.conv2 = Transpose1dLayer(8 * self.model_size, 4 * self.model_size, 25, stride, upsample=upsample)
        self.conv3 = Transpose1dLayer(4 * self.model_size, 2 * self.model_size, 25, stride, upsample=upsample)
        self.conv4 = Transpose1dLayer(2 * self.model_size, self.model_size, 25, stride, upsample=upsample)
        self.conv5 = Transpose1dLayer(self.model_size, ng, 25, stride, upsample=upsample)

        if self.post_proc_filt_len:
            self.ppfilter1 = nn.Conv1d(ng, ng, self.post_proc_filt_len)

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)

    def forward(self, x):
        x = self.linear(x).view(-1, 16 * self.model_size, 16)
        x = torch.relu(x)


        x = torch.relu(self.conv1(x))
      

        x = torch.relu(self.conv2(x))


        x = torch.relu(self.conv3(x))


        x = torch.relu(self.conv4(x))


        output = torch.tanh(self.conv5(x))
        return output

class PhaseShuffle(nn.Module):
    """Phase Shuffle layer as described by https://arxiv.org/pdf/1802.04208.pdf

    Args:
        shift_factor: absolute value of the maximum shift allowed. Default: 2.
    """

    def __init__(self, shift_factor=2):

        super(PhaseShuffle, self).__init__()
        self.shift_factor = shift_factor

    # def forward(self, x):

    #     seq_len = x.shape[-1]
    #     random_shift = torch.randint(low = -self.shift_factor, high= self.shift_factor, size=(x.shape[0],))
  
    #     abs_shift = torch.abs(random_shift)

    #     shifted_batch = torch.empty(x.size())
    #     for idx, sample in enumerate(torch.split(x, 1, dim=0)):

    #         current_shift = abs_shift[idx]
    #         # sample = torch.unsqueeze(sample, dim=0)
 
    #         if (abs_shift[idx] == 0):
    #             shifted = sample
    #         elif (abs_shift[idx] > 0):
    #             # shifted = torch.empty(sample.size(), device=torch.device("cuda"))
    #             padded = torch.nn.functional.pad(sample, (current_shift, 0), mode='circular')
    #             shifted = torch.narrow(padded, dim=-1, start=0, length=seq_len)
    #         else:
    #             padded = torch.nn.functional.pad(sample, (0, current_shift), mode='circular')
    #             shifted = torch.narrow(padded, dim=-1, start=x.shape[-1] - seq_len, length=seq_len)
 
    #         shifted_batch[idx] = shifted

    #     # # x = torch.cat(shifted_batch, dim=0)
    #     # x = shifted_batch

    #     return x

    def forward(self, x):
        if self.shift_factor == 0:
            return x
        # uniform in (L, R)
        k_list = torch.Tensor(x.shape[0]).random_(0, 2 * self.shift_factor + 1) - self.shift_factor
        k_list = k_list.numpy().astype(int)

        # Combine sample indices into lists so that less shuffle operations
        # need to be performed
        k_map = {}
        for idx, k in enumerate(k_list):
            k = int(k)
            if k not in k_map:
                k_map[k] = []
            k_map[k].append(idx)

        # Make a copy of x for our output
        x_shuffle = x.clone()

        # Apply shuffle to each sample
        for k, idxs in k_map.items():
            if k > 0:
                x_shuffle[idxs] = torch.nn.functional.pad(x[idxs][..., :-k], (k, 0), mode='reflect')
            else:
                x_shuffle[idxs] = torch.nn.functional.pad(x[idxs][..., -k:], (0, -k), mode='reflect')

        assert x_shuffle.shape == x.shape, "{}, {}".format(x_shuffle.shape,
                                                       x.shape)
        return x_shuffle

        

# class Discriminative(nn.Module):
#     """Discriminative model of the gan: could act as critic or catch fake samples depending on the training algorithm.

#     Args:
#         ng (int): number of channels of the data space (generated samples). Default: 1.
#         ndf (ndf): dimensionality factor of the discriminator. Default: 64.
#         extended_seq (bool): extended_seq (bool): set if extended sequences are required. Default: ``False``.
#         wgan (bool): set if wgan is used as training algorithm. Default: ``False``.
#         attention (bool): set if apply attention. Default: ``False``.
#     """

#     def __init__(self, ng=1, ndf=64, extended_seq=False, wgan=False, attention=False):

#         super(Discriminative, self).__init__()

#         self.ng = ng
#         self.ndf = ndf
#         self.extended_seq = extended_seq
#         self.wgan = wgan
#         self.attention = attention

#         self.conv1 = nn.Conv1d(ng, ndf, 25, 4, 11, bias=True)
#         self.conv2 = nn.Conv1d(ndf, ndf * 2, 25, 4, 11, bias=True)
#         self.conv3 = nn.Conv1d(ndf * 2, ndf * 4, 25, 4, 11, bias=True)
#         self.conv4 = nn.Conv1d(ndf * 4, ndf * 8, 25, 4, 11, bias=True)
#         self.conv5 = nn.Conv1d(ndf * 8, ndf*16, 25, 4, 11, bias=True)

#         self.linear = nn.Linear(ndf*(512 if self.extended_seq else 256), 1)

#         if self.extended_seq:
#             self.extra_layer = nn.Conv1d(ndf * 16, ndf*32, 25, 4, 11, bias=True)

#         if self.attention:
#             self.att1 = AttentionLayer(ndf * 4) 
#             self.att2 = AttentionLayer(ndf * 8)
        
#         for m in self.modules():
#             if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
#                 nn.init.kaiming_normal_(m.weight.data)

      
#     def forward(self, x):

#         x = self.conv1(x)
#         x = nn.LeakyReLU(0.2, inplace=True)(x)
#         x = PhaseShuffle(shift_factor=2)(x)

#         x = self.conv2(x)
#         x = nn.LeakyReLU(0.2, inplace=True)(x)
#         x = PhaseShuffle(shift_factor=2)(x)

#         x = self.conv3(x)
#         x = nn.LeakyReLU(0.2, inplace=True)(x)
#         x = PhaseShuffle(shift_factor=2)(x)

#         if self.attention:
#             x = self.att1(x)
        
#         x = self.conv4(x)
#         x = nn.LeakyReLU(0.2, inplace=True)(x)
#         x = PhaseShuffle(shift_factor=2)(x)

#         if self.attention:
#             x = self.att2(x)
        
#         x = self.conv5(x)
#         x = nn.LeakyReLU(0.2, inplace=True)(x)
        
#         if self.extended_seq:
#             x = PhaseShuffle(shift_factor=2)(x)
#             x = self.extra_layer(x)
        
#         x = nn.Flatten()(x)
#         x = self.linear(x)
        
#         if not self.wgan:
#             x = nn.Sigmoid()(x)

#         return x

class Discriminative(nn.Module):
    def __init__(self, num_channels=1, model_size=64, extended_seq=False, wgan=False, attention=False):
        super(Discriminative, self).__init__()
        shift_factor = 2
        self.model_size = model_size  # d
        self.num_channels = num_channels  # c
        self.shift_factor = 2  # n
        self.alpha = 0.2

        self.conv1 = nn.Conv1d(num_channels, model_size, 25, stride=4, padding=11)
        self.conv2 = nn.Conv1d(model_size, 2 * model_size, 25, stride=4, padding=11)
        self.conv3 = nn.Conv1d(2 * model_size, 4 * model_size, 25, stride=4, padding=11)
        self.conv4 = nn.Conv1d(4 * model_size, 8 * model_size, 25, stride=4, padding=11)
        self.conv5 = nn.Conv1d(8 * model_size, 16 * model_size, 25, stride=4, padding=11)

        self.ps1 = PhaseShuffle(shift_factor)
        self.ps2 = PhaseShuffle(shift_factor)
        self.ps3 = PhaseShuffle(shift_factor)
        self.ps4 = PhaseShuffle(shift_factor)

        self.linear = nn.Linear(256 * model_size, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)

    def forward(self, x):
        x = nn.functional.leaky_relu(self.conv1(x), negative_slope=self.alpha)

        x = self.ps1(x)

        x = nn.functional.leaky_relu(self.conv2(x), negative_slope=self.alpha)

        x = self.ps2(x)

        x = nn.functional.leaky_relu(self.conv3(x), negative_slope=self.alpha)

        x = self.ps3(x)

        x = nn.functional.leaky_relu(self.conv4(x), negative_slope=self.alpha)

        x = self.ps4(x)

        x = nn.functional.leaky_relu(self.conv5(x), negative_slope=self.alpha)


        x = x.view(-1, 256 * self.model_size)


        return self.linear(x)
        



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
    epsilon = epsilon.expand(batch.size())
    rnd_assgn = torch.empty((batch_size,1, latent_dim), device=device).uniform_(-1, 1)

    # computing the batch
    fake_batch = gen(rnd_assgn)
    # print(fake_batch.requires_grad)
    
    # computing the critic of real samples
    D_real = disc(batch)
    D_real = D_real.mean()
    D_real.backward(n_one)

    # computing the critic of fake samples
    D_fake = disc(fake_batch.detach())
    D_fake = D_fake.mean()
    D_fake.backward(one)

    
    

    # computing the gp loss
    interpolation = epsilon * batch + (1-epsilon) * fake_batch.detach()
    D_interpolation = disc(interpolation)
    gradients = torch.autograd.grad(outputs=D_interpolation, inputs=interpolation,
                              grad_outputs=torch.ones_like(D_interpolation, device=device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    
    gradients = gradients.view(gradients.size(0),  -1)
    gradient_norm = gradients.norm(2, dim=1)
    gp = ((gradient_norm - 1)**2)
    gp = lmbda * gp.mean()
    gp.backward(one)

    # print("disc grad: ", disc.conv1.weight.grad.norm().item())

    loss = D_fake - D_real + gp
    W_loss = D_real - D_fake

    print("D_real:", D_real.item()," D_fake:", D_fake.item(), " D_wass:", W_loss.item())
    # loss.backward()

    # gathering the loss and updating
    disc_optimizer.step()
        
    
    # disc_top = disc.main[0].weight.grad.norm()
    # disc_bottom = disc.main[-1].weight.grad.norm()
    disc_top = disc.conv1.weight.grad.norm()
    disc_bottom = disc.linear.weight.norm()

    return loss.item(), D_real.item(), D_fake.item(), gp.item(), W_loss.item(), disc_top.item(), disc_bottom.item()

    
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

    # print("Gen grad: ", gen.linear.weight.grad.norm().item())

    gen_optimizer.step()

    # gen_top = gen.linear.weight.grad.norm()
    gen_top = gen.linear.weight.grad.norm()
    gen_bottom = gen.conv5.conv1d.weight.grad.norm()
    
    # gen_bottom = gen.main[-2].weight.grad.norm()

    return G_loss.item(), gen_top.item(), gen_bottom.item()
    

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
    

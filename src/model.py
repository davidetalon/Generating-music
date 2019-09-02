# -*- coding: utf-8 -*-

from torch import nn
import numpy as np
import time
import torch

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
    def __init__(self, nz=256 , ng=256, ngf=64):

        super(Generative, self).__init__()

        self.main = nn.Sequential(

            # nn.ConvTranspose1d( nz, ngf * 16, 25, stride=4, padding=6, bias=True),
            # nn.ReLU(True),
    
            nn.ConvTranspose1d( nz, ngf * 8, 25, stride=4, padding=6, bias=True),
            nn.BatchNorm1d(ngf * 8),
            nn.ReLU(True),
            # nn.Dropout(0.5),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose1d(ngf * 8, ngf * 4, 25, 4, 6, bias=True),
            nn.BatchNorm1d(ngf * 4),
            nn.ReLU(True),
            # nn.Dropout(0.5),
            # # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose1d( ngf * 4, ngf * 2, 16, 4, 6, bias=True),
            nn.BatchNorm1d(ngf * 2),
            nn.ReLU(True),
            # nn.Dropout(0.5),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose1d( ngf * 2, ngf, 25, 4, 6, bias=True),
            nn.BatchNorm1d(ngf),
            nn.ReLU(True),
            # nn.Dropout(0.5),

            # state size. (ngf) x 32 x 32
            nn.ConvTranspose1d( ngf, ng, 16, 4, 1, bias=True),
            # nn.Tanh()
            nn.Softmax(dim=1)
            # state size. (nc) x 64 x 64
        )
    
    def forward(self, x):
        x = self.main(x)

        return x
        

class Discriminative(nn.Module):
    
    def __init__(self, ng=256, ndf=64):

        super(Discriminative, self).__init__()

        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv1d(ng, ndf, 25, 4, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv1d(ndf, ndf * 2, 25, 4, 1, bias=True),
            # nn.BatchNorm1d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv1d(ndf * 2, ndf * 4, 25, 4, 1, bias=True),
            # nn.BatchNorm1d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv1d(ndf * 4, ndf * 8, 25, 4, 1, bias=True),
            # nn.BatchNorm1d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv1d(ndf * 8, ndf*16, 25, 4, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(ndf * 16, 1, 25, 4, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.linear = nn.Sequential(
            nn.Linear(14, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.main(x)

        x = x.view(x.shape[0], -1)
        x=self.linear(x)

        return x
        



def train_batch(gen, disc, batch, loss_fn, disc_optimizer, gen_optimizer, device):

    

    batch = torch.transpose(batch, 0, 1)
    batch = torch.transpose(batch, 1, 2)

    batch_size = batch.shape[0]
    # print(target_real_data.shape)

    
    ############################
    # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
    ###########################
    disc_optimizer.zero_grad()

    # flipped labels and smoothing
    real = torch.empty((batch_size,1), device=device).uniform_(0, 0.3)
    fake = torch.empty((batch_size,1), device=device).uniform_(0.7, 1.2)

    # noisy labels
    noisy = torch.empty((batch_size,1), device=device).uniform_(0.7, 1.2)
    random = torch.rand(*real.shape, device=device)
    real = torch.where(random <= 0.05, noisy, real)

    noisy = torch.empty((batch_size,1), device=device).uniform_(0, 0.3)
    random = torch.rand(*fake.shape, device=device)
    fake = torch.where(random <= 0.05, noisy, fake)


    # computing the loss
    output = disc(batch)

    real_loss = loss_fn(output, real)
    D_x = output.mean().item()
    start = time.time()
    real_loss.backward()
    end = time.time()

    rnd_assgn = torch.randn((batch_size, 1, 80), device=device)

    start = time.time()
    fake_batch = gen(rnd_assgn)
    end = time.time()
    
    output = disc(fake_batch.detach())
    fake_loss = loss_fn(output, fake)
    start = time.time()
    fake_loss.backward()
    end = time.time()
    D_G_z1 = output.mean().item()

    disc_top = disc.main[0].weight.grad.norm()
    disc_bottom = disc.linear[-2].weight.grad.norm()

    disc_loss = (real_loss + fake_loss)/2

    start = time.time()
    disc_optimizer.step()
    end = time.time()
 

     ############################
    # (2) Update G network: maximize log(D(G(z)))
    ###########################
    gen_optimizer.zero_grad()


    output = disc(fake_batch)
    gen_loss = loss_fn(output, real)
    
    D_G_z2 = output.mean().item()
    gen_loss.backward()

    gen_top = gen.main[0].weight.grad.norm()
    gen_bottom = gen.main[-2].weight.grad.norm()

    gen_optimizer.step()

    return gen_loss.item(), disc_loss.item(), D_x, D_G_z1, D_G_z2, disc_top.item(), disc_bottom.item(), gen_top.item(), gen_bottom.item()

if __name__=='__main__':

    from dataset import MusicDataset, ToTensor, collate, RandomCrop, OneHotEncoding
    from torch.utils.data import Dataset, DataLoader
    import torch
    from torchvision import transforms

    # generative model params
    nz = 1
    ngf = 16

    # discriminative model params
    ng = 256
    ndf = 16
  
    # set up the generator network
    gen = Generative(nz, ng, ngf)
    # set up the discriminative models
    disc = Discriminative(ng, ndf)


    seq_len = 16000 * 8
    subseq_len = 65536
    trans = transforms.Compose([RandomCrop(seq_len, subseq_len),
                                OneHotEncoding(),
                                ToTensor()
                                ])
    # load data
    dataset = MusicDataset("dataset/FMA/dataset_pcm_8000/Rock", transform=trans)
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate(), shuffle=True)

    # test training
    gen_optimizer = torch.optim.Adam(gen.parameters())
    disc_optimizer = torch.optim.Adam(disc.parameters())

    adversarial_loss = torch.nn.BCELoss()

    # Test the network output
    for i, batch_sample in enumerate(dataloader):

        train_batch(gen, disc, batch_sample, adversarial_loss, disc_optimizer, gen_optimizer)

        if i == 3:
            break
    

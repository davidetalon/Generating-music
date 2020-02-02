#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import time
import datetime
import torch
import torchaudio
import numpy as np
from dataset import MusicDataset, OneHotEncoding, ToTensor, ToMulaw
from model import Generative, Discriminative, train_batch, weights_init, ReplayMemory, train_disc, train_gen
from torch.utils.data import DataLoader, Dataset
# from torchvision import transforms
import json
from pathlib import Path
import scipy.io.wavfile
import soundfile as sf
import librosa


# create the parser
parser = argparse.ArgumentParser(description='Train the GAN for generating Piano Music')

# seed
parser.add_argument('--seed',            type=int, default=30,    help=' Seed for the generation process')

parser.add_argument('--gen_lr',          type=float, default=1e-4,    help=' Generator\'s learning rate')
parser.add_argument('--discr_lr',        type=float, default=1e-4,    help=' Generator\'s learning rate')
parser.add_argument('--wgan',            type=int,   default=0,         help='Choose to train with wgan or vanilla-gan')
parser.add_argument('--disc_updates',    type=int,   default=5,         help='Number of critic updates')
parser.add_argument('--post_proc',       type=int,   default=1,         help='Choose to apply post processing to generated samples')
parser.add_argument('--phase_shift',    type=int,   default=2,         help='Choose the phase shuffle shift param')
parser.add_argument('--attention',       type=int,   default=0,         help='Choose to add attention or not')
parser.add_argument('--extended_seq',    type=int,   default=0,         help='Choose the seq_len 16384/65536')


parser.add_argument('--notes',          type=str, default="Standard model",    help=' Notes on the model')

parser.add_argument('--batch_size',          type=int, default=1,    help='Dimension of the batch')
parser.add_argument('--generated_samples',   type=int, default=8,    help='Number of generated samples for inspection')
parser.add_argument('--latent_dim',          type=int, default=100,    help='Dimension of the latent space')
parser.add_argument('--num_epochs',             type=int, default=5,    help='Number of epochs')

parser.add_argument('--save',             type=bool, default=True,    help='Save the generator and discriminator models')
parser.add_argument('--save_interleaving',type=int, default=100,       help='Number of epochs between backups')

parser.add_argument('--out_dir',          type=str, default='models/',    help='Folder where to save the model')
parser.add_argument('--prod_dir',          type=str, default='produced/',    help='Folder where to save the model')
parser.add_argument('--ckp_dir',          type=str, default='ckps',    help='Folder where to save the model')
parser.add_argument('--metrics_dir',          type=str, default='metrics/',    help='Folder where to save the model')

parser.add_argument('--model_path',          type=str, default='',    help='Path to models to restore')


def save_models(date, prod_dir, gen_file_name, disc_file_name):
    """Save the model parameters

    Args:
        date (String): date for the name of the model to save.
        prod_dir (String): directory path where to save files
        gen_file_name (String): generator model name
        disc_file_name (String): discriminator model name
    """
    print("Backing up the discriminator and generator models")
    torch.save(gen.state_dict(), prod_dir / gen_file_name)
    torch.save(disc.state_dict(), prod_dir / disc_file_name)

def sample_fake(latent, date, epoch, prod_dir):
    """Generate samples from the latent and save them as wav files

    Args:
        latent (Tensor): tensor of the noise to map.
        date (String): date for the name of the samples to save
        epoch (int): epoch of generation
        prod_dir (Path): directory path where to save files
    """

    with torch.no_grad():
        print("Sampling from generator distribution: store samples for inspection.")
        fake = gen(latent).detach().cpu()
        # saving each generated audio independently
        target_folder = prod_dir/ (date + "_" + str(epoch))
        target_folder.mkdir(parents=True, exist_ok=True)
        for idx, sample in enumerate(torch.split(fake, 1, dim=0)):
            sample = torch.squeeze(sample)
            target_path  = target_folder / (str(idx) + ".wav")
            sample = sample.numpy()
            librosa.output.write_wav(target_path, sample, 16000)
            # sf.write(target_path, sample, 16000, format='WAV')

            # torchaudio.save(str(path), fake, 16000)
            # scipy.io.wavfile.write(prod_dir / ("epoch" + str(epoch) + ".wav"), 16000, fake.T )
            # scipy.io.wavfile.write(prod_dir / (date + "epoch" + str(epoch) + ".wav"), 16000, fake.T )

# def calc_gradient_penalty(net_dis, real_data, fake_data, batch_size, lmbda, device):
#     # Compute interpolation factors
#     alpha = torch.rand(batch_size, 1, 1, device=device)
#     alpha = alpha.expand(real_data.size())


#     # Interpolate between real and fake data.
#     interpolates = alpha * real_data + (1 - alpha) * fake_data
#     interpolates.requires_grad=True

#     # Evaluate discriminator
#     disc_interpolates = net_dis(interpolates)

#     # Obtain gradients of the discriminator with respect to the inputs
#     gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
#                               grad_outputs=torch.ones(disc_interpolates.size(), device=device),
#                               create_graph=True, retain_graph=True, only_inputs=True)[0]
#     gradients = gradients.view(gradients.size(0), -1)

#     # Compute MSE between 1.0 and the gradient of the norm penalty to make discriminator
#     # to be a 1-Lipschitz function.
#     gradient_penalty = lmbda * ((gradients.norm(2, dim=1) - 1) ** 2).mean()
#     return gradient_penalty

if __name__ == '__main__':
    
    # Parse input arguments
    args = parser.parse_args()

    # seed
    torch.manual_seed(args.seed)

    #%% Check device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # device = torch.device("cpu")
    # generative model params
    ngf = 64
    # discriminative model params
    ng = 1
    ndf = 64

    extended_seq = True if args.extended_seq >=1 else False
    latent_dim = args.latent_dim
    post_proc = True if args.post_proc >=1 else False
  
    # set up the generator network
    gen = Generative(ng, ngf, extended_seq=extended_seq, latent_dim=args.latent_dim, post_proc=post_proc, attention=args.attention)
    gen.to(device)
    # set up the discriminative models
    disc = Discriminative(ng, ndf, extended_seq=extended_seq, wgan=args.wgan, attention=args.attention, phase_shift=args.phase_shift)
    disc.to(device)

    # gen.apply(weights_init)
    # disc.apply(weights_init)


    # seq_len = 16 * sample_length
    seq_len = 16384
    if extended_seq:
        seq_len = seq_len * 4
    normalize = True
    trans = None

    # test dataloader
    dataset = MusicDataset("dataset/piano/training", seq_len=seq_len, normalize=normalize, transform=trans)
    dataloader = DataLoader(dataset, batch_size=args.batch_size)

    valid_set = MusicDataset("dataset/piano/valid", seq_len=seq_len, normalize=normalize, transform=trans, restart_streams=True)
    valid_dataloader = DataLoader(valid_set, batch_size=args.batch_size//8)

    if args.model_path != '':
        print("Loading the model")
        disc_path = "models/discr_params" + args.model_path + ".pth"
        gen_path = "models/gen_params" + args.model_path + ".pth"
        gen.load_state_dict(torch.load(gen_path, map_location=device))
        disc.load_state_dict(torch.load(disc_path, map_location=device))
   
    # test training
    gen_optimizer = torch.optim.Adam(gen.parameters(), lr=args.gen_lr, betas=(0.5, 0.9))
    disc_optimizer = torch.optim.Adam(disc.parameters(), lr=args.discr_lr, betas=(0.5, 0.9))
    # disc_optimizer = torch.optim.SGD(disc.parameters(), lr=args.discr_lr)

    adversarial_loss = torch.nn.BCELoss()

    # initializing weights
    print("Start training")

    disc_loss_history = []
    gen_loss_history = []

    D_real_history = []
    D_fake_history = []

    disc_top_grad =[]
    disc_bottom_grad=[]
    gen_top_grad = []
    gen_bottom_grad = []
    
    if args.wgan:
        gp_history = []
        W_loss_history = []
        disc_ave_grads_history = []
        gen_ave_grads_history = []

        valid_disc_loss_history = []
        valid_D_real_history = []
        valid_D_fake_history = []
        valid_gp_history = []
        valid_W_loss_history = []
    else:
        D_x_history = []
        D_G_z1_history = []
        D_G_z2_history = []

    
    fixed_noise = torch.empty((args.generated_samples, 1, latent_dim), device=device).uniform_(-1, 1)


    date = datetime.datetime.now()
    date = date.strftime("%d-%m-%Y_%H-%M-%S")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.save:
        prod_dir = Path(args.prod_dir)
        prod_dir.mkdir(parents=True, exist_ok=True)
        prod_dir = prod_dir / str(date)
        prod_dir.mkdir(parents=True, exist_ok=True)
        gen_file_name = 'gen_params.pth'
        disc_file_name = 'discr_params.pth'

    num_batches = (len(dataloader)//args.disc_updates)//args.batch_size

    disc_layers = []
    for n, p in disc.named_parameters():
        if(p.requires_grad) and ("bias" not in n):
            disc_layers.append(n)

    gen_layers = []
    for n, p in gen.named_parameters():
        if(p.requires_grad) and ("bias" not in n):
            gen_layers.append(n)

    # replay_memory = torch.empty((args.batch_size, ng, subseq_len), device=device)
    replay_memory = ReplayMemory(capacity=512)
    for epoch in range(args.num_epochs):

        start_epoch = time.time()

        # Iterate batches
        data_iter = iter(dataloader)
        valid_iter = iter(valid_dataloader)
        # print(len(data_iter))
        # epoch_batches = len(data_iter)//(5 if args.wgan>=1 else 1)
        i = -1

        disc_losses = []
        D_reals =[]
        D_fakes = []
        gps = []
        W_losses = []

        for i in range(num_batches):

            
            start = time.time()

            if(args.wgan >= 1):

                # disc_losses = []
                # D_reals =[]
                # D_fakes = []
                # gps = []
                # W_losses = []
                for p in disc.parameters():
                    p.requires_grad = True

                for t in range(args.disc_updates):
                    batch_sample = data_iter.next()
                    batch = batch_sample.to(device)
                    disc_loss, D_real, D_fake, gp, W_loss, disc_top, disc_bottom, ave_grads = train_disc(gen, disc, batch, 10, disc_optimizer, latent_dim, True, device)
                    disc_losses.append(disc_loss)
                    D_reals.append(D_real)
                    D_fakes.append(D_fake)
                    gps.append(gp)
                    W_losses.append(W_loss)
                    disc_ave_grads_history.extend(ave_grads)

                    # evaluate on the validation
     
                    valid_batch = valid_iter.next()
                    valid_batch = valid_batch.to(device)
                    valid_disc_loss, valid_D_real, valid_D_fake, valid_gp, valid_W_loss, _, _, _ = train_disc(gen, disc, valid_batch, 10, disc_optimizer, latent_dim, False, device)
                    valid_disc_loss_history.append(valid_disc_loss)
                    valid_D_real_history.append(valid_D_real)
                    valid_D_fake_history.append(valid_D_fake)
                    valid_gp_history.append(valid_gp)
                    valid_W_loss_history.append(valid_W_loss)

                # disc_loss = np.mean(np.asarray(disc_losses))
                # D_real = np.mean(np.asarray(D_reals))
                # D_fake = np.mean(np.asarray(D_fakes))
                # gp = np.mean(np.asarray(gps))
                # W_loss = np.mean(np.asarray(W_losses))

                for p in gen.parameters():
                    p.requires_grad = True
                    
                gen_loss, gen_top, gen_bottom, ave_grads = train_gen(gen, disc, batch, gen_optimizer, latent_dim, device)
                gen_ave_grads_history.extend(ave_grads)

                real_loss = fake_loss = 0        
            else:
                batch_sample = data_iter.next()
                batch = batch_sample.to(device)
                gen_loss, D_real, D_fake, disc_loss, D_x, D_G_z1, D_G_z2, disc_top, disc_bottom, gen_top, gen_bottom = train_batch(gen, disc, \
                batch, adversarial_loss, disc_optimizer, gen_optimizer, latent_dim, device, replay_memory)
            
            # train_batch(gen, disc, batch, adversarial_loss, disc_optimizer, gen_optimizer, device, replay_memory)

            disc_loss = np.mean(np.asarray(disc_losses))
            D_real = np.mean(np.asarray(D_reals))
            D_fake = np.mean(np.asarray(D_fakes))
            gp = np.mean(np.asarray(gps))
            W_loss = np.mean(np.asarray(W_losses))

            # saving metrics
            disc_loss_history.append(disc_loss)
            gen_loss_history.append(gen_loss)

            D_real_history.append(D_real)
            D_fake_history.append(D_fake)

            disc_top_grad.append(disc_top)
            disc_bottom_grad.append(disc_bottom)
            gen_top_grad.append(gen_top)
            gen_bottom_grad.append(gen_bottom)

            if args.wgan:
                gp_history.append(gp)
                W_loss_history.append(W_loss)
                end = time.time()
                # print("[Time %d s][Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [W_loss: %f]" % (end-start, epoch + 1, args.num_epochs, i+1, len(dataloader)//5, disc_loss, gen_loss, W_loss))
            else:
                D_x_history.append(D_x)
                D_G_z1_history.append(D_G_z1)
                D_G_z2_history.append(D_G_z2)
                end = time.time()
                print("[Time %d s][Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [D(x): %f, D(G(Z)): %f / %f]" % (end-start, epoch + 1, args.num_epochs, i+1, len(dataloader)//5, disc_loss, gen_loss, D_x, D_G_z1, D_G_z2))
            

            



        end_epoch = time.time()
        # print("\033[92m Epoch %d completed, time %d s \033[0m" %(epoch + 1, end_epoch - start_epoch))
        print("[Time %d s][Epoch %d/%d][D loss: %f] [G loss: %f] [W_loss: %f]" % (end_epoch-start_epoch, epoch + 1, args.num_epochs, disc_loss, gen_loss, W_loss))

        if args.save and ((epoch+1) % args.save_interleaving == 0):
            save_models(date, prod_dir, gen_file_name, disc_file_name)
            sample_fake(fixed_noise, date, epoch, prod_dir)

    if args.save:
        save_models(date, prod_dir, gen_file_name, disc_file_name)
        sample_fake(fixed_noise, date, epoch, prod_dir)

    #Save all needed parameters
    print("Saving parameters")
    # Create output dir


    metrics = {'parameters':vars(args),
                'disc_layers':disc_layers,
                'gen_layers':gen_layers,
                'disc_loss':disc_loss_history, 
                'gen_loss':gen_loss_history, 
                'D_real': D_real_history,
                'D_fake': D_fake_history,
                'gen_top':gen_top_grad, 
                'gen_bottom':gen_bottom_grad,
                'discr_top':disc_top_grad,
                'discr_bottom':disc_bottom_grad,
                'disc_ave_grads':disc_ave_grads_history,
                'gen_ave_grads': gen_ave_grads_history,
                'valid_disc_loss_history': valid_disc_loss_history,
                'valid_D_real_history': valid_D_real_history,
                'valid_D_fake_history': valid_D_fake_history,
                'valid_gp_history': valid_gp_history,
                'valid_W_loss_history': valid_W_loss_history}
    
    if args.wgan:
        wgan_metrics = {'gp':gp_history,
                        'W_loss':W_loss_history}

        metrics = dict(metrics, **wgan_metrics)
    else:
        gan_metrics = {'D_x':D_x_history, 
                     'D_G_z1':D_G_z1_history,
                     'D_G_z2':D_G_z2_history}

        metrics = dict(metrics, **gan_metrics)

  

    # Save metrics
    if args.save:
        metric_file_name = 'metrics.json'
        with open(prod_dir / metric_file_name, 'w') as f:
            json.dump(metrics, f, indent=4)
        print("Completed successfully.")



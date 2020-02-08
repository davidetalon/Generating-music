#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import time
import datetime
import torch
import numpy as np
from dataset import MusicDataset
from model import Generative, Discriminative, train_batch, weights_init, ReplayMemory, train_disc, train_gen
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from utils import save_models, sample_fake, collect_and_save


# create the parser
parser = argparse.ArgumentParser(description='Train the GAN for generating Piano Music')


parser.add_argument('--seed',               type=int,   default=30,         help=' Seed for the generation process')
parser.add_argument('--gen_lr',             type=float, default=1e-4,       help=' Generator\'s learning rate')
parser.add_argument('--discr_lr',           type=float, default=1e-4,       help=' Generator\'s learning rate')
parser.add_argument('--wgan',               type=int,   default=0,          help='Choose to train with wgan or vanilla-gan')
parser.add_argument('--disc_updates',       type=int,   default=5,          help='Number of critic updates')
parser.add_argument('--post_proc',          type=int,   default=1,          help='Choose to apply post processing to generated samples')
parser.add_argument('--phase_shift',        type=int,   default=2,          help='Choose the phase shuffle shift param')
parser.add_argument('--attention',          type=int,   default=0,          help='Choose to add attention or not')
parser.add_argument('--extended_seq',       type=int,   default=0,          help='Choose the seq_len 16384/65536')
parser.add_argument('--notes',              type=str,   default="Std",      help=' Notes on the model')
parser.add_argument('--batch_size',         type=int,   default=1,          help='Dimension of the batch')
parser.add_argument('--generated_samples',  type=int,   default=8,          help='Number of generated samples for inspection')
parser.add_argument('--latent_dim',         type=int,   default=100,        help='Dimension of the latent space')
parser.add_argument('--ngf',                type=int,   default=64,         help='Generator dimensionality factor')
parser.add_argument('--ng',                 type=int,   default=1,          help='Number of channels of produced samples')
parser.add_argument('--ndf',                type=int,   default=64,         help='Discriminator dimensionality factor')
parser.add_argument('--num_epochs',         type=int,   default=5,          help='Number of epochs')
parser.add_argument('--save',               type=bool,  default=True,       help='Save the generator and discriminator models')
parser.add_argument('--save_interleaving',  type=int,   default=100,        help='Number of epochs between backups')
parser.add_argument('--data_dir',           type=str,   default='dataset/', help='Data folder')
parser.add_argument('--prod_dir',           type=str,   default='produced/',help='Folder where to save the model')
parser.add_argument('--model_folder',       type=str,   default='',         help='Path to models to restore')





if __name__ == '__main__':
    
    # Parse input arguments
    args = parser.parse_args()

    # seed
    torch.manual_seed(args.seed)

    #%% Check device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # generative model params
    ngf = args.ngf
    ng = args.ng

    # discriminative model params
    ndf = args.ndf

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
    data = Path(args.data_dir)
    training_data = data / "training"
    dataset = MusicDataset(training_data, seq_len=seq_len, hop=int(seq_len//2), normalize=normalize, transform=trans)
    dataloader = DataLoader(dataset, batch_size=args.batch_size)

    valid_batch_size = 1 if args.batch_size//8==0 else args.batch_size//8
    validation_data = data / "validation"
    valid_set = MusicDataset(validation_data, seq_len=seq_len, normalize=normalize, transform=trans, restart_streams=True)
    valid_dataloader = DataLoader(valid_set, batch_size=1)

    # let's use already stored model
    if args.model_folder != '':
        print("Loading the model")
        model_dir = Path(args.model_folder)
        disc_path = model_dir / "discr_params.pth"
        gen_path = model_dir / "gen_params.pth"
        gen.load_state_dict(torch.load(gen_path, map_location=device))
        disc.load_state_dict(torch.load(disc_path, map_location=device))
   
    # test training
    gen_optimizer = torch.optim.Adam(gen.parameters(), lr=args.gen_lr, betas=(0.5, 0.9))
    disc_optimizer = torch.optim.Adam(disc.parameters(), lr=args.discr_lr, betas=(0.5, 0.9))
    

    adversarial_loss = torch.nn.BCELoss()

    
    print("Start training")

    disc_loss_history = []
    gen_loss_history = []

    D_real_history = []
    D_fake_history = []

    disc_top_grad_history =[]
    disc_bottom_grad_history=[]
    gen_top_grad_history = []
    gen_bottom_grad_history = []
    
    gp_history = []
    W_loss_history = []
    disc_ave_grads_history = []
    gen_ave_grads_history = []

    valid_disc_loss_history = []
    valid_D_real_history = []
    valid_D_fake_history = []
    valid_gp_history = []
    valid_W_loss_history = []

    D_x_history = []
    D_G_z1_history = []
    D_G_z2_history = []

    # random noise input of the generator for inspection
    fixed_noise = torch.empty((args.generated_samples, 1, latent_dim), device=device).uniform_(-1, 1)


    date = datetime.datetime.now()
    date = date.strftime("%d-%m-%Y_%H-%M-%S")

    if args.save:
        prod_dir = Path(args.prod_dir)
        prod_dir.mkdir(parents=True, exist_ok=True)
        prod_dir = prod_dir / str(date)
        prod_dir.mkdir(parents=True, exist_ok=True)
        gen_file_name = 'gen_params.pth'
        disc_file_name = 'discr_params.pth'

    # get the number of batches
    num_batches = (len(dataloader)//args.disc_updates)//args.batch_size
    if num_batches == 0:
        raise ValueError("Limited dataset, a discriminator update cycle is not possbile(5xbatch_size samples needed). Try reducing the batch size")

    # collect discriminator layers
    disc_layers = []
    for n, p in disc.named_parameters():
        if(p.requires_grad) and ("bias" not in n):
            disc_layers.append(n)

    # collect generator layers
    gen_layers = []
    for n, p in gen.named_parameters():
        if(p.requires_grad) and ("bias" not in n):
            gen_layers.append(n)

    
    replay_memory = ReplayMemory(capacity=512)
    for epoch in range(args.num_epochs):

        start_epoch = time.time()

        # Iterate batches
        data_iter = iter(dataloader)
        valid_iter = iter(valid_dataloader)

        i = -1

        disc_losses = []
        D_reals =[]
        D_fakes = []
        gps = []
        W_losses = []

        valid_disc_losses = []
        valid_D_reals = []
        valid_D_fakes = []
        valid_gps = []
        valid_W_losses = []

        for i in range(num_batches):

            start = time.time()

            if(args.wgan >= 1):
                
                # do not require gradient for the generator
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
                    valid_disc_losses.append(valid_disc_loss)
                    valid_D_reals.append(valid_D_real)
                    valid_D_fakes.append(valid_D_fake)
                    valid_gps.append(valid_gp)
                    valid_W_losses.append(valid_W_loss)

                # reactivate gradient
                for p in gen.parameters():
                    p.requires_grad = True
                    
                gen_loss, gen_top, gen_bottom, ave_grads = train_gen(gen, disc, batch, gen_optimizer, latent_dim, device)
                gen_ave_grads_history.extend(ave_grads)
   
            else:
                # vanilla gan optimization procedure
                batch_sample = data_iter.next()
                batch = batch_sample.to(device)
                gen_loss, D_real, D_fake, disc_loss, D_x, D_G_z1, D_G_z2, disc_top, disc_bottom, gen_top, gen_bottom = train_batch(gen, disc, \
                batch, adversarial_loss, disc_optimizer, gen_optimizer, latent_dim, device, replay_memory)
                disc_losses.append(D_real + D_fake)
            

            # saving discriminator metrics
            disc_loss_history.append(np.mean(np.asarray(disc_losses)))
            disc_top_grad_history.append(np.mean(np.asarray(disc_top)))
            disc_bottom_grad_history.append(np.mean(np.asarray(disc_bottom)))
            
        
            # saving generator metrics
            gen_loss_history.append(gen_loss)
            gen_top_grad_history.append(gen_top)
            gen_bottom_grad_history.append(gen_bottom)

            if args.wgan:

                # specific wgan metrics
                D_real_history.append(np.mean(np.asarray(D_reals)))
                D_fake_history.append(np.mean(np.asarray(D_fakes)))
                gp_history.append(np.mean(np.asarray(gps)))
                W_loss_history.append(np.mean(np.asarray(W_losses)))

                # saving validation metrics
                valid_disc_loss_history.append(np.mean(np.asarray(valid_disc_losses)))
                valid_D_real_history.append(np.mean(np.asarray(valid_D_reals)))
                valid_D_fake_history.append(np.mean(np.asarray(valid_D_fakes)))
                valid_gp_history.append(np.mean(np.asarray(valid_gps)))
                valid_W_loss_history.append(np.mean(np.asarray(valid_W_losses)))
                end = time.time()
                
            else:
                D_x_history.append(D_x)
                D_G_z1_history.append(D_G_z1)
                D_G_z2_history.append(D_G_z2)
                end = time.time()
                print("[Time %d s][Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [D(x): %f, D(G(Z)): %f / %f]" % (end-start, epoch + 1, args.num_epochs, i+1, num_batches, disc_loss, gen_loss, D_x, D_G_z1, D_G_z2))
            
        end_epoch = time.time()

        if args.wgan:
            print("[Time %d s][Epoch %d/%d][D loss: %f] [G loss: %f] [W_loss: %f]" % (end_epoch-start_epoch, epoch + 1, args.num_epochs, disc_loss, gen_loss, W_loss))

        # store current metrics, models and samples
        if args.save and ((epoch+1) % args.save_interleaving == 0):
            save_models(gen, disc, date, prod_dir, gen_file_name, disc_file_name)
            sample_fake(gen, fixed_noise, date, epoch, prod_dir)
            collect_and_save(prod_dir,
                            parameters=vars(args),
                            disc_layers=disc_layers,
                            gen_layers=gen_layers,
                            disc_loss=disc_loss_history, 
                            gen_loss=gen_loss_history, 
                            D_real=D_real_history,
                            D_fake=D_fake_history,
                            gen_top=gen_top_grad_history, 
                            gen_bottom=gen_bottom_grad_history,
                            discr_top=disc_top_grad_history,
                            discr_bottom=disc_bottom_grad_history,
                            disc_ave_grads=disc_ave_grads_history,
                            gen_ave_grads=gen_ave_grads_history,
                            valid_disc_loss_history=valid_disc_loss_history,
                            valid_D_real_history=valid_D_real_history,
                            valid_D_fake_history=valid_D_fake_history,
                            valid_gp_history=valid_gp_history,
                            valid_W_loss_history=valid_W_loss_history,
                            gp=gp_history,
                            W_loss=W_loss_history,
                            D_x=D_x_history, 
                            D_G_z1=D_G_z1_history,
                            D_G_z2=D_G_z2_history
                            )

    # concluded training, store everything
    if args.save:
        save_models(gen, disc, date, prod_dir, gen_file_name, disc_file_name)
        sample_fake(gen, fixed_noise, date, epoch, prod_dir)
        collect_and_save(prod_dir,
                        parameters=vars(args),
                        disc_layers=disc_layers,
                        gen_layers=gen_layers,
                        disc_loss=disc_loss_history, 
                        gen_loss=gen_loss_history, 
                        D_real=D_real_history,
                        D_fake=D_fake_history,
                        gen_top=gen_top_grad_history, 
                        gen_bottom=gen_bottom_grad_history,
                        discr_top=disc_top_grad_history,
                        discr_bottom=disc_bottom_grad_history,
                        disc_ave_grads=disc_ave_grads_history,
                        gen_ave_grads=gen_ave_grads_history,
                        valid_disc_loss_history=valid_disc_loss_history,
                        valid_D_real_history=valid_D_real_history,
                        valid_D_fake_history=valid_D_fake_history,
                        valid_gp_history=valid_gp_history,
                        valid_W_loss_history=valid_W_loss_history,
                        gp=gp_history,
                        W_loss=W_loss_history,
                        D_x=D_x_history, 
                        D_G_z1=D_G_z1_history,
                        D_G_z2=D_G_z2_history
                        )

    print("Completed successfully.")



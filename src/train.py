import argparse
import time
import datetime
import torch
import torchaudio
import numpy as np
from dataset import MusicDataset, collate, OneHotEncoding, ToTensor, ToMulaw
from model import Generative, Discriminative, train_batch, weights_init, ReplayMemory, train_disc, train_gen
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import json
from pathlib import Path
import scipy.io.wavfile


# create the parser
parser = argparse.ArgumentParser(description='Train the CSP GAN')

# seed
parser.add_argument('--seed',            type=int, default=30,    help=' Seed for the generation process')
parser.add_argument('--gen_lr',          type=float, default=0.0002,    help=' Generator\'s learning rate')
parser.add_argument('--discr_lr',        type=float, default=0.0001,    help=' Generator\'s learning rate')
parser.add_argument('--wgan',            type=int,   default=0,         help='Choose to train with wgan or vanilla-gan')
parser.add_argument('--notes',          type=str, default="Standard model",    help=' Notes on the model')

parser.add_argument('--batch_size',          type=int, default=5,    help='Dimension of the batch')
parser.add_argument('--latent_dim',          type=int, default=90,    help='Dimension of the latent space')
parser.add_argument('--num_epochs',             type=int, default=5,    help='Number of epochs')

parser.add_argument('--save',             type=bool, default=True,    help='Save the generator and discriminator models')
parser.add_argument('--out_dir',          type=str, default='models/',    help='Folder where to save the model')
parser.add_argument('--prod_dir',          type=str, default='produced/',    help='Folder where to save the model')
parser.add_argument('--ckp_dir',          type=str, default='ckps',    help='Folder where to save the model')
parser.add_argument('--metrics_dir',          type=str, default='metrics/',    help='Folder where to save the model')

parser.add_argument('--model_path',          type=str, default='',    help='Path to models to restore')


if __name__ == '__main__':
    
    # Parse input arguments
    args = parser.parse_args()

    # seed
    torch.manual_seed(args.seed)

    #%% Check device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # device = torch.device("cpu")
    # generative model params
    nz = 1
    ngf = 64
    # discriminative model params
    ng = 1
    ndf = 64

    latent_dim = args.latent_dim
  
    # set up the generator network
    gen = Generative(nz, ng, ngf, latent_dim)
    gen.to(device)
    # set up the discriminative models
    disc = Discriminative(ng, ndf)
    disc.to(device)

    gen.apply(weights_init)
    disc.apply(weights_init)

    # since sampling rate is 16 KHz we want sample_length milliseconds audio files 
    sample_length = 5000
    # seq_len = 16 * sample_length
    seq_len = 16384

    normalize = True
    trans = None

    # test dataloader
    dataset = MusicDataset("dataset/maestro_mono",
                                        seq_len = seq_len,
                                        normalize = normalize,
                                        transform=trans)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    if args.model_path != '':
        print("Loading the model")
        disc_path = "models/discr_params" + args.model_path + ".pth"
        gen_path = "models/gen_params" + args.model_path + ".pth"
        gen.load_state_dict(torch.load(gen_path, map_location=device))
        disc.load_state_dict(torch.load(disc_path, map_location=device))
   
    # test training
    gen_optimizer = torch.optim.Adam(gen.parameters(), lr=args.gen_lr, betas=(0.5, 0.9))
    disc_optimizer = torch.optim.Adam(gen.parameters(), lr=args.gen_lr, betas=(0.5, 0.9))
    # disc_optimizer = torch.optim.SGD(disc.parameters(), lr=args.discr_lr)

    adversarial_loss = torch.nn.BCELoss()

    # initializing weights
    print("Start training")
    gen_loss_history = []
    real_loss_history = []
    fake_loss_history = []
    discr_loss_history = []
    D_x_history = []
    D_G_z1_history = []
    D_G_z2_history = []

    discr_top_grad =[]
    discr_bottom_grad=[]
    gen_top_grad = []
    gen_bottom_grad = []


    fixed_noise = torch.randn((1, 1, 100), device=device)


    date = datetime.datetime.now()
    date = date.strftime("%d-%m-%Y,%H-%M-%S")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.save:
        ckp_dir = Path(args.ckp_dir)
        ckp_dir.mkdir(parents=True, exist_ok=True)
        prod_dir = Path(args.prod_dir)
        prod_dir.mkdir(parents=True, exist_ok=True)

    # replay_memory = torch.empty((args.batch_size, ng, subseq_len), device=device)
    replay_memory = ReplayMemory(capacity=512)
    for epoch in range(args.num_epochs):

        start_epoch = time.time()

        # Iterate batches
        data_iter = iter(dataloader)
        i = -1
        while i < len(dataloader) and i < 26280:

            

        # for i, batch_sample in enumerate(dataloader):
           
            # batch_sample = data_iter.next()
            # i += 1
            # batch = batch_sample.to(device)

            # Update network
            start = time.time()

            if(args.wgan >= 1):
                for t in range(5):
                    batch_sample = data_iter.next()
                    i += 1
                    batch = batch_sample.to(device)
                    disc_loss, D_x, D_G_z1, D_G_z2, discr_top, discr_bottom = train_disc(gen, disc, batch, 10, disc_optimizer, latent_dim, device)
                    
                gen_loss, gen_top, gen_bottom = train_gen(gen, disc, batch, gen_optimizer, latent_dim, device)

                real_loss = fake_loss = 0        
            else:
                gen_loss, real_loss, fake_loss, discr_loss, D_x, D_G_z1, D_G_z2, discr_top, discr_bottom, gen_top, gen_bottom = train_batch(gen, disc, \
                batch, adversarial_loss, disc_optimizer, gen_optimizer, latent_dim, device, replay_memory)
            
            # train_batch(gen, disc, batch, adversarial_loss, disc_optimizer, gen_optimizer, device, replay_memory)

            # saving metrics
            gen_loss_history.append(gen_loss)
            real_loss_history.append(real_loss)
            fake_loss_history.append(fake_loss)
            discr_loss_history.append(disc_loss)
            D_x_history.append(D_x)
            D_G_z1_history.append(D_G_z1)
            D_G_z2_history.append(D_G_z2)
            discr_top_grad.append(discr_top)
            discr_bottom_grad.append(discr_bottom)
            gen_top_grad.append(gen_top)
            gen_bottom_grad.append(gen_bottom)

            end = time.time()
            print("[Time %d s][Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [D(x): %f, D(G(Z)): %f / %f]" % (end-start, epoch + 1, args.num_epochs, i+1, len(dataloader), disc_loss, gen_loss, D_x, D_G_z1, D_G_z2))

            if args.save and ((i+1) % 5000 == 0) :
            
                gen_file_name = 'gen_params'+date+'.pth'
                discr_file_name = 'discr_params'+date+'.pth'
                torch.save(gen.state_dict(), ckp_dir / gen_file_name)
                torch.save(disc.state_dict(), ckp_dir / discr_file_name)
                with torch.no_grad():
                    fake = gen(fixed_noise).detach().cpu()
                    fake = torch.squeeze(fake, dim = 0)
                    path  = prod_dir / (date + "epoch" + str(epoch) + ".wav")
                    torchaudio.save(str(path), fake, 16000)
                    # torchaudio.save(prod_dir / (date + "epoch" + str(epoch) + ".wav"), prova, 16000)
                    # scipy.io.wavfile.write(prod_dir / ("epoch" + str(epoch) + ".wav"), 16000, fake.T )
                    # scipy.io.wavfile.write(prod_dir / (date + "epoch" + str(i) + ".wav"), 16000, fake.T )


        end_epoch = time.time()
        print("\033[92m Epoch %d completed, time %d s \033[0m" %(epoch, end_epoch - start_epoch))
        if args.save :
            
            gen_file_name = 'gen_params'+date+'.pth'
            discr_file_name = 'discr_params'+date+'.pth'
            torch.save(gen.state_dict(), ckp_dir / gen_file_name)
            torch.save(disc.state_dict(), ckp_dir / discr_file_name)
            with torch.no_grad():
                fake = gen(fixed_noise).detach().cpu()
                fake = torch.squeeze(fake, dim = 0)
                path  = prod_dir / (date + "epoch" + str(epoch) + ".wav")
                torchaudio.save(str(path), fake, 16000)
                # scipy.io.wavfile.write(prod_dir / ("epoch" + str(epoch) + ".wav"), 16000, fake.T )
                # scipy.io.wavfile.write(prod_dir / (date + "epoch" + str(epoch) + ".wav"), 16000, fake.T )

    #Save all needed parameters
    print("Saving parameters")
    # Create output dir

    # Save network parameters
    # date = datetime.datetime.now()
    # date = date.strftime("%d-%m-%Y,%H-%M-%S")
    if args.save:
        gen_file_name = 'gen_params'+date+'.pth'
        discr_file_name = 'discr_params'+date+'.pth'
        torch.save(gen.state_dict(), out_dir / gen_file_name)
        torch.save(disc.state_dict(), out_dir / discr_file_name)


    # Save training parameters
    # params_file_name = 'training_args'+date+'.json'
    # with open(out_dir / params_file_name, 'w') as f:
    #     json.dump(vars(args), f, indent=4)

    metrics = {'parameters':vars(args),
            'gen_loss':gen_loss_history, 
            'real_loss': real_loss_history,
            'fake_loss':fake_loss_history,
            'discr_loss':discr_loss_history, 
            'D_x':D_x_history, 
            'D_G_z1':D_G_z1_history,
            'D_G_z2':D_G_z2_history, 
            'gen_top':gen_top_grad, 
            'gen_bottom':gen_bottom_grad,
            'discr_top':discr_top_grad,
            'discr_bottom':discr_bottom_grad}

    # Save metrics
    metrics_dir = Path(args.metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metric_file_name = 'metrics'+ date +'.json'
    with open(metrics_dir / metric_file_name, 'w') as f:
        json.dump(metrics, f, indent=4)
    print("Completed successfully.")
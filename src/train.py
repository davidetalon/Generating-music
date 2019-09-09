import argparse
import time
import datetime
import torch
import numpy as np
from dataset import MusicDataset, collate, RandomCrop, OneHotEncoding, ToTensor, Crop_and_pad
from model import Generative, Discriminative, train_batch, weights_init, ReplayMemory
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
parser.add_argument('--notes',          type=str, default="Standard model",    help=' Notes on the model')

parser.add_argument('--batch_size',          type=int, default=16,    help='Dimension of the batch')
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

    # generative model params
    nz = 1
    ngf = 64
    # discriminative model params
    ng = 1
    ndf = 64
  
    # set up the generator network
    gen = Generative(nz, ng, ngf)
    gen.to(device)
    # set up the discriminative models
    disc = Discriminative(ng, ndf)
    disc.to(device)

    gen.apply(weights_init)
    disc.apply(weights_init)

    seq_len = 16000 * 8
    subseq_len = 16384
    trans = transforms.Compose([Crop_and_pad(subseq_len),
                                # OneHotEncoding(),
                                ToTensor()
                                ])
    # load data
    dataset = MusicDataset("dataset/words_f32le", transform=trans)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate(), shuffle=True)

    if args.model_path != '':
        print("Loading the model")
        disc_path = "models/discr_params" + args.model_path + ".pth"
        gen_path = "models/gen_params" + args.model_path + ".pth"
        gen.load_state_dict(torch.load(gen_path, map_location=device))
        disc.load_state_dict(torch.load(disc_path, map_location=device))
   
    # test training
    gen_optimizer = torch.optim.RMSprop(gen.parameters(), lr=5e-5)
    disc_optimizer = torch.optim.RMSprop(disc.parameters(), lr=5e-5)
    # gen_optimizer = torch.optim.Adam(gen.parameters(), lr=args.gen_lr, betas=(0.5, 0.999))
    # disc_optimizer = torch.optim.Adam(gen.parameters(), lr=args.gen_lr, betas=(0.5, 0.999))
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

        # Iterate batches
        # for i, batch_sample in enumerate(dataloader):

        start = time.time()
        data_iter = iter(dataloader)
        i=0
        while i < len(dataloader):
            # moving to device
            # batch = batch_sample.to(device)

            for p in disc.parameters(): # reset requires_grad
                p.requires_grad = True

            j = 0
            while j < 5 and i < len(dataloader):
                j += 1

                for p in disc.parameters():
                    p.data.clamp_(-0.01, 0.01)

                batch_sample = data_iter.next()
                i += 1

                batch = batch_sample.to(device)


                # (batch_size, seq_len)
                batch = torch.transpose(batch, 0, 1)
                # (batch_size, channels, seq_len)
                batch = torch.unsqueeze(batch, dim=1)
                # batch = torch.transpose(batch, 1, 2)

                batch_size = batch.shape[0]
                # print(target_real_data.shape)
                
                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                disc_optimizer.zero_grad()

                # flipped labels and smoothing
                real = torch.ones((batch_size,1), device=device)
                fake = -torch.ones((batch_size,1), device=device)

                # computing the loss
                output = disc(batch)
                D_real = output
                D_real.backward(real)

                D_x = output.mean().item()
  
                # generating the fake_batch
                rnd_assgn = torch.randn((batch_size, 1, 100), device=device)
                fake_batch = gen(rnd_assgn)

                # adding to replay memory
                replay_memory.push(fake_batch.detach())
                experience = replay_memory.sample(batch_size)


                output = disc(experience)
                D_fake = output
                D_fake.backward(fake)
                D_G_z1 = output.mean().item()

                disc_top = disc.main[0].weight.grad.norm()
                disc_bottom = disc.linear[-1].weight.grad.norm()

                # disc_loss = (real_loss + fake_loss)/
                real_loss = torch.mean(D_real)
                fake_loss = torch.mean(D_fake)
                discr_loss = -(real_loss - fake_loss)

                disc_optimizer.step()

            discr_top = disc.main[0].weight.grad.norm()
            discr_bottom = disc.linear[-1].weight.grad.norm()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            # for p in disc.parameters():
            #     p.requires_grad = False # to avoid computation
            gen_optimizer.zero_grad()


            output = disc(fake_batch)
            # gen_loss = loss_fn(output, real)
            gen_loss = -torch.mean(output)


            D_G_z2 = output.mean().item()
            gen_loss.backward(real)

            gen_top = gen.linear.weight.grad.norm()
            gen_bottom = gen.main[-2].weight.grad.norm()

            gen_optimizer.step()

            # saving metrics
            gen_loss_history.append(gen_loss.item())
            real_loss_history.append(real_loss.item())
            fake_loss_history.append(fake_loss.item())
            discr_loss_history.append(discr_loss.item())
            D_x_history.append(D_x)
            D_G_z1_history.append(D_G_z1)
            D_G_z2_history.append(D_G_z1)
            discr_top_grad.append(discr_top.item())
            discr_bottom_grad.append(discr_bottom.item())
            gen_top_grad.append(gen_top.item())
            gen_bottom_grad.append(gen_bottom.item())

            end = time.time()
            print("[Time %d s][Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [D(x): %f, D(G(Z)): %f / %f]"
            % (end-start, epoch + 1, args.num_epochs, i+1, len(dataloader), discr_loss, gen_loss, D_x, D_G_z1, D_G_z2))

        if args.save and (epoch % 20 == 0):
            gen_file_name = 'gen_params'+date+'.pth'
            discr_file_name = 'discr_params'+date+'.pth'
            torch.save(gen.state_dict(), ckp_dir / gen_file_name)
            torch.save(disc.state_dict(), ckp_dir / discr_file_name)
            with torch.no_grad():
                fake = gen(fixed_noise).detach().cpu().numpy()
                # scipy.io.wavfile.write(prod_dir / ("epoch" + str(epoch) + ".wav"), 16000, fake.T )
                scipy.io.wavfile.write(prod_dir / (date + "epoch" + str(epoch) + ".wav"), 16000, fake.T )

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
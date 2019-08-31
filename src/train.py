import argparse
import time
import datetime
import torch
from dataset import MusicDataset, collate, RandomCrop, OneHotEncoding, ToTensor
from model import Generative, Discriminative, train_batch, weights_init
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import json
from pathlib import Path
# create the parser
parser = argparse.ArgumentParser(description='Train the CSP GAN')

# seed
parser.add_argument('--seed',            type=int, default=30,    help=' Seed for the generation process')
parser.add_argument('--gen_lr',          type=float, default=0.0002,    help=' Generator\'s learning rate')
parser.add_argument('--discr_lr',        type=float, default=0.0001,    help=' Generator\'s learning rate')

parser.add_argument('--batch_size',          type=int, default=16,    help='Dimension of the batch')
parser.add_argument('--num_epochs',             type=int, default=5,    help='Number of epochs')

parser.add_argument('--save',             type=bool, default=True,    help='Save the generator and discriminator models')
parser.add_argument('--out_dir',          type=str, default='models/',    help='Folder where to save the model')
parser.add_argument('--ckp_dir',          type=str, default='ckps',    help='Folder where to save the model')
parser.add_argument('--metrics_dir',          type=str, default='metrics/',    help='Folder where to save the model')

if __name__ == '__main__':
    
    # Parse input arguments
    args = parser.parse_args()

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

    gen.apply(weights_init)
    disc.apply(weights_init)

    seq_len = 16000 * 8
    subseq_len = 65536
    trans = transforms.Compose([RandomCrop(seq_len, subseq_len),
                                OneHotEncoding(),
                                ToTensor()
                                ])
    # load data
    dataset = MusicDataset("dataset/FMA/dataset_pcm_8000/Rock", transform=trans)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate(), shuffle=True)

   
    # test training
    gen_optimizer = torch.optim.Adam(gen.parameters(), lr=args.gen_lr, betas=(0.5, 0.999))
    disc_optimizer = torch.optim.SGD(disc.parameters(), lr=args.discr_lr)

    adversarial_loss = torch.nn.BCELoss()

    # initializing weights
    print("Start training")
    gen_loss_history = []
    discr_loss_history = []
    D_x_history = []
    D_G_z1_history = []
    D_G_z2_history = []

    discr_top_grad =[]
    discr_bottom_grad=[]
    gen_top_grad = []
    gen_bottom_grad = []

    fixed_noise = torch.randn((args.batch_size, 1, 256))
    song_list=[]


    date = datetime.datetime.now()
    date = date.strftime("%d-%m-%Y,%H-%M-%S")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.save:
        (out_dir / args.ckp_dir).mkdir(parents=True, exist_ok=True)

    for epoch in range(args.num_epochs):

        # Iterate batches
        for i, batch_sample in enumerate(dataloader):

            # moving to device
            # batch = batch_sample.to(device)
            batch = batch_sample

            # Update network
            start = time.time()

            gen_loss, discr_loss, D_x, D_G_z1, D_G_z2, discr_top, discr_bottom, gen_top, gen_bottom = train_batch(gen, disc, \
                batch_sample, adversarial_loss, disc_optimizer, gen_optimizer)

            # saving metrics
            gen_loss_history.append(gen_loss)
            discr_loss_history.append(discr_loss)
            D_x_history.append(D_x)
            D_G_z1_history.append(D_G_z1)
            D_G_z2_history.append(D_G_z1)
            discr_top_grad.append(discr_top)
            discr_bottom_grad.append(discr_bottom)
            gen_top_grad.append(gen_top)
            gen_bottom_grad.append(gen_bottom)

            end = time.time()
            print("[Time %d s][Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [D(x): %f, D(G(Z)): %f / %f]"
            % (end-start, epoch + 1, args.num_epochs, i+1, len(dataloader), discr_loss, gen_loss, D_x, D_G_z1, D_G_z2))

            if (epoch % 25 == 0):
                with torch.no_grad():
                    fake = gen(fixed_noise).detach().numpy().tolist()
            song_list.append(fake)

            if args.save and (epoch % 20 == 0):
                gen_file_name = 'gen_params'+date+'.pth'
                discr_file_name = 'discr_params'+date+'.pth'
                torch.save(gen.state_dict(), out_dir / args.ckp_dir / gen_file_name)
                torch.save(disc.state_dict(), out_dir / args.ckp_dir / discr_file_name)
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
            'discr_loss':discr_loss_history, 
            'D_x':D_x_history, 
            'D_G_z1':D_G_z1_history,
            'D_G_z2':D_G_z2_history, 
            'gen_top':gen_top_grad, 
            'gen_bottom':gen_bottom_grad,
            'discr_top':discr_top_grad,
            'discr_bottom':discr_bottom_grad,
            'song_list':song_list}

    # Save metrics
    metrics_dir = Path(args.metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metric_file_name = 'metrics'+ date +'.json'
    with open(metrics_dir / metric_file_name, 'w') as f:
        json.dump(metrics, f, indent=4)
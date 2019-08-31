# -*- coding: utf-8 -*-

from torch import nn
import numpy as np

class Generative(nn.Module):
    def __init__(self, input_size, hidden_sizes, layers_num, dropout_prob):

        super(Generative, self).__init__()

        self.frame_size = [8, 2, 1]

        first_tier_size = 64
        self.first_tier = [nn.LSTM(input_size = 1,
                                hidden_size = hidden_sizes[0],
                                num_layers = layers_num[0],
                                dropout = dropout_prob,
                                batch_first = False) for tier in range(first_tier_size)]
        print('First tier RNNs:', len(self.first_tier), self.first_tier[0])
        

        self.second_tier_input = nn.Linear(1, 256, bias=False)
        self.second_linear_hidden = nn.Linear(256, 256, bias =False)

        
        second_tier_size = 256
        self.second_tier = [nn.LSTM(input_size = 256,
                                hidden_size = 128,
                                num_layers = 2,
                                dropout = dropout_prob,
                                batch_first = False) for tier in range(second_tier_size)]

        print('Second tier RNNs:', len(self.second_tier), self.second_tier[0])
        

        self.third_linear_input = nn.Linear(1, 512, bias=False)
        self.third_linear_hidden = nn.Linear(256, 512, bias=False)

        third_tier_size = 512
        self.third_tier = [nn.Sequential(
                            nn.Linear(512, 1024),
                            nn.ReLU(),
                            nn.Linear(1024, 1024),
                            nn.ReLU(),
                            # TODO: output layer must have dimension 256 - categorical out 1hot encoded
                            nn.Linear(1024,1),
                            nn.ReLU(),
                            nn.Softmax(dim=0)) for tier in range(third_tier_size)]

        print('Third tier MLP:', len(self.third_tier), self.third_tier[0])

    def forward(self, x, state=None):

      
        #####################################
        #  first tier: macro sequences level
        #####################################
        first_tier_out = []
        states_first_tier = []
        i = 0
    
        for net in self.first_tier:

            # each RNN receive only its time steps input
            input_to_net = x[i * self.frame_size[0]:i * self.frame_size[0] + self.frame_size[0], :, :]
            # the first net of the tier does not have any input from previous time steps
            if ( i == 0):
                out, hidden = net(input_to_net, state)
            else:
                out, hidden = net(input_to_net, (states_first_tier[i-1], torch.zeros(states_first_tier[i-1].shape)))
            # append the output of RNNs to get the tier output
            first_tier_out.append(out[-1])
            # pass the hidden state to following RNN
            states_first_tier.append(hidden[-1])
            i +=1

        # each RNN resumes the its time steps into the output
        first_tier_out = torch.stack(first_tier_out)

        # upsampling of size ratio
        ratio = 4
        # (seq_len, batch_size, channels)
        states_first_tier = first_tier_out.squeeze().repeat(1,1,ratio).view(-1, x.shape[1], 256)
        print("first tier shape: ", states_first_tier.shape)

     

        #####################################
        #  second tier: sub-sequences level
        #####################################

        # pass macro-sequence information to lower tier
        states_first_tier = self.second_linear_hidden(states_first_tier)

        # linear layer for input sequence
        seq_to_second_tier = self.second_tier_input(x)
        
        second_tier_out = []
        states_second_tier = []
        i = 0
        for net in self.second_tier:
            # each RNN receive only its time steps input + long term dependecies from higher tier
            input_to_net = torch.sum(seq_to_second_tier[i * self.frame_size[1]:i * self.frame_size[1] + self.frame_size[1], :, :], 0) +\
                 states_first_tier[i, :, :]

            input_to_net = torch.unsqueeze(input_to_net, 0)

            # the first net of the tier does not have any input from previous time steps
            if ( i == 0):
                out, hidden = net(input_to_net)
            else:
                out, hidden = net(input_to_net, (states_second_tier[i-1], torch.zeros(states_second_tier[i-1].shape)))
            # append the output of RNNs to get the tier output
            second_tier_out.append(out[-1])
            # pass the hidden state to following RNN
            states_second_tier.append(hidden[-1])
            i +=1

        # each RNN resumes the its time steps into the output
        second_tier_out = torch.stack(second_tier_out)

        # upsampling of size ratio
        ratio = 4
        # (seq_len, batch_size, channels)
        states_second_tier = second_tier_out.squeeze().repeat(1,1,ratio).view(-1, x.shape[1], 128)

        print("second_tier shape:", states_second_tier.shape)
        #####################################
        #  third tier: per sample lavel
        #####################################
        
        # linear layers
        # pass sub-sequences information to lower tier
        second_tier_out = self.third_linear_hidden(states_second_tier)

        # linear layer for input sequence
        seq_to_third_tier = self.third_linear_input(x)

        third_tier_out = []
        i = 0
        for net in self.third_tier:
            input_to_net = torch.sum(seq_to_third_tier[i * self.frame_size[2]:i * self.frame_size[2] + self.frame_size[2], :, :], 0) +\
                 second_tier_out[i, :, :]
        
            input_to_net = torch.squeeze(input_to_net)

            out =  net(input_to_net)

            third_tier_out.append(out[-1])
            i += 1

         # each RNN resumes the its time steps into the output
        third_tier_out = torch.stack(third_tier_out)
        print("third_tier_out", third_tier_out.shape)

        return third_tier_out
        

class Discriminative(nn.Module):
    
    def __init__(self, input_size, hidden_units, layers_num, dropout_prob=0):

        super().__init__()

        self.rnn = nn.LSTM(input_size = input_size,
                        hidden_size = hidden_units,
                        num_layers = layers_num,
                        dropout=dropout_prob,
                        batch_first= False)
        # self.rnn2 = nn.LSTM(input_size = 128,
        #                 hidden_size = 1,
        #                 num_layers = layers_num,
        #                 dropout=dropout_prob,
        #                 batch_first= False)
        self.linear = nn.Linear(128, 1)
        
        # self.out = nn.Linear(hidden_units, 1)

    def forward(self, x, state=None):
        # LSTM
        print("before forward", x.shape)
        x, rnn_state = self.rnn(x, state)
        print("before forward", x.shape)
        x = self.linear(x)
        print("after forward", x.shape)



        # Linear layer
        # x = self.out(x)
        return out, rnn_state

def train_batch(gen, disc, batch, loss, disc_optimizer, gen_optimizer):

    source_real_data = batch['source']
    target_real_data = batch['target']

    print(source_real_data.shape)

    print(source_real_data.shape)

    batch_size = source_real_data.shape[1]

    # data labels
    labels_real = torch.ones(batch_size)
    labels_fake = torch.zeros(batch_size)

    ###################################
    ## Train the generator
    ###################################

    gen_optimizer.zero_grad()

    # TODO: implement TBPTT passing hidden state from last layer, last time step
    # to next subsequence

    # adversarial loss
    # adding channel dimension (seq_len, batch, channels)
    fake_target = gen(torch.unsqueeze(source_real_data, dim=-1))
    print("fake target", fake_target.shape)

    # adding channel dimension (seq_len, batch, channels)
    fake_target = torch.unsqueeze(fake_target, dim=-1)
    gan_loss = loss['gan']
    discriminator = disc['target']
    output = discriminator(fake_target)
    gan2target_loss = gan_loss(output, labels_real)
    fake_source = gen(target_real_data)
    gan2source_loss = loss['gan'](disc['source'], labels_real)

    gan_loss = (gan2target_loss + gan2source_loss) / 2


    # cycle loss
    reconstruct_source = gen(fake_target)
    cycle_source_loss = loss['cycle'](fake_source, source)
    reconstruct_target = gen(fake_source)
    cycle_target_loss = loss['cycle'](fake_target, target)

    cycle_loss = (cycle_source_loss + cycle_target_loss) / 2

    # Total loss
    gen_total_loss = gan_loss + lambda_cyc * cycle_loss

    gen_total_loss.backward()
    gen_optimizer.step()



    ###################################
    ## Train discriminator target
    ###################################

    disc_optimizer['target'].zero_grad()
    target_real_loss = loss['gan'](disc['target'](target_real_data, labels_real))

    # TODO: fake loss with replay buffer, train the discriminator on previous generated samples

    target_fake_loss = loss['gan'](disc['target'](fake_target, labels_fake))
    target_disc_total_loss = (target_real_loss + target_fake_loss) / 2

    disc_optimizer['target'].backward()
    disc_optimizer['target'].step()

    ###################################
    ## Train discriminator source
    ###################################

    disc_optimizer['source'].zero_grad()
    source_real_loss = loss['gan'](disc['source'](source_real_data, labels_real))

    # TODO: fake loss with replay buffer, train the discriminator on previous generated samples
    source_fake_loss = loss['gan'](disc['source'](fake_source, labels_fake))
    source_disc_total_loss = (source_real_loss + source_fake_loss) / 2

    disc_optimizer['source'].backward()
    disc_optimizer['source'].step()

    print ("Gen error: %f, Disc-target error: %f, Disc-source error: ", gen_total_loss, target_disc_total_loss, source_disc_total_loss)
    
if __name__=='__main__':

    from dataset import MusicDataset, ToTensor, collate, Crop, RandomCrop
    from torch.utils.data import Dataset, DataLoader
    import torch
    from torchvision import transforms

    # generative model params
    gen_input_size = 1
    gen_hidden_units = [256]
    gen_hidden_layers = [2]
    gen_dropout_prob = 0.4

    # set up the generator network
    gen = Generative(gen_input_size, gen_hidden_units, gen_hidden_layers, gen_dropout_prob)

    # discriminative model params
    disc_input_size = 1
    disc_hidden_units = [128, 128]
    disc_hidden_layers = [2, 2]
    disc_dropout_prob = 0.4

    # set up the discriminative models
    disc_target = Discriminative(disc_input_size, disc_hidden_units[0], disc_hidden_layers[0], disc_dropout_prob)
    disc_source = Discriminative(disc_input_size, disc_hidden_units[1], disc_hidden_layers[1], disc_dropout_prob)

    disc = {'target':disc_target, 'source': disc_source}

    seq_len = 16000 * 8
    subseq_len = 32000
    trans = transforms.Compose([RandomCrop(seq_len, subseq_len),
                                ToTensor()
                                ])
    # load data
    dataset = MusicDataset("dataset/FMA/dataset_pcm_8000/Rock", "dataset/FMA/dataset_pcm_8000/Hip-Hop", transform=trans)
    dataloader = DataLoader(dataset, batch_size=5, collate_fn=collate(), shuffle=True)

    # Test the network output
    for i, batch_sample in enumerate(dataloader):
        sample = batch_sample
        print (batch_sample['source'].shape)

        if i == 3:
            break

    # input_tensor = sample['source']
    # print(input_tensor.shape)
    # input_tensor = torch.unsqueeze(input_tensor, -1)
    # print(input_tensor.shape)

    input_tensor = torch.randn([512, 5, 1])
    out = gen(input_tensor)
    # print(out.shape)

    # out.backward()
    # print(out.grad)

    # out, rnn_state = disc_target(input_tensor)
    # print(out.shape)

    # # print(out.shape)
    # # print(rnn_state[0].shape)
    # # print(rnn_state[1].shape)

    # test training
    gen_optimizer = torch.optim.Adam(gen.parameters())
    disc_target_optimizer = torch.optim.Adam(disc['target'].parameters())
    disc_source_optimizer = torch.optim.Adam(disc['source'].parameters())

    disc_optimizer = {'target': disc_target_optimizer, 'source': disc_source_optimizer}

    adversarial_loss = torch.nn.MSELoss()
    cycle_loss = torch.nn.L1Loss()

    loss = {'gan': adversarial_loss, 'cycle': cycle_loss}

    # Test the network output
    for i, batch_sample in enumerate(dataloader):
        train_batch(gen, disc, batch_sample, loss, disc_optimizer, gen_optimizer)

        if i == 3:
            break
    

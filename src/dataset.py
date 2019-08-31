# -*- coding: utf-8 -*-

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import os
from pathlib import Path
import numpy as np
from torchvision import transforms


class MusicDataset(Dataset):

    def __init__(self, source_filepath, transform = None):

        source_folder = Path(source_filepath)

        # get songs' path and keep it in two different arrays
        songs = []
        for root, dirs, files in os.walk(source_folder):
            for name in files:
                songs.append(os.path.join(root, name))


        self.transform = transform
        self.songs = songs
        
    def __len__(self):

        # we are working with unpaired songs, we could have different sizes
        return len(self.songs)

    def __getitem__(self, idx):
        
        song_path = self.songs[idx % len(self.songs)]
       
        # load songs as pulse code modulation
        song = song_loader(song_path)

        sample = song

        if self.transform:
            sample = self.transform(sample)

        return sample

def song_loader(path):
    song = np.memmap(path, dtype="int8", mode='r')
    return song.tolist()

# TODO: split dataset into same-length chunks to have a better management,
# deal with the last segment that have lower size -> variable size sequences
class RandomCrop():
    def __init__(self, seq_len, subseq_len):
        self.seq_len = seq_len
        self.subseq_len = subseq_len

    def __call__(self, sample):

        starting_point = np.random.randint(0, self.seq_len - self.subseq_len +1)
        chunk = sample[starting_point:starting_point + self.subseq_len]

        print("After crop shape: ", len(chunk))
        return chunk

def encode_one_hot(sample):
    encoded = np.zeros((sample.shape[0], 256))
    for i in range(sample.shape[0]):
        encoded[i, sample[i] + 127] = 1
    return encoded

class OneHotEncoding():
    # with mu-law we have values bewteen -127 and 128, therefore to pass to a one-hot encoding we must
    # set to 1 the element associated to the indicated value: value + 127.
    def __call__(self, sample):

        sample = np.array(sample)
        encoded_onehot = encode_one_hot(sample)
        print("encode_one_hot shape", encoded_onehot.shape)


        return encoded_onehot

# class Crop():
#     def __init__(self, seq_len, subseq_len):
#         self.seq_len = seq_len
#         self.subseq_len = subseq_len

#     def __call__(self, sample):
#         source = sample['source']
#         target = sample['target']


#         # chunking source 
#         n_seq = len(source) // self.seq_len
#         n_subseq = self.seq_len // self.subseq_len
#         chunked_source = np.zeros((n_seq, n_subseq, self.subseq_len))

#         for seq in range(n_seq):
#             for subseq in range(n_subseq):
#                 startpoint = seq * self.seq_len + subseq * self.subseq_len
#                 endpoint = seq * self.seq_len + (subseq + 1) * self.subseq_len

#                 # sequence length not divisible by subseq length
#                 # if (endpoint > len(source)):
#                 #     endpoint = len(source)

#                 chunked_source[seq, subseq, :] = source[startpoint:endpoint]
#         # chunking target
#         n_seq = len(target) // self.seq_len
#         n_subseq = self.seq_len // self.subseq_len

#         chunked_target = np.zeros((n_seq, n_subseq, self.subseq_len))

#         for seq in range(n_seq):
#             for subseq in range(n_subseq):
#                 startpoint = seq * self.seq_len + subseq * self.subseq_len
#                 endpoint = seq * self.seq_len + (subseq + 1) * self.subseq_len

#                 # sequence length not divisible by subseq length
#                 # if (endpoint > len(source)):
#                 #     endpoint = len(source)

#                 chunked_target[seq, subseq, :] = target[startpoint:endpoint]


#         return {'source': chunked_source, 'target': chunked_target}

        

class ToTensor():
    def __call__(self, sample):
        # Convert songs to tensor
        tensor = torch.tensor(sample).float()

        print("after to tensor:", tensor.shape)
        return tensor


class collate():
    def __call__(self, batch):

        # we need to collate the batch 
        # in the considered batch each source sequence must have the same size
        # pad with respect to max source sequence size, the same for target sequences. Stack together

        # pad source sequences
        sequences = [seq for seq in batch]
        seq_lengths = [len(seq) for seq in sequences]
        padded_sequences = pad_sequence(sequences, batch_first=False)

       

        # stack together source and target sequences
        return padded_sequences

if __name__ == "__main__":

    # since sampling rate is 16 KHz we want 8 seconds audio files 
    seq_len = 16000 * 8
    subseq_len = 32000
    trans = transforms.Compose([RandomCrop(seq_len, subseq_len),
                                OneHotEncoding(),
                                ToTensor()
                                ])
    # test dataset
    dataset = MusicDataset("dataset/FMA/dataset_pcm_8000/Rock", "dataset/FMA/dataset_pcm_8000/Hip-Hop", transform=trans)
    sample = dataset[0]

    # test dataloader
    dataset = MusicDataset("dataset/FMA/dataset_pcm_8000/Rock", "dataset/FMA/dataset_pcm_8000/Hip-Hop",transform=trans)
    print(dataset[0])

    
    dataloader = DataLoader(dataset, batch_size=2, collate_fn= collate(), shuffle=True)
    
    for i_batch, batch_sample in enumerate(dataloader):
        print(i_batch, batch_sample['source'].shape, batch_sample['target'].shape)

        if (i_batch == 3):
            break 
       
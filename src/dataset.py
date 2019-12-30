# -*- coding: utf-8 -*-

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torchaudio
import os
from torchvision import transforms
from pathlib import Path
import numpy as np
import scipy
import math
import time



class MusicDataset(Dataset):

    def __init__(self, source_filepath, seq_len=512, normalize=True, transform = None):

        source_folder = Path(source_filepath)
        print(source_folder)
        self.seq_len = seq_len
        # get songs' path
        songs = []
        for root, dirs, files in os.walk(source_folder):
            for name in files:
                songs.append(os.path.join(root, name))
        
        # let's restrict to wav files (damn .DS_Store)
        songs = [song for song in songs if song.endswith('.wav')]



        # self.transform = transform
        # self.songs = songs

        # we cannot waste songs - let's analyze the entire dataset and divide songs into chunks
        # each chunk will be a sample - notice that we could need to load multiple times the same file

        # notice that each song could have different number of chunks, let's save path and idx of chunk
        dataset_chunks = []

        for song in songs:

            data = torchaudio.info(song)
            # get the number of chunks in the audio file
            n_chunks = int((data[0].length/data[0].channels)/self.seq_len)

            # for each audiofile save the chunks - easy to retrieve
            for idx in range(n_chunks):
                dataset_chunks.append({"path":song, "idx":idx})

        self.dataset_chunks = dataset_chunks
        self.transform = transform

        print("Dataset loaded: %d songs, %d chunks." % (len(songs), len(dataset_chunks)))
        
    def __len__(self):

        # we are working with unpaired songs, we could have different sizes
        return len(self.dataset_chunks)


    def __getitem__(self, idx):
        
        
        chunk_info = self.dataset_chunks[idx]
        # # load the song
        sample = song_loader(chunk_info, self.seq_len)

        # # # get the selected chunk
        # # chunks = torch.split(song, self.seq_len, dim=-1)
        
        # # # print(song_path['idx'])
        # # sample = chunks[song_path['idx']]

        if self.transform:
            sample = self.transform(sample)


        return sample

def song_loader(chunk_info, seq_len, normalize=True):
    # song = np.memmap(path, dtype="int8", mode='r')

    # sampling_rate, song = scipy.io.wavfile.read(path, mmap=False)

    # data is loaded as a tensor
    song, _ = torchaudio.load(chunk_info['path'], normalization=True, num_frames=seq_len, offset=seq_len*chunk_info['idx'])
   

    return song

class Normalize():

    def __call__(self, sample):
        sample = sample - sample.mean()
        sample = sample / sample.abs().max()

        return sample

class ToChunks():
    def __init__(self, seq_len):
        self.seq_len = seq_len

    def __call__(self, sample):

        chunks = torch.split(sample, self.seq_len)

        return chunks

class Crop_and_pad():
    def __init__(self, target_len):
        self.target_len = target_len

    def __call__(self, sample):

        sample_len = len(sample)
        mid_point = sample_len // 2
 
        if(sample_len > self.target_len):

            # center crop
            start = mid_point - (self.sample_len//2)
            chunk = sample[start:start + self.target_len]
        elif(sample_len < self.target_len):

            # zero padding to target
            start = (self.target_len - sample_len)//2
            end = int(math.ceil((self.target_len - sample_len)/2))
            # print(start, end)
            chunk = np.array(sample)
            chunk = np.pad(chunk, (start, end), 'linear_ramp')

        # print(chunk)
        return chunk


def encode_one_hot(sample):
    sample = torch.squeeze(sample)
    encoded = torch.zeros((sample.shape[0], 256))
    for i in range(sample.shape[0]):
        encoded[i, int(sample[i])] = 1
    return encoded

class OneHotEncoding():
    # with mu-law we have values bewteen -127 and 128, therefore to pass to a one-hot encoding we must
    # set to 1 the element associated to the indicated value: value + 127.
    def __call__(self, sample):

        encoded_onehot = encode_one_hot(sample)

        return encoded_onehot

class ToMulaw():

    def __call__(self, sample):

        return torchaudio.transforms.MuLawEncoding()(sample)

class ToTensor():
    def __call__(self, sample):
        # Convert songs to tensor
        tensor = torch.tensor(sample).float()
        

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

    # since sampling rate is 16 KHz we want sample_length milliseconds audio files 
    sample_length = 5000
    seq_len = 16 * sample_length

    normalize = True
    trans = ToMulaw()

    # test dataloader
    dataset = MusicDataset("dataset/maestro_mono",
                                        seq_len = seq_len,
                                        normalize = normalize,
                                        transform=trans)

    
    dataloader = DataLoader(dataset, batch_size=15, collate_fn= collate(), shuffle=True)

    start = time.time()
    for i_batch, batch_sample in enumerate(dataloader):
        print(i_batch, batch_sample.shape)
        end = time.time()
        print("time: ", end-start)
        start = time.time()
        if (i_batch == 3):
            break 
       
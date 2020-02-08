#!/usr/bin/env python
# -*- coding: utf-8 -*-

# TODO: check imports
import torch
from torch.utils.data import Dataset, DataLoader
# from torch.nn.utils.rnn import pad_sequence
import torchaudio
# import librosa
import os
from pathlib import Path
import numpy as np
# import scipy
import pescador
# import math
import time



def generate_rnd_chunk(track_path, track_len, seq_len, normalize=True, transform=None):
    """Split the stream into fixed size chunks starting from an initial random samples
    
    Args:
        track_path (String): path of source file.
        track_len (int): length of the source file.
        seq_len (int): length of the chunks. Default: 512. 
        normalize (bool): set if the normalization is required. Default: ``True``.
        transform (Transform): tranformation to apply to samples. Default: ``None``.
    
    Yields:
        chunk of the desired size.
    """

    # choose random initial sample
    rnd = torch.randint(track_len - seq_len, (1,))
    rnd = int(rnd)

    # load selected chunk
    sample, _ = torchaudio.load(track_path, normalization=normalize, num_frames=seq_len, offset=rnd)
    sample = sample[0].type(torch.float32)
    sample = torch.unsqueeze(sample, dim=0)

    # apply transformation
    if transform:
        sample = transform(sample)
    yield sample


def generate_chunk(track_path, track_len, seq_len, hop, normalize=True, transform=None):
    """Split the stream into fixed size chunks with a moving window of desired
        length and hop.
    
    Args:
        track_path (String): path of source file.
        track_len (int): length of the source file.
        seq_len (int): length of the chunks.
        hop (int): hop step between adjacent windows.      
        normalize (bool): set if the normalization is required. Default: ``True``.
        transform (Transform): tranformation to apply to samples. Default: ``None``.
    
    Yields:
        chunk of the desired size.
    """

    for idx in range(0, track_len - seq_len, hop):

        # load selected chunk
        sample, _ = torchaudio.load(track_path, normalization=normalize, num_frames=seq_len, offset=idx)
        sample = sample[0].type(torch.float32)
        sample = torch.unsqueeze(sample, dim=0)

        # apply transformation
        if transform:
            sample = transform(sample)
        yield sample


# let's build an iterable-style dataset - exploit random chunks over the files
class MusicDataset(torch.utils.data.IterableDataset):
    """Music dataset: get a dataset with audio files used as multiple source streams.
        Streams are divided into chunks on the fly by a generating function.
    
    Args:
        source_filepath (String): root folder of the dataset.
        seq_len (int): length of the chunks. Default: 512.
        hop (int): hop step between adjacent windows.
        normalize (bool): set if the normalization is required. Default: ``True``.
        transform (Transform): tranformation to apply to samples. Default: ``None``.
        restart_streams (bool): set if exhausted streams are re-activated. Default: ``False``.
    """

    def __init__(self, source_filepath, seq_len=512, hop=None, normalize=True, transform=None, restart_streams=False):
        super(MusicDataset).__init__()
        source_folder = Path(source_filepath)
        self.seq_len = seq_len
        
        if hop==None:
            hop = seq_len
        
        self.hop = hop

        self.normalize = normalize
        self.transform = transform
    

        # get songs' path
        songs = []
        for root, dirs, files in os.walk(source_folder):
            for name in files:
                songs.append(os.path.join(root, name))
        
        # let's restrict to wav files (damn .DS_Store)
        songs = [song for song in songs if song.endswith('.wav')]

        
        # get songs length
        data = []
        for song in songs:
            # get audio info
            song_info = torchaudio.info(song)
            data.append({"path":song, "len":int(song_info[0].length/song_info[0].channels)})

        self.data = data


        # muxing different streams
        if restart_streams:
            streams = [pescador.Streamer(generate_rnd_chunk, track['path'], track['len'], seq_len, normalize, transform) for track in data]
            self.mux = pescador.ShuffledMux(streams)
        else:
            streams = [pescador.Streamer(generate_chunk, track['path'], track['len'], seq_len, hop, normalize, transform) for track in data]
            self.mux = pescador.StochasticMux(streams, len(streams), rate=None, mode='exhaustive')


    def __iter__(self):
        return self.mux.iterate()


    def __len__(self):
        num_batches = 0
        for song in self.data:
            batches = int((song['len'] - self.seq_len)/self.hop)
            num_batches += batches
        return num_batches
        

class Normalize():
    """Normalize a sample removing the mean and squashing it between [-1,1].

    Args:
        sample (tensor): sample to normalize.
    
    Returns:
        normalized tensor.
    """

    def __call__(self, sample):
        sample = sample - sample.mean()
        sample = sample / sample.abs().max()

        return sample


def encode_one_hot(sample):
    """Get a one hot encoding.

    Args:
        sample (tensor): sample to transform.
    
    Returns:
        one hot encoded tensor.
    """

    # with mu-law we have values bewteen -127 and 128, therefore to pass to a one-hot encoding we must
    # set to 1 the element associated to the indicated value: value + 127.
    sample = torch.squeeze(sample)
    encoded = torch.zeros((sample.shape[0], 256))
    for i in range(sample.shape[0]):
        encoded[i, int(sample[i])] = 1
    return encoded

class OneHotEncoding():
    """ Apply one-hot encoding to the tensor.

    Args:
        sample (tensor): tensor to transform with one-hot encoding.
    
    Returns:
        the one-hot encoded tensor.
    """
    def __call__(self, sample):

        encoded_onehot = encode_one_hot(sample)

        return encoded_onehot

class ToMulaw():
    """ Apply Mulaw quantization to the sample.

    Args:
        sample (tensor): tensor to transform with Mulaw quantization.
    
    Returns:
        the quantized sample.
    """

    def __call__(self, sample):

        # sample = torchaudio.transforms.MuLawEncoding()(sample)
        sample = torchaudio.functional.mu_law_encoding(sample, 256)
        return sample

class ToTensor():
    """ Transform to a tensor.

    Args:
        sample (ndarray, list): sample to wrap in a tensor.

    Returns:
        the tensor version of the input.
    """
    def __call__(self, sample):
        # Convert songs to tensor
        tensor = torch.tensor(sample).float()
        

        return tensor



if __name__ == "__main__":

    from torchvision import transforms

    normalize = True
    trans = transforms.Compose([ToTensor(),ToMulaw(), OneHotEncoding()])

    # test dataloader
    dataset = MusicDataset("dataset/piano_f32le/training", seq_len=16384, normalize=normalize, transform=trans)
    dataloader = DataLoader(dataset, batch_size=10)


    start = time.time()
    for i_batch, batch_sample in enumerate(dataloader):
        print(i_batch, batch_sample.shape)
        end = time.time()
        print("time: ", end-start)
        start = time.time()
        if (i_batch == 3):
            break 
       
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import torch
import librosa

def save_models(gen, disc, date, prod_dir, gen_file_name, disc_file_name):
    """Save the model parameters

    Args:
        gen (Generative): generator of the WGAN.
        disc (Discriminative): critic of the WGAN.
        date (String): date for the name of the model to save.
        prod_dir (String): directory path where to save files
        gen_file_name (String): generator model name
        disc_file_name (String): discriminator model name
    """
    print("Backing up the discriminator and generator models")
    torch.save(gen.state_dict(), prod_dir / gen_file_name)
    torch.save(disc.state_dict(), prod_dir / disc_file_name)

def sample_fake(gen, latent, date, prod_dir, epoch=0):
    """Generate samples from the latent and save them as wav files

    Args:
        gen (Generative): generator of the WGAN.
        latent (Tensor): tensor of the noise to map.
        date (String): date for the name of the samples to save
        epoch (int): epoch of generation.
        prod_dir (Path): directory path where to save files
    """

    with torch.no_grad():
        print("Sampling from generator distribution: store samples for inspection.")

        # get the samples from the generated distribution
        fake = gen(latent).detach().cpu()

        # get the target folder
        target_folder = prod_dir/ (date + "_" + str(epoch))
        target_folder.mkdir(parents=True, exist_ok=True)

        # process each single sample
        for idx, sample in enumerate(torch.split(fake, 1, dim=0)):

            sample = torch.squeeze(sample)
            target_path  = target_folder / (str(idx) + ".wav")
            sample = sample.numpy()

            # write the wav file
            librosa.output.write_wav(target_path, sample, 16000)


def collect_and_save(prod_dir, **kwargs):
    """Collect given metrics and save them as a JSON file

    Args:
        prod_dir (Path): directory path where to save the JSON file
        kwargs (dictionary): dictionary of metrics to save
    """
    metrics = kwargs
    print("Saving current metrics")
    metric_file_name = 'metrics.json'
    with open(prod_dir / metric_file_name, 'w') as f:
        json.dump(metrics, f, indent=4)

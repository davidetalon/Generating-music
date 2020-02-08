# AttentionWaveGAN

Pytorch implementation of WaveGAN models ([Donahue et al., 2018](https://arxiv.org/pdf/1802.04208.pdf)).
Following the approach of SAGAN([Zhang et al., 2018](https://arxiv.org/pdf/1805.08318.pdf)), a pair of attention layers has been added to both the generator and discriminator. Improvement are observed on the long-term structure representation of audio modality.

Generated samples are available on [SoundCloud](https://soundcloud.com/davidetalon/sets/raw-audio-generation).

## Requirements
This code requires following packages which could be installed via `conda` or `pip`:

* `pytorch`
* `torchaudio`
* `librosa`
* `pescador`


## Dataset

You can train the model on audio datasets with long sequences. You can use any folder containing audio given that a training and a validation folder are present.
Tests has been performed with Bach piano dataset properly encoded with 32-bit floating point PCM.

- [Bach piano performances](http://deepyeti.ucsd.edu/cdonahue/wavegan/data/mancini_piano.tar.gz)

## Train the model

Here is how you can start model training from the root directory. Set model and training parameters in `train.sh`. In particular you need to specify the dataset directory with `--data_dir`. Then you can run the code:

```

bash src/train.sh
```

To load a model, move both `gen_params.pth` and `discr_params.pth` under the same folder. Set it as model folder with `--model_folder`.

Explore other parameters with:
```
python src/train.py --help
```



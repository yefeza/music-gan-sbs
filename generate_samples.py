import numpy as np
import tensorflow as tf
from tensorflow import keras
from base import WGAN
# audio manipulation
import librosa
import os
import soundfile as sf
# arguments command line   
import argparse

# Size of the noise vector
NOISE_SHAPE = (1, 4000)

# get arguments from command line
parser = argparse.ArgumentParser()

parser.add_argument("--model", help="path to model", type=str)
parser.add_argument("--output", help="path to output", type=str)
parser.add_argument("--n_samples", help="number of samples to generate", type=int, default=1)
parser.add_argument("--number_seconds", help="number of seconds of each sample", type=int, default=4)

args = parser.parse_args()

# load model
model = keras.models.load_model(args.model)
output_path = args.output

if output_path[-1] != '/':
    output_path += '/'

# generate samples
for i in range(args.n_samples):
    random_latent_vectors = tf.random.normal(shape=(1, NOISE_SHAPE[0], NOISE_SHAPE[1]))
    sound_track = []
    for j in range(args.number_seconds-1):
        generated_samples = model(random_latent_vectors, training=False)
        sample = generated_samples[0].numpy()
        sound_track.append(sample[0])
        random_latent_vectors = generated_samples
    # reshape the sound track to be a 1D array
    sound_track = np.array(sound_track).reshape(-1)
    # create a directory for the generated samples
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
    # reshape from (4000,) to (4000, 1)
    reshaped = sound_track.reshape(-1, 1)
    # save the generated samples as a wav file
    sf.write(output_path+'generated_samples_{}.wav'.format(i), reshaped, 4000, 'PCM_24')
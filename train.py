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

# Size of the samples inputs
SAMPLE_SHAPE = (2, 4000)
# Size of the noise vector
NOISE_SHAPE = (1, 4000)
# Size of the gen input vector
GEN_INPUT_SHAPE = (2, 4000)

def conv_block(x, filters, activation, kernel_size=(4, 4), strides=(2, 2)):
    x = keras.layers.Conv2D(filters, kernel_size, strides=strides, padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = activation(x)
    return x

def get_discriminator_model():
    sample_input = keras.layers.Input(shape=SAMPLE_SHAPE)
    x = keras.layers.Reshape((2, 4000, 1))(sample_input)
    x = conv_block(x, 1024, activation=keras.layers.LeakyReLU(0.2), kernel_size=(2, 4))
    x = keras.layers.Dropout(0.2)(x)
    x = conv_block(x, 512, activation=keras.layers.LeakyReLU(0.2), kernel_size=(2, 4))
    x = conv_block(x, 512, activation=keras.layers.LeakyReLU(0.2), kernel_size=(2, 4))
    x = keras.layers.Dropout(0.2)(x)
    x = conv_block(x, 256, activation=keras.layers.LeakyReLU(0.2), kernel_size=(2, 4))
    x = conv_block(x, 256, activation=keras.layers.LeakyReLU(0.2), kernel_size=(2, 4))
    x = keras.layers.Dropout(0.2)(x)
    x = conv_block(x, 128, activation=keras.layers.LeakyReLU(0.2), kernel_size=(2, 4))
    x = conv_block(x, 128, activation=keras.layers.LeakyReLU(0.2), kernel_size=(2, 4))
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(1)(x)
    d_model = keras.models.Model(sample_input, x, name="discriminator")
    return d_model

def get_generator_model():
    noise = keras.layers.Input(shape=GEN_INPUT_SHAPE)
    x = keras.layers.Dense(4000)(noise)
    x = keras.layers.Reshape((4, 2000, 1))(x)
    # deconv block
    x = keras.layers.Conv2DTranspose(256, (4, 4), strides=(1, 1), padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    # x = keras.layers.Activation("relu")(x)
    # deconv block
    x = keras.layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    # x = keras.layers.Activation("relu")(x)
    # deconv block
    x = keras.layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    # x = keras.layers.Activation("relu")(x)
    # deconv block
    x = keras.layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    # x = keras.layers.Activation("relu")(x)
    # conv block
    x = keras.layers.Conv2D(32, (4, 4), strides=(2, 2), padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    # conv block
    x = keras.layers.Conv2D(2, (4, 4), strides=(8, 2), padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    # conv block
    x = keras.layers.Conv2D(1, (28, 28), strides=(1, 1), padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    # dense layer
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(8000)(x)
    # output normalization
    x = keras.layers.LayerNormalization()(x)
    x = keras.layers.Reshape(GEN_INPUT_SHAPE)(x)
    g_model = keras.models.Model(noise, x, name="generator")
    return g_model

# Define the loss functions for the discriminator.
def discriminator_loss(real_sample, fake_sample):
    real_loss = tf.reduce_mean(real_sample)
    fake_loss = tf.reduce_mean(fake_sample)
    return fake_loss - real_loss

# Define the loss functions for the generator.
def generator_loss(fake_sample):
    return -tf.reduce_mean(fake_sample)

# Define the callback for saving generated samples during training.
class GANMonitor(keras.callbacks.Callback):
    def __init__(self, num_samples=10, latent_dim=NOISE_SHAPE, start_epoch=0):
        self.num_samples = num_samples
        self.latent_dim = latent_dim
        self.start_epoch = start_epoch

    def on_epoch_end(self, epoch, logs=None):
        epoch += self.start_epoch
        epoch += 1
        random_latent_vectors = tf.random.normal(shape=(1, self.latent_dim[0], self.latent_dim[1]))
        historical_mem = tf.zeros(shape=(1, self.latent_dim[0], self.latent_dim[1]))
        random_latent_vectors = tf.concat([random_latent_vectors, historical_mem], axis=1)
        sound_track = []
        for i in range(self.num_samples-1):
            generated_samples = self.model.generator(random_latent_vectors, training=False)
            sample = generated_samples[0, :, :].numpy()
            sound_track.append(sample[0])
            random_latent_vectors = generated_samples
        # reshape the sound track to be a 1D array
        sound_track = np.array(sound_track).reshape(-1)
        # create a directory for the generated samples
        if not os.path.exists('data/generated_samples'):
            os.makedirs('data/generated_samples', exist_ok=True)
        # reshape from (4000,) to (4000, 1)
        reshaped = sound_track.reshape(-1, 1)
        # save the generated samples as a wav file
        sf.write('data/generated_samples/generated_samples_epoch_{}.wav'.format(epoch), reshaped, 4000, 'PCM_24')
        # save the model on h5 format every 10 epochs and on the first epoch
        if not os.path.exists('data/models'):
            os.makedirs('data/models', exist_ok=True)
        if epoch % 5 == 0:
            self.model.generator.save('data/models/generator_epoch_{}.h5'.format(epoch))
            self.model.discriminator.save('data/models/discriminator_epoch_{}.h5'.format(epoch))

# main function
def main():
    # get arguments from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train the model')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size for training')
    parser.add_argument('--d_extra_steps', type=int, default=3, help='number of discriminator extra steps')
    parser.add_argument('--load_model', type=bool, default=False, help='load model from h5 file')
    parser.add_argument('--checkpoint', type=str, default=None, help='checkpoint file to load model from')
    parser.add_argument('--input_path', type=str, default='data/input/', help='path to input data')

    args = parser.parse_args()
    epochs = args.epochs
    batch_size = args.batch_size
    d_extra_steps = args.d_extra_steps
    load_model = args.load_model
    checkpoint = args.checkpoint
    input_path = args.input_path

    d_model = get_discriminator_model()
    d_model.summary()
    g_model = get_generator_model()
    g_model.summary()

    start_epoch = 0
    if load_model:
        g_model = keras.models.load_model('data/models/generator_epoch_{0}.h5'.format(checkpoint))
        d_model = keras.models.load_model('data/models/discriminator_epoch_{0}.h5'.format(checkpoint))
        print('loaded model from epoch {0}'.format(checkpoint))
        start_epoch = int(checkpoint)

    wgan = WGAN(
        discriminator=d_model,
        generator=g_model,
        latent_dim=NOISE_SHAPE,
        discriminator_extra_steps=d_extra_steps,
    )
    # Compile the wgan model
    wgan.compile(
        d_optimizer=keras.optimizers.Adam(learning_rate=0.0003, beta_1=0.5, beta_2=0.9),
        g_optimizer=keras.optimizers.Adam(learning_rate=0.0003, beta_1=0.5, beta_2=0.9),
        g_loss_fn=generator_loss,
        d_loss_fn=discriminator_loss,
    )
    # Load the dataset
    input_samples = []
    split_length = 4000
    all_files=os.listdir(input_path)
    for file in all_files:
        if os.path.isfile(input_path+file):
            sample_wav, sr = librosa.load(input_path+file, sr=None)
            resampled_wav = librosa.resample(sample_wav, orig_sr=sr, target_sr=4000)
            # save resampled wav
            # reshape from (4000,) to (4000, 1)
            reshaped = resampled_wav.reshape(-1, 1)
            sf.write('data/resampled{}.wav'.format(file), reshaped, 4000, 'PCM_24')
            # insert a start sample at the beginning of the input samples with the same length as the other samples with zeros
            input_samples.append(np.zeros(split_length))
            # get array with split samples
            for i in range(0, len(resampled_wav), split_length):
                input_samples.append(resampled_wav[i:i+split_length])
                # pad with zeros if the last sample is not the same length as the others
                if len(input_samples[-1]) != split_length:
                    input_samples[-1] = np.pad(input_samples[-1], (0, split_length - len(input_samples[-1])), 'constant')
            if len(input_samples) % 2 != 0:
                input_samples = np.append(input_samples, np.zeros(split_length).reshape(1, split_length), axis=0)
            input_samples=list(input_samples)
    # make input samples as a list of 2
    input_samples = [input_samples[i:i + 2] for i in range(len(input_samples) - 1)]
    print(np.array(input_samples).shape)
    # convert to numpy array
    input_samples = np.array(input_samples)
    # convert to Tensor
    input_samples = tf.convert_to_tensor(input_samples, dtype=tf.float32)
    # make the inpiut samples a multiple of the batch size
    input_samples = input_samples[:-(len(input_samples) % batch_size)]
    print('input samples shape: {}'.format(input_samples.shape))
    # train the model
    cbk = GANMonitor(num_samples=31, latent_dim=NOISE_SHAPE, start_epoch=start_epoch)
    wgan.fit(input_samples, batch_size=batch_size, epochs=epochs, callbacks=[cbk])

if __name__ == '__main__':
    main()
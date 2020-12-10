import os
import pickle

import numpy as np
from PIL import Image
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import LeakyReLU
from keras.layers import Reshape
from keras.models import Sequential
from keras.optimizers import Adam
from matplotlib import pyplot
from numpy import ones
from numpy import zeros
from numpy.random import randint
from numpy.random import randn
from skimage.transform import resize


# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# define the standalone discriminator model


def define_discriminator(in_shape=(240, 160, 3)):
    model = Sequential()
    # normal
    model.add(Conv2D(64, (3, 3), padding='same', input_shape=in_shape))
    model.add(LeakyReLU(alpha=0.2))
    # model.add(BatchNormalization())
    # down-sample
    model.add(Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # model.add(BatchNormalization())
    # down-sample
    model.add(Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # model.add(BatchNormalization())
    # down-sample
    model.add(Conv2D(256, (3, 3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # model.add(BatchNormalization())
    model.add(Conv2D(256, (3, 3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # model.add(BatchNormalization())
    # classifier
    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid'))
    # compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model


# define the standalone generator model
def define_generator(latent_dim):
    model = Sequential()
    # foundation for 4x4 image
    n_nodes = 256 * 30 * 20
    model.add(Dense(n_nodes, input_dim=latent_dim))
    # model.add(LeakyReLU(alpha=0.2))
    # model.add(Dropout(0.4))
    # model.add(Dense(n_nodes,activation='relu'))
    model.add(Reshape((30, 20, 256)))
    # up-sample to 8x8
    model.add(Conv2DTranspose(256, (4, 4), activation='relu', strides=(2, 2), padding='same'))
    # model.add(BatchNormalization())
    # up-sample to 16x16
    model.add(Conv2DTranspose(128, (4, 4), activation='relu', strides=(2, 2), padding='same'))
    # model.add(BatchNormalization())
    # up-sample to 32x32
    model.add(Conv2DTranspose(64, (4, 4), activation='relu', strides=(2, 2), padding='same'))
    # output layer
    model.add(Conv2D(3, (3, 3), activation='tanh', padding='same'))
    return model


# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model):
    # make weights in the discriminator not trainable
    d_model.trainable = False
    # connect them
    model = Sequential()
    # add generator
    model.add(g_model)
    # add the discriminator
    model.add(d_model)
    # compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model


# load and prepare training images
def load_real_samples():
    prefix_dir = 'book-covers'
    dataset = []
    for directory in os.listdir(prefix_dir):
        for file in os.listdir(prefix_dir + '/' + directory):
            if file.endswith(".jpg") and len(dataset) < 2000:
                dataset.append(
                    resize(np.asarray(Image.open(os.path.join(prefix_dir, os.path.join(directory, file)))),
                           (240, 160, 3)))
                if len(dataset) % 100 == 0:
                    print("Number of images loaded in memory: " + str(len(dataset)))
    for i in range(49):
        # define subplot
        pyplot.subplot(7, 7, 1 + i)
        # turn off axis
        pyplot.axis('off')
        # plot raw pixel data
        pyplot.imshow(dataset[i])
    # X = dataset.astype('float32')
    # # scale from [0,255] to [-1,1]
    # X = (X - 127.5) / 127.5
    return np.asarray(dataset)


def load_real_images_from_directory(name):
    prefix_dir = 'book-covers-no-text\\'
    dataset = []
    for file in os.listdir(prefix_dir + name):
        if file.endswith(".jpg"):
            dataset.append(
                resize(np.asarray(Image.open(os.path.join(prefix_dir, os.path.join(name, file)))), (240, 160, 3)))
            if len(dataset) % 100 == 0:
                print("Number of images loaded in memory: " + str(len(dataset)))
    for i in range(49):
        # define subplot
        pyplot.subplot(7, 7, 1 + i)
        # turn off axis
        pyplot.axis('off')
        # plot raw pixel data
        pyplot.imshow(dataset[i])
    # X = dataset.astype('float32')
    # # scale from [0,255] to [-1,1]
    # X = (X - 127.5) / 127.5
    return np.asarray(dataset)


# select real samples
def generate_real_samples(dataset, n_samples):
    # choose random instances
    ix = randint(0, dataset.shape[0], n_samples)
    # retrieve selected images
    x = dataset[ix]
    # generate 'real' class labels (1)
    y = ones((n_samples, 1))
    return x, y


# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
    # generate points in the latent space
    x_input = randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input


# use the generator to generate n fake examples, with class labels
def generate_fake_samples(g_model, latent_dim, n_samples):
    # generate points in latent space
    x_input = generate_latent_points(latent_dim, n_samples)
    # predict outputs
    x = g_model.predict(x_input)
    # create 'fake' class labels (0)
    y = zeros((n_samples, 1))
    return x, y


# create and save a plot of generated images
def save_plot(examples, epoch, n=5):
    # scale from [-1,1] to [0,1]
    examples = (examples + 1) / 2.0
    # plot images
    for i in range(n * n):
        # define subplot
        pyplot.subplot(n, n, 1 + i)
        # turn off axis
        pyplot.axis('off')
        # plot raw pixel data
        pyplot.imshow(examples[i])
    # save plot to file
    filename = 'generated_plot_e%03d.png' % (epoch + 1)
    pyplot.savefig(filename)
    pyplot.close()


# evaluate the discriminator, plot generated images, save generator model
def summarize_performance(epoch, g_model, d_model, dataset, latent_dim, n_samples=150):
    # prepare real samples
    x_real, y_real = generate_real_samples(dataset, n_samples)
    # evaluate discriminator on real examples
    _, acc_real = d_model.evaluate(x_real, y_real, verbose=0)
    # prepare fake examples
    x_fake, y_fake = generate_fake_samples(g_model, latent_dim, n_samples)
    # evaluate discriminator on fake examples
    _, acc_fake = d_model.evaluate(x_fake, y_fake, verbose=0)
    # summarize discriminator performance
    print('>Accuracy real: %.0f%%, fake: %.0f%%' % (acc_real * 100, acc_fake * 100))
    # save plot
    save_plot(x_fake, epoch)
    with open('generator_model_' + FOLDER_NAME + '_BIG.pickle', 'wb') as handle:
        pickle.dump(g_model, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('discriminator_model_' + FOLDER_NAME + '_BIG.pickle', 'wb') as handle:
        pickle.dump(d_model, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('gan_model_' + FOLDER_NAME + '_BIG.pickle', 'wb') as handle:
        pickle.dump(g_model, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # save the generator model tile file
    # filename = 'generator_model_%03d.h5' % (epoch + 1)
    # g_model.save(filename)


# train the generator and discriminator
def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=2000, n_batch=16):
    bat_per_epo = int(dataset.shape[0] / n_batch)
    half_batch = int(n_batch / 2)
    # manually enumerate epochs
    for i in range(n_epochs):
        # enumerate batches over the training set
        for j in range(bat_per_epo):
            # get randomly selected 'real' samples
            x_real, y_real = generate_real_samples(dataset, half_batch)
            # update discriminator model weights
            d_loss1, _ = d_model.train_on_batch(x_real, y_real)
            # generate 'fake' examples
            x_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
            # update discriminator model weights
            d_loss2, _ = d_model.train_on_batch(x_fake, y_fake)
            # prepare points in latent space as input for the generator
            x_gan = generate_latent_points(latent_dim, n_batch)
            # create inverted labels for the fake samples
            y_gan = ones((n_batch, 1))
            # update the generator via the discriminator's error
            g_loss = gan_model.train_on_batch(x_gan, y_gan)
            # summarize loss on this batch
            print('>%d, %d/%d, d1=%.3f, d2=%.3f g=%.3f' %
                  (i + 1, j + 1, bat_per_epo, d_loss1, d_loss2, g_loss))
        # evaluate the model performance, sometimes
        if (i + 1) % 10 == 0:
            summarize_performance(i, g_model, d_model, dataset, latent_dim)


# size of the latent space
LATENT_DIM = 200
# create the discriminator
# FOLDER_NAME = 'Art-Photography'
FOLDER_NAME = 'Romance'
if os.path.isfile("discriminator_new_model_" + FOLDER_NAME + "_BIG.pickle"):
    with open('discriminator_new_model_' + FOLDER_NAME + '_BIG.pickle', 'rb') as pickle_file:
        D_MODEL = pickle.load(pickle_file)
else:
    D_MODEL = define_discriminator()
# create the generator
if os.path.isfile("generator_new_model_" + FOLDER_NAME + "_BIG.pickle"):
    with open("generator_new_model_" + FOLDER_NAME + "_BIG.pickle", 'rb') as pickle_file:
        G_MODEL = pickle.load(pickle_file)
else:
    G_MODEL = define_generator(LATENT_DIM)
# create the gan
GAN_MODEL = define_gan(G_MODEL, D_MODEL)
# load image data
# DATASET = load_real_samples()
DATASET = load_real_images_from_directory(FOLDER_NAME)
# train model
train(G_MODEL, D_MODEL, GAN_MODEL, DATASET, LATENT_DIM)

# # Uncomment to plot the model architecture:
# plot_model(G_MODEL, to_file='generator.png')
# plot_model(D_MODEL, to_file='discriminator.png')
# plot_model(GAN_MODEL, to_file='gan.png')

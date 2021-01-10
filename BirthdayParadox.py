import pickle
from numpy.random import randn
from matplotlib import pyplot


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
    return x


def save_plot(examples, epoch,start, n=5):
    # scale from [-1,1] to [0,1]
    # examples = (examples + 1) / 2.0
    # plot images
    for i in range(start, start+n * n):
        # define subplot
        pyplot.subplot(n, n, 1 + i - start)
        # turn off axis
        pyplot.axis('off')
        # plot raw pixel data
        pyplot.imshow(examples[i])
    # save plot to file
    filename = 'plots/generated_plot_%03d.png' % (epoch + 1)
    pyplot.savefig(filename)
    pyplot.close()


def save_images(examples, n=5):
    # scale from [-1,1] to [0,1]
    # examples = (examples + 1) / 2.0
    # plot images
    for i in range(n):
        # turn off axis
        pyplot.axis('off')
        # plot raw pixel data
        pyplot.imshow(examples[i])
        # save plot to file
        filename = 'images/generated_image_%03d.png' % (i)
        pyplot.savefig(filename)
        pyplot.close()


# with open("generator_model_Art-Photography_BIG.pickle", 'rb') as pickle_file:
#     G_MODEL = pickle.load(pickle_file)
with open("generator_model_Graphic-Novels-Anime-Manga_BIG.pickle", 'rb') as pickle_file:
    G_MODEL = pickle.load(pickle_file)

samples = generate_fake_samples(G_MODEL, 200, 100)
for i in range(4):
    save_plot(samples, i, i*25)
save_images(samples, 100)

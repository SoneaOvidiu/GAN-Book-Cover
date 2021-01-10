import os
import numpy as np
from skimage.transform import resize
from PIL import Image
from matplotlib import pyplot
import pickle
from numpy.random import randn

def load_real_samples(name):
    prefix_dir = 'book-covers-no-text\\'
    dataset = []
    for file in os.listdir(prefix_dir + name):
        if file.endswith(".jpg"):
            dataset.append(
                resize(np.asarray(Image.open(os.path.join(prefix_dir, os.path.join(name, file)))), (240, 160, 3)))
            if len(dataset) % 100 == 0:
                print("Number of images loaded in memory: " + str(len(dataset)))
    return np.asarray(dataset)


def generate_latent_points(latent_dim, n_samples):
    # generate points in the latent space
    x_input = randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input

def load_fake_samples(g_model, latent_dim, n_samples):
    # generate points in latent space
    x_input = generate_latent_points(latent_dim, n_samples)
    x = g_model.predict(x_input)
    return x


def average_pooling(img, stride, window_size):
    new_img = []
    for i in range(0, img.shape[0] - window_size, stride):
        new_img.append([])
        for j in range(0, img.shape[1] - window_size, stride):
            j2 = i + window_size
            z = j + window_size
            mean_ = img[i:j2, j:z, :]
            new_img[i // stride].append(np.mean(np.mean(mean_, axis=0), axis=0))
    return np.array(new_img)


def save_plot(img1, img2, name):
    pyplot.subplot(1, 2, 1)
    pyplot.axis('off')
    pyplot.imshow(img1)
    pyplot.subplot(1, 2, 2)
    pyplot.axis('off')
    pyplot.imshow(img2)
    pyplot.savefig(name)
    pyplot.close()


def nearest_neighbour_distance(img1, img2):
    distance = 0
    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            distance += abs(img1[i][j][1] - img2[i][j][1])
            distance += abs(img1[i][j][2] - img2[i][j][2])
            distance += abs(img1[i][j][0] - img2[i][j][0])
    return distance / (img1.shape[0] * img1.shape[1] * img1.shape[2])


FOLDER_NAME='Graphic-Novels-Anime-Manga'
latent_dim=200
with open("generator_model_" + FOLDER_NAME + "_BIG.pickle", 'rb') as pickle_file:
    G_MODEL = pickle.load(pickle_file)
print("Load real Images")
real_dataset = load_real_samples(FOLDER_NAME)
print("Faking Images")
fake_dataset = load_fake_samples(G_MODEL, latent_dim, 100)
new_real_dataset = []
new_fake_dataset = []
print("Just making some average pooling for the real dataset")
i=0
for elem in real_dataset:
    i+=1
    new_real_dataset.append(average_pooling(elem, 2, 4))
    if i%100==0:
        print(str(i)+"/"+str(len(real_dataset)))
print("Just making some average pooling for the fake dataset")
i=0
for elem in fake_dataset:
    i+=1
    new_fake_dataset.append(average_pooling(elem, 2, 4))
    if i%100==0:
        print(str(i)+"/"+str(len(fake_dataset)))
print("Who are my neighbours? Those bastards...")
nearest_neighbour = [-1] * len(new_fake_dataset)
min_dist = [2] * len(new_fake_dataset)
for i in range(len(new_fake_dataset)):
    for j in range(len(new_real_dataset)):
        val = nearest_neighbour_distance(new_fake_dataset[i], new_real_dataset[j])
        if val < min_dist[i]:
            min_dist[i] = val
            nearest_neighbour[i] = j
    print(str(i)+"/"+str(len(new_fake_dataset)))
    save_plot(fake_dataset[i], real_dataset[nearest_neighbour[i]], "nearestNeighbourPlots/" + str(i) + ".png")

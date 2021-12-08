import tensorflow as tf
import numpy as np
import os
import tensorflow.keras.layers as layers
import time
import matplotlib.pyplot as plt
from imageio import v3 as iio

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
noise_dim = 100

BUFFER_SIZE = 60000
BATCH_SIZE = 256


def generator_loss(fake_example):
    return cross_entropy(tf.ones_like(fake_example), fake_example)

def discriminator_loss(real_example, fake_example):
    fake_ce = cross_entropy(tf.zeros_like(fake_example), fake_example)
    real_ce = cross_entropy(tf.ones_like(real_example), real_example)

    return fake_ce + real_ce
 
# def generator_loss(fake_output):
#     return cross_entropy(tf.ones_like(fake_output), fake_output)

# def discriminator_loss(real_output, fake_output):
#     real_loss = cross_entropy(tf.ones_like(real_output), real_output)
#     fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
#     total_loss = real_loss + fake_loss
#     return total_loss

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

def mnist_generator():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(7 * 7 * 128, input_shape=(noise_dim,), use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(tf.keras.layers.Reshape(( 7, 7, 128)))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.UpSampling2D(2))
    model.add(tf.keras.layers.Conv2DTranspose(128,4, padding='same', use_bias=False, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Conv2DTranspose(64,4, padding='same', use_bias=False, activation='relu'))
    model.add(tf.keras.layers.UpSampling2D(2))
    model.add(layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Conv2DTranspose(1, 6, padding='same', use_bias=False, activation='tanh'))
    return model

def mnist_discriminator():
    model = tf.keras.Sequential()
    # model.add(tf.keras.layers.Conv2D(64, 4, padding='same', input_shape=(28, 28, 1)))
    model.add(tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',activation='relu',
                                     input_shape=[28, 28,1]))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Conv2D(128, (5,5), strides=(2,2), padding='same', activation='relu'))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.1))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1))

    return model

# generator = mnist_generator()
# generator.summary()

# discriminator = mnist_discriminator()
# discriminator.summary()

# noise = tf.random.normal([1, 100])
# generated_image = generator(noise, training=False)

# plt.imshow(generated_image[0, :, :, 0], cmap='gray')
# plt.show()

# decision = discriminator(generated_image)
# tf.print (decision)
# discriminator.summary()

# checkpoint_dir = './training_checkpoints'
# checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
# checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
#                                 discriminator_optimizer=discriminator_optimizer,
#                                 generator=generator,
#                                 discriminator=discriminator)

# You will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
seed = tf.random.normal([100, noise_dim])

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

@tf.function
def train_step(generator, discriminator, images, batch_size):
    noise = tf.random.normal([batch_size, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(noise, training=True)

      real_output = discriminator(images, training=True)
      fake_output = discriminator(generated_images, training=True)

      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

def train(generator, discriminator, dataset, epochs, batch_size):
    for epoch in range(epochs):
        start = time.time()
        n = len(dataset)
        for (i, image_batch) in enumerate(dataset):
            # image_batch = image_batch.reshape(-1,28,28,1)
            train_step(generator, discriminator, image_batch, batch_size)
        print("EPOCH finished: ", epoch, " duration: ", time.time() - start)

def load_mnist_dataset():
    (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
    train_images = train_images.reshape(-1, 28, 28, 1).astype('float32')
    train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]
    return tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)


def train_and_show_mnist():
    train_dataset = load_mnist_dataset()
    train(train_dataset, 50)

    plt.figure()
    noise = tf.random.normal([10,100])
    test_images = generator(noise, training=False)
    for i in range (0,10):
        plt.subplot(2, 5, i + 1)
    
        plt.imshow(test_images[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
    plt.show()


import requests
from PIL import Image
from io import BytesIO
import os

import imageio
import matplotlib.pyplot as plt


generation_range = [
    [1, 151], # Gen1
    [152, 251], #Gen2
    [252, 386], # Gen3
    [387, 493], # Gen4
    [494, 649], # Gen5
    [650, 721], # Gen6
    [722, 809]] # Gen7


def get_poke_dataset():
    os.mkdir('pokemon-image', 777)

    # crawling
    for i in range(809):
        r = requests.get(f"https://assets.pokemon.com/assets/cms2/img/pokedex/detail/{i+1:03}.png")
        im = Image.open(BytesIO(r.content))
        im.save(f'./pokemon-image/{i+1:03}.png')
        if i and i % 100 == 0 : 
            print(f"{i+1}th Image Save Compelete")

def create_pokemon_dict(gen, rng):    
    ln = rng[1] - rng[0] + 1
    fig = plt.figure(figsize=(10, (ln + 9)//10), dpi=300)
    for j in range(rng[0], rng[1]+1):
        ax = fig.add_subplot((ln + 9)//10, 10, j-rng[0] + 1)
        im = imageio.imread(f'./pokemon-image/{j:03}.png')
        ax.imshow(im)
        ax.axis('off')
    fig.suptitle(f'Generation {gen} Pokemon', fontweight='bold')


# train_and_show()

# get_poke_dataset()
def load_pokemon():
    ims = []
    for i in range(1, 721):
        im = iio.imread(f'./pokemon/{i}.png', mode='RGB')
        if im.shape == (256, 256, 3): 
            ims.append(im/255.0)

    return np.array(ims)

def pokemon_generator():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(16 * 16 * 512, input_shape=(noise_dim,), use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(tf.keras.layers.Reshape(( 16, 16, 512)))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.UpSampling2D(2))
    model.add(tf.keras.layers.Conv2DTranspose(32,8, padding='same', use_bias=False, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(layers.UpSampling2D(2))
    model.add(tf.keras.layers.Conv2DTranspose(32,8, padding='same', use_bias=False, activation='relu'))
    model.add(tf.keras.layers.UpSampling2D(2))
    model.add(layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Conv2DTranspose(32,8, padding='same', use_bias=False, activation='relu'))
    model.add(tf.keras.layers.UpSampling2D(2))
    model.add(layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Conv2DTranspose(3, 8, padding='same', use_bias=False, activation='relu'))
    return model

def pokemon_discriminator():
    model = tf.keras.Sequential()
    # model.add(tf.keras.layers.Conv2D(64, 4, padding='same', input_shape=(28, 28, 1)))
    model.add(tf.keras.layers.Conv2D(256, 8, strides=(2, 2), padding='same',activation='relu',
                                     input_shape=[256, 256,3]))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Conv2D(256, 8, strides=(2,2), padding='same', activation='relu'))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.1))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1))

    return model


def train_and_show_pokemon(batch_size = 24, epochs=20):
    generator = pokemon_generator() 
    generator.summary()
    discriminator = pokemon_discriminator()
    discriminator.summary()
    pokemon = load_pokemon()
    train_dataset = np.array_split(pokemon, batch_size)
    print("pokemon loaded:", pokemon.shape, "batch size: ", batch_size)
    train(generator, discriminator, train_dataset, epochs, batch_size)
    print("training finished")

    plt.figure()
    noise = tf.random.normal([10,100])
    test_images = generator(noise, training=False)
    for i in range (0,10):
        plt.subplot(2, 5, i + 1)
    
        plt.imshow(test_images[i])
    plt.show()

train_and_show_pokemon(epochs=40)
# noise = tf.random.normal([1, 100])
# generated_image = generator(noise, training=False)

# plt.imshow(generated_image[0, :, :, 0], cmap='gray')
# plt.show()




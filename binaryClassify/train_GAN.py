from GAN import create_generator, create_discriminator
from keras import Input
import os
from keras.models import Model
from tensorflow.keras.optimizers import RMSprop
import time
import numpy as np
from PIL import Image as Img
from matplotlib import pyplot as plt
from data import TrainData
import argparse


def train(LATENT_DIM, WIDTH, HEIGHT, images):
    iters = 100
    batch_size = 20

    RES_DIR = './newfaces'
    FILE_PATH = '%s/generated_%d.png'
    if not os.path.isdir(RES_DIR):
        os.mkdir(RES_DIR)

    CONTROL_SIZE_SQRT = 6
    control_vectors = np.random.normal(size=(CONTROL_SIZE_SQRT**2, LATENT_DIM)) / 2

    start = 0
    d_losses = []
    a_losses = []
    images_saved = 0
    for step in range(iters):
        start_time = time.time()
        latent_vectors = np.random.normal(size=(batch_size, LATENT_DIM))
        generated = generator.predict(latent_vectors)

        real = images[start:start + batch_size]
        combined_images = np.concatenate([generated, real])

        labels = np.concatenate([np.ones((batch_size, 1)), np.zeros((batch_size, 1))])
        labels += .05 * np.random.random(labels.shape)

        d_loss = discriminator.train_on_batch(combined_images, labels)
        d_losses.append(d_loss)

        latent_vectors = np.random.normal(size=(batch_size, LATENT_DIM))
        misleading_targets = np.zeros((batch_size, 1))

        a_loss = gan.train_on_batch(latent_vectors, misleading_targets)
        a_losses.append(a_loss)

        start += batch_size
        if start > images.shape[0] - batch_size:
            start = 0

        if step % 50 == 49:
            gan.save_weights('/gan.h5')

            print('%d/%d: d_loss: %.4f,  a_loss: %.4f.  (%.1f sec)' % (
            step + 1, iters, d_loss, a_loss, time.time() - start_time))

            control_image = np.zeros((WIDTH * CONTROL_SIZE_SQRT, HEIGHT * CONTROL_SIZE_SQRT, CHANNELS))
            control_generated = generator.predict(control_vectors)

            for i in range(CONTROL_SIZE_SQRT ** 2):
                x_off = i % CONTROL_SIZE_SQRT
                y_off = i // CONTROL_SIZE_SQRT
                control_image[x_off * WIDTH:(x_off + 1) * WIDTH, y_off * HEIGHT:(y_off + 1) * HEIGHT,
                :] = control_generated[i, :, :, :]
            im = Img.fromarray(np.uint8(control_image * 255))  # .save(StringIO(), 'jpeg')
            im.save(FILE_PATH % (RES_DIR, images_saved))
            images_saved += 1

    visualize(d_losses, a_losses)


def visualize(d_losses, a_losses):
    plt.figure(1, figsize=(12, 8))
    plt.subplot(121)
    plt.plot(d_losses, color='red')
    plt.xlabel('epochs')
    plt.ylabel('discriminant losses')
    plt.subplot(122)
    plt.plot(a_losses)
    plt.xlabel('epochs')
    plt.ylabel('adversary losses')
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='C:/archive/img_align_celeba/img_align_celeba/')
    args = parser.parse_args()

    data_path = args.data_path
    LATENT_DIM = 32
    CHANNELS = 3
    WIDTH = 128
    HEIGHT = 128

    generator = create_generator(LATENT_DIM, CHANNELS)

    discriminator = create_discriminator(CHANNELS, WIDTH, HEIGHT)
    discriminator.trainable = False



    gan_input = Input(shape=(LATENT_DIM,))
    gan_output = discriminator(generator(gan_input))

    gan = Model(gan_input, gan_output)
    optimizer = RMSprop(learning_rate=.0001, clipvalue=1.0, decay=1e-8)
    gan.compile(optimizer=optimizer, loss='binary_crossentropy')

    images = TrainData(data_path)

    train(LATENT_DIM, WIDTH, HEIGHT, images)
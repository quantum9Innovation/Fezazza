import keras as kr
import numpy as np


# helper functions

def encoder_block(layer_in, filters, batchnorm=True):
    """
    Acts as a basic downsampling block
    Supports batch normalization on most layers

    :param layer_in: previous Keras layer
    :param filters: Number of convolutional filters to run through
    :param batchnorm: (=True) - initializes batch normalization in Keras
    :return: updated Keras layer
    """

    init = kr.initializers.RandomNormal(stddev=0.02)

    # downsampling
    g = kr.layers.Conv2D(filters, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(layer_in)
    if batchnorm:
        g = kr.layers.BatchNormalization()(g, training=True)
    g = kr.layers.LeakyReLU(alpha=0.2)(g)
    
    return g


def decoder_block(layer_in, skips, filters, dropout=True):
    """
    Acts as a basic upsampling block
    Supports dropout

    :param layer_in: previous Keras layer
    :param skips: to support combining previous convolutional layers with similar resolutions
    :param filters: number of convolutional filters
    :param dropout: (=True) - initializes dropout in Keras
    :return: updated Keras layer
    """

    init = kr.initializers.RandomNormal(stddev=0.02)

    # upsampling
    g = kr.layers.Conv2DTranspose(filters, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(layer_in)
    g = kr.layers.BatchNormalization()(g, training=True)
    if dropout:
        g = kr.layers.Dropout(0.5)(g, training=True)
    g = kr.layers.Concatenate()([g, skips])
    g = kr.layers.Activation('relu')(g)

    return g


# neural nets

def define_generator(image_shape=(256, 256, 3)):
    """
    Creates the generator
    Basic U-net architecture:
    E1 (nb) ---------------------------->D7 (nd) -> g (nd) -> tanh
      E2------------------------------>D6 (nd)
        E3-------------------------->D5 (nd)
          E4---------------------->D4 (nd)
            E5------------------>D3
              E6-------------->D2
                E7->B (nb) ->D1

    nd --> no dropout
    nb --> no batchnorm

    :param image_shape: (=(256, 256, 3)) designed for 256x256 image with RGB channels
    :return: generator Keras model
    """

    # initialization
    init = kr.initializers.RandomNormal(stddev=0.02)
    x = kr.models.Input(shape=image_shape)

    # encoder model
    e1 = encoder_block(x, 64, batchnorm=False)
    e2 = encoder_block(e1, 128)
    e3 = encoder_block(e2, 256)
    e4 = encoder_block(e3, 512)
    e5 = encoder_block(e4, 512)
    e6 = encoder_block(e5, 512)
    e7 = encoder_block(e6, 512)

    # bottleneck (no batch norm)
    b = kr.layers.Conv2D(512, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(e7)
    b = kr.layers.Activation('relu')(b)

    # decoder model
    d1 = decoder_block(b, e7, 512)
    d2 = decoder_block(d1, e6, 512)
    d3 = decoder_block(d2, e5, 512)
    d4 = decoder_block(d3, e4, 512, dropout=False)
    d5 = decoder_block(d4, e3, 256, dropout=False)
    d6 = decoder_block(d5, e2, 128, dropout=False)
    d7 = decoder_block(d6, e1, 64, dropout=False)

    # output
    # upsampling
    g = kr.layers.Conv2DTranspose(3, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d7)
    out_image = kr.layers.Activation('tanh')(g)  # activation range [-1, 1]

    # define model
    model = kr.models.Model(x, out_image)

    return model


def define_discriminator(image_shape):
    """
    Create the discriminator
    Constructs a patch-based discriminator which process the image in "patches"
    Each value in the output represents a 70x70 "patch" in the image
    Input images are concatenated together before analysis
    Discriminator learns at 1/2 the rate as the generator

    C64 --> C128 --> C256 --> C512 --> 512 --> Patch Output

    :param image_shape: Any image shape
    :return: Keras discriminator model
    """

    # initialization
    init = kr.initializers.RandomNormal(stddev=0.02)
    x = kr.models.Input(shape=image_shape)  # source image
    y = kr.models.Input(shape=image_shape)  # generator output
    merged = kr.layers.Concatenate()([x, y])

    # C64
    d = kr.layers.Conv2D(64, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(merged)
    d = kr.layers.LeakyReLU(alpha=0.2)(d)

    # C128
    d = kr.layers.Conv2D(128, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = kr.layers.BatchNormalization()(d)
    d = kr.layers.LeakyReLU(alpha=0.2)(d)

    # C256
    d = kr.layers.Conv2D(256, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = kr.layers.BatchNormalization()(d)
    d = kr.layers.LeakyReLU(alpha=0.2)(d)

    # C512
    d = kr.layers.Conv2D(512, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = kr.layers.BatchNormalization()(d)
    d = kr.layers.LeakyReLU(alpha=0.2)(d)

    # second last output layer
    d = kr.layers.Conv2D(512, (4, 4), padding='same', kernel_initializer=init)(d)
    d = kr.layers.BatchNormalization()(d)
    d = kr.layers.LeakyReLU(alpha=0.2)(d)

    # patch output
    d = kr.layers.Conv2D(1, (4, 4), padding='same', kernel_initializer=init)(d)
    patch_out = kr.layers.Activation('sigmoid')(d)

    # define model
    model = kr.models.Model([x, y], patch_out)
    optimizer = kr.optimizers.Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, loss_weights=[0.5])

    return model


def define_gan(g_model, d_model, image_shape):
    """
    Creates the GAN
    Input --> Generator --> Fake
    Input, Fake --> Discriminator --> Likelihood of Fake (in patches)
    Generator (gradient) <-----------/

    :param g_model: generator model
    :param d_model: discriminator model
    :param image_shape: image shape for both discriminator & generator
    :return: compiled GAN model
    """

    # initialization
    d_model.trainable = False
    # ignore training the discriminator
    # (discriminator is already trained in `define_discriminator`)

    x = kr.layers.Input(shape=image_shape)
    
    # generator --> discriminator
    gen_out = g_model(x)
    dis_out = d_model([x, gen_out])
    
    # src image as input, generated image and classification output
    model = kr.models.Model(x, [dis_out, gen_out])
    optimizer = kr.optimizers.Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss=['binary_crossentropy', 'mae'], optimizer=optimizer, loss_weights=[1, 100])
    
    return model


# dataset

def load_real_samples(filename):
    """
    Loads all real samples in pairs (Input, Output)
    Scales all inputs so that mean RGB is 0 and max variance is 127.5

    :param filename: .npz file to extract
    :return: tupled training data
    """

    # loading
    data = np.load(filename)
    x, y = data['arr_0'], data['arr_1']

    # scaling
    x = (x - 127.5) / 127.5
    y = (y - 127.5) / 127.5

    return [x, y]


def generate_real_samples(dataset, samples, patch):
    """
    Prepares input, output pairs for the discriminator

    :param dataset: list of (Input, Output) images
    :param samples: number of samples
    :param patch: patch size
    :return: (src_img, dis_output)
    """

    x, y = dataset
    imgs = np.random.randint(0, x.shape[0], samples)
    x1, y1 = x[imgs], y[imgs]  # get selected images
    z = np.ones((samples, patch, patch, 1))  # create discriminator output (class=1 to indicate real images)

    return [x1, y1], z  # return two inputs `[src_img, real_img]`, `class=1 to indicate real images`


def generate_fake_samples(g_model, samples, patch_shape):
    """
    Prepares fake samples for discriminator by running generator

    :param g_model: generator model
    :param samples: number of samples
    :param patch_shape: patch size
    :return: (fake sample, patch of 0s)
    """

    y = g_model.predict(samples)
    z = np.zeros((len(y), patch_shape, patch_shape, 1))

    return y, z


# TODO: Code review for remaining lines
# TODO: Create class to contain pix2pix model

""" FOR REVIEW
# training

def summarize_performance(step, g_model, dataset, samples=3):
    
    # initialization
    [y, x_re], _ = generate_real_samples(dataset, samples, 1)
    x_fa, _ = generate_fake_samples(g_model, y, 1)
    
    # scaling
    y = (y + 1) / 2.0
    x_re = (x_re + 1) / 2.0
    x_fa = (x_fa + 1) / 2.0
    
    # plot real source images
    for i in range(samples):
        pyplot.subplot(3, samples, 1 + i)
        pyplot.axis('off')
        pyplot.imshow(y[i])
    # plot generated target image
    for i in range(samples):
        pyplot.subplot(3, samples, 1 + samples + i)
        pyplot.axis('off')
        pyplot.imshow(x_fa[i])
    # plot real target image
    for i in range(samples):
        pyplot.subplot(3, samples, 1 + samples*2 + i)
        pyplot.axis('off')
        pyplot.imshow(x_re[i])

    # save plot to file
    filename1 = 'plot_%06d.png' % (step+1)
    pyplot.savefig(filename1)
    pyplot.close()

    # save the generator model
    filename2 = 'model_%06d.h5' % (step+1)
    g_model.save(filename2)

    print('>Saved: %s and %s' % (filename1, filename2))


def train(d_model, g_model, gan_model, dataset, epochs=2, batch_size=1):
    # initialization
    patch_size = d_model.output_shape[1]
    x, y = dataset
    batches = int(len(x) / batch_size)
    n_steps = batches * epochs
    
    # manually enumerate epochs
    for i in range(n_steps):
        # select a batch of real samples
        [x1, y1], z1 = generate_real_samples(dataset, batch_size, patch_size)  # [src img, target img], real dis out
        # generate a batch of fake samples
        y2, z2 = generate_fake_samples(g_model, x1, patch_size)  # fake target img, fake dis out

        # train discriminator
        d_loss1 = d_model.train_on_batch([x1, y1], z1)
        d_loss2 = d_model.train_on_batch([x1, y2], z2)

        # train generator
        g_loss, _, _ = gan_model.train_on_batch(x1, [z1, y1])

        # summarize performance
        print('>%d, d1[%.3f] d2[%.3f] g[%.3f]' % (i+1, d_loss1, d_loss2, g_loss))
        print('Total Loss: %d' % (d_loss1 + d_loss2 + g_loss))
        if (i+1) % (batches * 1) == 0:
            summarize_performance(i, g_model, dataset)


# load images

# dataset
dataset = load_real_samples('maps_256.npz')
print('Loaded', dataset[0].shape, dataset[1].shape)
image_shape = dataset[0].shape[1:]

# models
d_model = define_discriminator(image_shape)
g_model = define_generator(image_shape)

# training
gan_model = define_gan(g_model, d_model, image_shape)
print('Training has begun ...')
train(d_model, g_model, gan_model, dataset)
"""

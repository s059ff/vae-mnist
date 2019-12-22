import chainer
import chainer.functions as F
import chainer.links as L
import chainer.cuda
import cupy as xp
import datetime
import gzip
import numpy as np
import os
import urllib.request

from model import Encoder
from model import Decoder
from visualize import visualize

# Define constants
N = 100     # Minibatch size
M = 70000   # Total samples
C = 1.0     # Reconstruction error factor
SNAPSHOT_INTERVAL = 100


def main():

    # (Make directories)
    os.mkdir('dataset/') if not os.path.isdir('dataset') else None
    os.mkdir('train/') if not os.path.isdir('train') else None

    # (Download dataset)
    if not os.path.exists('dataset/mnist.npy'):
        url = 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'
        response = urllib.request.urlopen(url)
        with open('dataset/train-images-idx3-ubyte.gz', 'wb') as stream:
            stream.write(response.read())
        url = 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz'
        response = urllib.request.urlopen(url)
        with open('dataset/t10k-images-idx3-ubyte.gz', 'wb') as stream:
            stream.write(response.read())
        with gzip.open('dataset/train-images-idx3-ubyte.gz') as stream:
            _ = np.frombuffer(stream.read(), dtype=np.uint8, offset=16).astype('f').reshape((-1, 784))
        with gzip.open('dataset/t10k-images-idx3-ubyte.gz') as stream:
            __ = np.frombuffer(stream.read(), dtype=np.uint8, offset=16).astype('f').reshape((-1, 784))
        train = np.vstack((_, __)) / 255.
        np.save('dataset/mnist', train)
    os.remove('dataset/train-images-idx3-ubyte.gz') if os.path.exists('dataset/train-images-idx3-ubyte.gz') else None
    os.remove('dataset/t10k-images-idx3-ubyte.gz') if os.path.exists('dataset/t10k-images-idx3-ubyte.gz') else None

    # Create samples.
    train = np.load('dataset/mnist.npy').reshape((-1, 784))
    train = np.random.permutation(train)
    train_varidation = train[0:100]

    # Create the model
    enc = Encoder()
    dec = Decoder()

    # (Use GPU)
    chainer.cuda.get_device(0).use()
    enc.to_gpu()
    dec.to_gpu()

    # Setup optimizers
    optimizer_enc = chainer.optimizers.Adam()
    optimizer_dec = chainer.optimizers.Adam()
    optimizer_enc.setup(enc)
    optimizer_dec.setup(dec)

    # (Change directory)
    os.chdir('train/')
    time = datetime.datetime.today().strftime("%Y-%m-%d %H.%M.%S")
    os.mkdir(time)
    os.chdir(time)

    # (Validate input images)
    visualize(train_varidation, 'input.png')

    # Training
    for epoch in range(1000):

        # (Validate generated images)
        if (epoch % SNAPSHOT_INTERVAL == 0):
            os.mkdir('%d' % epoch)
            os.chdir('%d' % epoch)

            x = xp.array(train_varidation)
            μ, σ = enc(x)
            # ε = xp.random.normal(0, 1, (N, 2), dtype='f')
            # z = μ + ε * σ
            z = F.gaussian(μ, σ)
            y = dec(z)
            reconstructed = chainer.cuda.to_cpu(y.data)
            visualize(reconstructed, 'reconstructed.png')

            z = xp.random.normal(0, 1, (N, 2), dtype='f')
            y = dec(z)
            sampled = chainer.cuda.to_cpu(y.data)
            visualize(sampled, 'sampled.png')

            chainer.serializers.save_hdf5("enc.h5", enc)
            chainer.serializers.save_hdf5("dec.h5", dec)

            os.chdir('..')

        # (Random shuffle samples)
        train = np.random.permutation(train)

        total_loss_reg = 0.0
        total_loss_rec = 0.0

        for n in range(0, M, N):

            x = xp.array(train[n:n + N])
            μ, σ = enc(x)
            # ε = xp.random.normal(0, 1, (N, 2))
            # z = μ + ε * σ
            z = F.gaussian(μ, σ)
            y = dec(z)
            
            # loss_reg = F.gaussian_kl_divergence(μ, F.log(σ)) / N
            loss_reg = F.gaussian_kl_divergence(μ, σ) / N
            loss_rec = F.bernoulli_nll(x, y) / N
            loss = loss_reg + C * loss_rec

            enc.cleargrads()
            dec.cleargrads()
            loss.backward()
            optimizer_enc.update()
            optimizer_dec.update()

            total_loss_reg += loss_reg.data
            total_loss_rec += loss_rec.data

        # (View loss)
        total_loss_reg /= M / N
        total_loss_rec /= M / N
        print(epoch, total_loss_reg, total_loss_rec, flush=True)


if __name__ == '__main__':
    main()

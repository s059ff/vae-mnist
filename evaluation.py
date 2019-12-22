import chainer
import chainer.functions as F
import chainer.links as L
import chainer.cuda
import gzip
import itertools
import matplotlib.pylab as plt
import numpy as np
import os
import urllib.request

from model import Encoder
from model import Decoder

M = 70000

def main():

    # (Download dataset)
    if not os.path.exists('dataset/labels.npy'):
        url = 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz'
        response = urllib.request.urlopen(url)
        with open('dataset/train-labels-idx1-ubyte.gz', 'wb') as stream:
            stream.write(response.read())
        url = 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'
        response = urllib.request.urlopen(url)
        with open('dataset/t10k-labels-idx1-ubyte.gz', 'wb') as stream:
            stream.write(response.read())
        with gzip.open('dataset/train-labels-idx1-ubyte.gz') as stream:
            _ = np.frombuffer(stream.read(), dtype=np.uint8, offset=8)
        with gzip.open('dataset/t10k-labels-idx1-ubyte.gz') as stream:
            __ = np.frombuffer(stream.read(), dtype=np.uint8, offset=8)
        labels = np.concatenate((_, __))
        np.save('dataset/labels', labels)
    os.remove('dataset/train-labels-idx1-ubyte.gz') if os.path.exists('dataset/train-labels-idx1-ubyte.gz') else None
    os.remove('dataset/t10k-labels-idx1-ubyte.gz') if os.path.exists('dataset/t10k-labels-idx1-ubyte.gz') else None

    # Create samples.
    train = np.load('dataset/mnist.npy').reshape((-1, 784))
    labels = np.load('dataset/labels.npy')

    # Create the model
    enc = Encoder()
    dec = Decoder()
    chainer.serializers.load_hdf5("model/enc.h5", enc)
    chainer.serializers.load_hdf5("model/dec.h5", dec)

    # (Change directory)
    os.mkdir('evaluation/') if not os.path.isdir('evaluation') else None
    os.chdir('evaluation/')

    # Random sampling
    z = np.random.normal(0, 1, (100, 2)).astype('f')
    y = dec(z)
    sampled = y.data
    plt.figure(num=None, figsize=(20, 20), dpi=100, facecolor='w', edgecolor='k')
    for i in range(0, 100):
        plt.subplot(10, 10, i + 1)
        plt.tick_params(labelleft='off', top='off', bottom='off')
        plt.tick_params(labelbottom='off', left='off', right='off')
        plt.imshow(sampled[i].reshape((28, 28)), cmap='gray', vmin=0.0, vmax=1.0)
    plt.savefig('random-sampling.png')
    plt.close()

    # Linspace sampling
    v1 = np.linspace(3, -3, 20, dtype='f')
    v2 = np.linspace(-3, 3, 20, dtype='f')
    z = np.array(list(itertools.product(v1, v2)), dtype='f')
    _z = z.copy()
    z[:, 0] = _z[:, 1]
    z[:, 1] = _z[:, 0]
    y = dec(z)
    sampled = y.data
    plt.figure(num=None, figsize=(20, 20), dpi=100, facecolor='w', edgecolor='k')
    for i in range(0, 400):
        plt.subplot(20, 20, i + 1)
        plt.tick_params(labelleft='off', top='off', bottom='off')
        plt.tick_params(labelbottom='off', left='off', right='off')
        plt.imshow(sampled[i].reshape((28, 28)), cmap='gray', vmin=0.0, vmax=1.0)
    plt.savefig('linspace-sampling.png')
    plt.close()

    # Z plot
    plt.figure(num=None, figsize=(10, 10), dpi=100)
    for k in range(10):
        x = []
        for m in range(M):
            if (labels[m] == k):
                x.append(train[m])
        x = np.array(x, dtype='f').reshape((-1, 784))
        μ, σ = enc(x)
        z = F.gaussian(μ, σ).data
        z = np.random.permutation(z)
        z1 = z[0:2000, 0]
        z2 = z[0:2000, 1]
        plt.scatter(z1, z2, label=str(k), alpha=0.5)
    plt.xlim((-6, 6))
    plt.ylim((-6, 6))
    plt.legend()
    plt.xlabel('z1')
    plt.ylabel('z2')
    plt.savefig('mapping.png')
    plt.close()

    # Z plot every class
    for k in range(10):
        plt.figure(num=None, figsize=(10, 10), dpi=100)
        for l in range(10):
            if l == k:
                x = []
                for m in range(M):
                    if (labels[m] == k):
                        x.append(train[m])
                x = np.array(x, dtype='f').reshape((-1, 784))
                μ, σ = enc(x)
                z = F.gaussian(μ, σ).data
                z = np.random.permutation(z)
                z1 = z[0:2000, 0]
                z2 = z[0:2000, 1]
                plt.scatter(z1, z2, label=str(k), alpha=0.5)
            else:
                plt.scatter([], [], alpha=0.5)
        plt.xlim((-6, 6))
        plt.ylim((-6, 6))
        plt.legend()
        plt.xlabel('z1')
        plt.ylabel('z2')
        plt.savefig('mapping-%d.png' % k)
        plt.close()

    # Gaussian distribution plot
    plt.figure(num=None, figsize=(10, 10), dpi=100)
    z1 = np.random.normal(0, 1, 1000)
    z2 = np.random.normal(0, 1, 1000)
    plt.scatter(z1, z2, label='N(0,1)', alpha=0.5)
    plt.xlim((-5, 5))
    plt.ylim((-5, 5))
    plt.axes().set_aspect('equal', 'datalim')
    plt.legend()
    plt.xlabel('z1')
    plt.ylabel('z2')
    plt.savefig('normal.png')
    plt.close()

if __name__ == '__main__':
    main()

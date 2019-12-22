import chainer
import chainer.functions as F
import chainer.links as L

class Encoder(chainer.Chain):

    def __init__(self):
        super(Encoder, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(784, 512)
            self.l2_μ = L.Linear(512, 2)
            self.l2_σ = L.Linear(512, 2)

    def __call__(self, x):
        h = x
        h = F.tanh(self.l1(h))
        μ = self.l2_μ(h)
        σ = self.l2_σ(h)
        return μ, σ


class Decoder(chainer.Chain):

    def __init__(self):
        super(Decoder, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(2, 512)
            self.l2 = L.Linear(512, 784)

    def __call__(self, z):
        h = z
        h = F.tanh(self.l1(h))
        y = self.l2(h)
        return y

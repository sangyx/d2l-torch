import collections
import math
import os
import random
import sys
import tarfile
import time
import zipfile

from IPython import display
from matplotlib import pyplot as plt

import torch
from torch import autograd, nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import numpy as np


VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']


VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]


def bbox_to_rect(bbox, color):
    """Convert bounding box to matplotlib format."""
    return plt.Rectangle(xy=(bbox[0], bbox[1]), width=bbox[2]-bbox[0],
                         height=bbox[3]-bbox[1], fill=False, edgecolor=color,
                         linewidth=2)


class Benchmark():
    """Benchmark programs."""
    def __init__(self, prefix=None):
        self.prefix = prefix + ' ' if prefix else ''

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, *args):
        print('%stime: %.4f sec' % (self.prefix, time.time() - self.start))


def corr2d(X, K):
    """Compute 2D cross-correlation."""
    h, w = K.shape
    Y = torch.zeros(X.shape[0] - h + 1, X.shape[1] - w + 1)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i: i + h, j: j + w] * K).sum()
    return Y


# def count_tokens(samples):
#     """Count tokens in the data set."""
#     token_counter = collections.Counter()
#     for sample in samples:
#         for token in sample:
#             if token not in token_counter:
#                 token_counter[token] = 1
#             else:
#                 token_counter[token] += 1
#     return token_counter


def data_iter(batch_size, features, labels):
    """Iterate through a data set."""
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batc_size):
        j = indices[i: min(i + batc_size, num_examples)]
        yield features[j], labels[j]


def data_iter_consecutive(corpus_indices, batch_size, num_steps, device='cpu'):
    corpus_indices = torch.Tensor(corpus_indices)
    corpus_indices = corpus_indices.to(device)
    data_len = len(corpus_indices)
    batch_len = data_len // batch_size
    indices = corpus_indices[0: batch_size * batch_len].reshape(batch_size, batch_len)
    epoch_size = (batch_len - 1) // num_steps
    for i in range(epoch_size):
        i = i * num_steps
        X = indices[:, i: i + num_steps]
        Y = indices[:, i + 1: i + num_steps + 1]
        yield X, Y


def data_iter_random(corpus_indices, batch_size, num_steps, device='cpu'):
    num_examples = (len(corpus_indices) - 1) // num_steps
    epoch_size = num_examples // batch_size
    example_indices = list(range(num_examples))
    random.shuffle(example_indices)
    
    def _data(pos):
        return corpus_indices[pos: pos + num_steps]
    
    for i in range(epoch_size):
        i = i * batch_size
        batch_indices = example_indices[i: i + batch_size]
        X = torch.Tensor([_data(j * num_steps) for j in batch_indices]).to(device)
        Y = torch.Tensor([_data(j * num_steps + 1) for j in batch_indices]).to(device)
        yield X, Y


def download_imdb(data_dir='../data'):
    """Download the IMDB data set for sentiment analysis."""
    url = ('http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz')
    sha1 = '01ada507287d82875905620988597833ad4e0903'
    fname = gutils.download(url, data_dir, sha1_hash=sha1)
    with tarfile.open(fname, 'r') as f:
        f.extractall(data_dir)


def _download_pikachu(data_dir):
    root_url = ('https://apache-mxnet.s3-accelerate.amazonaws.com/'
                'gluon/dataset/pikachu/')
    dataset = {'train.rec': 'e6bcb6ffba1ac04ff8a9b1115e650af56ee969c8',
               'train.idx': 'dcf7318b2602c06428b9988470c731621716c393',
               'val.rec': 'd6c33f799b4d058e82f2cb5bd9a976f69d72d520'}
    for k, v in dataset.items():
        gutils.download(root_url + k, os.path.join(data_dir, k), sha1_hash=v)


def download_voc_pascal(data_dir='../data'):
    """Download the Pascal VOC2012 Dataset."""
    voc_dir = os.path.join(data_dir, 'VOCdevkit/VOC2012')
    url = ('http://host.robots.ox.ac.uk/pascal/VOC/voc2012'
           '/VOCtrainval_11-May-2012.tar')
    sha1 = '4e443f8a2eca6b1dac8a6c57641b67dd40621a49'
    fname = gutils.download(url, data_dir, sha1_hash=sha1)
    with tarfile.open(fname, 'r') as f:
        f.extractall(data_dir)
    return voc_dir


def evaluate_accuracy(data_iter, net, device='cpu'):
    """Evaluate accuracy of a model on the given data set."""
    acc_sum, n = 0, 0
    with torch.no_grad():
        if isinstance(net, nn.Module):
            net.eval()
        for X, y in data_iter:
            X, y = X.to(device), y.to(device)
            acc_sum += float((torch.argmax(net(X), dim=1) == y).sum())
            n += y.size(0)
        if isinstance(net, nn.Module):
            net.train()
    return acc_sum / n


# def _get_batch(batch, ctx):
#     """Return features and labels on ctx."""
#     features, labels = batch
#     if labels.dtype != features.dtype:
#         labels = labels.astype(features.dtype)
#     return (gutils.split_and_load(features, ctx),
#             gutils.split_and_load(labels, ctx), features.shape[0])


# def get_data_ch7():
#     """Get the data set used in Chapter 7."""
#     data = np.genfromtxt('../data/airfoil_self_noise.dat', delimiter='\t')
#     data = (data - data.mean(axis=0)) / data.std(axis=0)
#     return nd.array(data[:, :-1]), nd.array(data[:, -1])


def get_fashion_mnist_labels(labels):
    """Get text label for fashion mnist."""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


# def get_tokenized_imdb(data):
#     """Get the tokenized IMDB data set for sentiment analysis."""
#     def tokenizer(text):
#         return [tok.lower() for tok in text.split(' ')]
#     return [tokenizer(review) for review, _ in data]


# def get_vocab_imdb(data):
#     """Get the vocab for the IMDB data set for sentiment analysis."""
#     tokenized_data = get_tokenized_imdb(data)
#     counter = collections.Counter([tk for st in tokenized_data for tk in st])
#     return text.vocab.Vocabulary(counter, min_freq=5)


def grad_clipping(params, theta, device):
    """Clip the gradient."""
    norm = torch.Tensor([0]).to(device)
    for param in params:
        norm += (param.grad.data ** 2).sum()
    norm = norm.sqrt().item()
    if norm > theta:
        for param in params:
            param.grad.data.mul_(theta / norm)

def linreg(X, w, b):
    """Linear regression."""
    return torch.mm(X, w) + b


def load_data_fashion_mnist(root, batch_size, resize=None, download=False):
    """Download the fashion mnist dataset and then load into memory."""
#     root = os.path.expanduser(root)
    transformer = []
    if resize:
        transformer += [transforms.Resize(resize)]
    transformer += [transforms.ToTensor()]
    transformer = transforms.Compose(transformer)

    mnist_train = datasets.FashionMNIST(root=root, train=True, transform=transformer, download=download)
    mnist_test = datasets.FashionMNIST(root=root, train=False, transform=transformer, download=download)
    num_workers = 0 if sys.platform.startswith('win32') else 4

    train_iter = DataLoader(mnist_train, batch_size, shuffle=True, num_workers=num_workers)
    test_iter = DataLoader(mnist_test, batch_size, shuffle=False, num_workers=num_workers)
    
    return train_iter, test_iter


def load_data_jay_lyrics():
    """Load the Jay Chou lyric data set (available in the Chinese book)."""
    with zipfile.ZipFile('../data/jaychou_lyrics.txt.zip') as zin:
        with zin.open('jaychou_lyrics.txt') as f:
            corpus_chars = f.read().decode('utf-8')
    corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ')
    corpus_chars = corpus_chars[0:10000]
    idx_to_char = list(set(corpus_chars))
    char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])
    vocab_size = len(char_to_idx)
    corpus_indices = [char_to_idx[char] for char in corpus_chars]
    return corpus_indices, char_to_idx, idx_to_char, vocab_size


# def load_data_pikachu(batch_size, edge_size=256):
#     """Download the pikachu dataest and then load into memory."""
#     data_dir = '../data/pikachu'
#     _download_pikachu(data_dir)
#     train_iter = image.ImageDetIter(
#         path_imgrec=os.path.join(data_dir, 'train.rec'),
#         path_imgidx=os.path.join(data_dir, 'train.idx'),
#         batch_size=batch_size,
#         data_shape=(3, edge_size, edge_size),
#         shuffle=True,
#         rand_crop=1,
#         min_object_covered=0.95,
#         max_attempts=200)
#     val_iter = image.ImageDetIter(
#         path_imgrec=os.path.join(data_dir, 'val.rec'),
#         batch_size=batch_size,
#         data_shape=(3, edge_size, edge_size),
#         shuffle=False)
#     return train_iter, val_iter


# def load_data_time_machine():
#     """Load the time machine data set (available in the English book)."""
#     with open('../data/timemachine.txt') as f:
#         corpus_chars = f.read()
#     corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ').lower()
#     corpus_chars = corpus_chars[0:10000]
#     idx_to_char = list(set(corpus_chars))
#     char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])
#     vocab_size = len(char_to_idx)
#     corpus_indices = [char_to_idx[char] for char in corpus_chars]
#     return corpus_indices, char_to_idx, idx_to_char, vocab_size


# def _make_list(obj, default_values=None):
#     if obj is None:
#         obj = default_values
#     elif not isinstance(obj, (list, tuple)):
#         obj = [obj]
#     return obj


# def mkdir_if_not_exist(path):
#     """Make a directory if it does not exist."""
#     if not os.path.exists(os.path.join(*path)):
#         os.makedirs(os.path.join(*path))

def one_hot(idx, size, device='cpu'):
    """Returns a one-hot Tensor."""
    batch_size = idx.size(0)
    index = idx.reshape(-1, 1)
    return torch.zeros(batch_size, size).to(device).scatter_(dim=1, index=index, value=1)

def params_init(model, init, **kwargs):
    """Initialize the parameters."""
    def initializer(m):
        if isinstance(m, nn.Conv2d):
            init(m.weight.data, **kwargs)
            m.bias.data.fill_(0)

        elif isinstance(m, nn.Linear):
            init(m.weight.data, **kwargs)
            m.bias.data.fill_(0)

        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1.0)
            m.bias.data.fill_(0)

        elif isinstance(m, nn.BatchNorm1d):
            m.weight.data.fill_(1.0)
            m.bias.data.fill_(0)

    model.apply(initializer)

def predict_rnn(prefix, num_chars, rnn, params, init_rnn_state,
               num_hiddens, vocab_size, device, idx_to_char, char_to_idx):
    """Predict next chars with a RNN model"""
    state = init_rnn_state(1, num_hiddens, device)
    output = [char_to_idx[prefix[0]]]
    with torch.no_grad():
        for t in range(num_chars + len(prefix) - 1):
            X = to_onehot(torch.Tensor([[output[-1]]]).to(device), vocab_size, device)
            (Y, state) = rnn(X, state, params)
            if t < len(prefix) - 1:
                output.append(char_to_idx[prefix[t + 1]])
            else:
                output.append(int(Y[0].argmax(dim=1).item()))
    return ''.join([idx_to_char[i] for i in output])


def predict_rnn_nn(prefix, num_chars, model, vocab_size, device, idx_to_char,
                  char_to_idx):
    """Precit next chars with a nn RNN model"""
    state = None
    output = [char_to_idx[prefix[0]]]
    with torch.no_grad():
        for t in range(num_chars + len(prefix) - 1):
            X = torch.Tensor([output[-1]]).to(device).reshape(1, 1)
            (Y, state) = model(X, state)
            if t < len(prefix) - 1:
                output.append(char_to_idx[prefix[t + 1]])
            else:
                output.append(int(Y.argmax(dim=1).item()))
    return ''.join([idx_to_char[i] for i in output])


# def predict_sentiment(net, vocab, sentence):
#     """Predict the sentiment of a given sentence."""
#     sentence = nd.array(vocab.to_indices(sentence), ctx=try_gpu())
#     label = nd.argmax(net(sentence.reshape((1, -1))), axis=1)
#     return 'positive' if label.asscalar() == 1 else 'negative'


# def preprocess_imdb(data, vocab):
#     """Preprocess the IMDB data set for sentiment analysis."""
#     max_l = 500

#     def pad(x):
#         return x[:max_l] if len(x) > max_l else x + [0] * (max_l - len(x))

#     tokenized_data = get_tokenized_imdb(data)
#     features = nd.array([pad(vocab.to_indices(x)) for x in tokenized_data])
#     labels = nd.array([score for _, score in data])
#     return features, labels


# def read_imdb(folder='train'):
#     """Read the IMDB data set for sentiment analysis."""
#     data = []
#     for label in ['pos', 'neg']:
#         folder_name = os.path.join('../data/aclImdb/', folder, label)
#         for file in os.listdir(folder_name):
#             with open(os.path.join(folder_name, file), 'rb') as f:
#                 review = f.read().decode('utf-8').replace('\n', '').lower()
#                 data.append([review, 1 if label == 'pos' else 0])
#     random.shuffle(data)
#     return data


# def read_voc_images(root='../data/VOCdevkit/VOC2012', is_train=True):
#     """Read VOC images."""
#     txt_fname = '%s/ImageSets/Segmentation/%s' % (
#         root, 'train.txt' if is_train else 'val.txt')
#     with open(txt_fname, 'r') as f:
#         images = f.read().split()
#     features, labels = [None] * len(images), [None] * len(images)
#     for i, fname in enumerate(images):
#         features[i] = image.imread('%s/JPEGImages/%s.jpg' % (root, fname))
#         labels[i] = image.imread(
#             '%s/SegmentationClass/%s.png' % (root, fname))
#     return features, labels


class Residual(nn.Module):
    """The residual block."""
    def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1, **kwargs):
        super(Residual, self).__init__(**kwargs)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, 
                               stride=stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                  stride=stride)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
    def forward(self, X):
        Y = torch.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return torch.relu(Y + X)


# def resnet18(num_classes):
#     """The ResNet-18 model."""
#     net = nn.Sequential()
#     net.add(nn.Conv2D(64, kernel_size=3, strides=1, padding=1),
#             nn.BatchNorm(), nn.Activation('relu'))

#     def resnet_block(num_channels, num_residuals, first_block=False):
#         blk = nn.Sequential()
#         for i in range(num_residuals):
#             if i == 0 and not first_block:
#                 blk.add(Residual(num_channels, use_1x1conv=True, strides=2))
#             else:
#                 blk.add(Residual(num_channels))
#         return blk

#     net.add(resnet_block(64, 2, first_block=True),
#             resnet_block(128, 2),
#             resnet_block(256, 2),
#             resnet_block(512, 2))
#     net.add(nn.GlobalAvgPool2D(), nn.Dense(num_classes))
#     return net


class RNNModel(nn.Module):
    """RNN model."""
    def __init__(self, rnn_layer, num_hiddens, vocab_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.linear = nn.Linear(num_hiddens, vocab_size)
    
    def forward(self, inputs, state=None):
        # 将输入转置成(num_steps, batch_size)后获取one-hot向量表示
        X = torch.stack(to_onehot(inputs, self.vocab_size, inputs.device))
        Y, state = self.rnn(X, state)
        # 全连接层会首先将Y的形状变成(num_steps * batch_size, num_hiddens)，它的输出
        # 形状为(num_steps * batch_size, vocab_size)
        output = self.linear(Y.reshape(-1, Y.shape[-1]))
        return output, state


def semilogy(x_vals, y_vals, x_label, y_label, x2_vals=None, y2_vals=None,
             legend=None, figsize=(3.5, 2.5)):
    """Plot x and log(y)."""
    set_figsize(figsize)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.semilogy(x_vals, y_vals)
    if x2_vals and y2_vals:
        plt.semilogy(x2_vals, y2_vals, linestyle=':')
        plt.legend(legend)
    plt.show()


def set_figsize(figsize=(3.5, 2.5)):
    """Set matplotlib figure size."""
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize


def sgd(params, lr, batch_size):
    """Mini-batch stochastic gradient descent."""
    for param in params:
        param.data.sub_(lr * param.grad.data / batch_size)
        param.grad.data.zero_()


# def show_bboxes(axes, bboxes, labels=None, colors=None):
#     """Show bounding boxes."""
#     labels = _make_list(labels)
#     colors = _make_list(colors, ['b', 'g', 'r', 'm', 'k'])
#     for i, bbox in enumerate(bboxes):
#         color = colors[i % len(colors)]
#         rect = bbox_to_rect(bbox.asnumpy(), color)
#         axes.add_patch(rect)
#         if labels and len(labels) > i:
#             text_color = 'k' if color == 'w' else 'w'
#             axes.text(rect.xy[0], rect.xy[1], labels[i],
#                       va='center', ha='center', fontsize=9, color=text_color,
#                       bbox=dict(facecolor=color, lw=0))


def show_fashion_mnist(images, labels):
    """Plot Fashion-MNIST images with labels."""
    use_svg_display()
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.reshape(28, 28).numpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)


# def show_images(imgs, num_rows, num_cols, scale=2):
#     """Plot a list of images."""
#     figsize = (num_cols * scale, num_rows * scale)
#     _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
#     for i in range(num_rows):
#         for j in range(num_cols):
#             axes[i][j].imshow(imgs[i * num_cols + j].asnumpy())
#             axes[i][j].axes.get_xaxis().set_visible(False)
#             axes[i][j].axes.get_yaxis().set_visible(False)
#     return axes


# def show_trace_2d(f, res):
#     """Show the trace of 2d variables during optimization."""
#     x1, x2 = zip(*res)
#     set_figsize()
#     plt.plot(x1, x2, '-o', color='#ff7f0e')
#     x1 = np.arange(-5.5, 1.0, 0.1)
#     x2 = np.arange(min(-3.0, min(x2) - 1), max(1.0, max(x2) + 1), 0.1)
#     x1, x2 = np.meshgrid(x1, x2)
#     plt.contour(x1, x2, f(x1, x2), colors='#1f77b4')
#     plt.xlabel('x1')
#     plt.ylabel('x2')


def squared_loss(y_hat, y):
    """Squared loss."""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


def to_onehot(X, size, device='cpu'):
    """Represent inputs with one-hot encoding."""
    return [one_hot(x, size, device) for x in X.long().t()]


# def train(train_iter, test_iter, net, loss, trainer, ctx, num_epochs):
#     """Train and evaluate a model."""
#     print('training on', ctx)
#     if isinstance(ctx, mx.Context):
#         ctx = [ctx]
#     for epoch in range(num_epochs):
#         train_l_sum, train_acc_sum, n, m, start = 0.0, 0.0, 0, 0, time.time()
#         for i, batch in enumerate(train_iter):
#             Xs, ys, batch_size = _get_batch(batch, ctx)
#             ls = []
#             with autograd.record():
#                 y_hats = [net(X) for X in Xs]
#                 ls = [loss(y_hat, y) for y_hat, y in zip(y_hats, ys)]
#             for l in ls:
#                 l.backward()
#             trainer.step(batch_size)
#             train_l_sum += sum([l.sum().asscalar() for l in ls])
#             n += sum([l.size for l in ls])
#             train_acc_sum += sum([(y_hat.argmax(axis=1) == y).sum().asscalar()
#                                  for y_hat, y in zip(y_hats, ys)])
#             m += sum([y.size for y in ys])
#         test_acc = evaluate_accuracy(test_iter, net, ctx)
#         print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, '
#               'time %.1f sec'
#               % (epoch + 1, train_l_sum / n, train_acc_sum / m, test_acc,
#                  time.time() - start))


# def train_2d(trainer):
#     """Optimize the objective function of 2d variables with a customized trainer."""
#     x1, x2 = -5, -2
#     s_x1, s_x2 = 0, 0
#     res = [(x1, x2)]
#     for i in range(20):
#         x1, x2, s_x1, s_x2 = trainer(x1, x2, s_x1, s_x2)
#         res.append((x1, x2))
#     print('epoch %d, x1 %f, x2 %f' % (i+1, x1, x2))
#     return res


def train_and_predict_rnn(rnn, get_params, init_rnn_state, num_hiddens,
                          vocab_size, device, corpus_indices, idx_to_char,
                          char_to_idx, is_random_iter, num_epochs, num_steps,
                          lr, clipping_theta, batch_size, pred_period,
                          pred_len, prefixes):
    """Train an RNN model and predict the next item in the sequence."""
    if is_random_iter:
        data_iter_fn = data_iter_random
    else:
        data_iter_fn = data_iter_consecutive
    params = get_params()
    loss = nn.CrossEntropyLoss()
    
    for epoch in range(num_epochs):
        if not is_random_iter:
            state = init_rnn_state(batch_size, num_hiddens, device)
        l_sum, n, start = 0.0, 0, time.time()
        data_iter = data_iter_fn(corpus_indices, batch_size, num_steps, device)
        for X, Y in data_iter:
            if is_random_iter:
                state = init_rnn_state(batch_size, num_hiddens, device)
            else:
                for s in state:
                    s.detach_()
 
            inputs = to_onehot(X, vocab_size, device)
            (outputs, state) = rnn(inputs, state, params)
            outputs = torch.cat(outputs, dim=0)
            y = Y.t().flatten().long()
            
            l = loss(outputs, y).mean()
            l.backward()
            grad_clipping(params, clipping_theta, device)
            sgd(params, lr, 1)
            l_sum += l.data.item() * torch.numel(y.data)
            n += torch.numel(y.data)
            
        if (epoch + 1) % pred_period == 0:
            print('epoch %d, perplexity %f, time %.2f sec' % (
                epoch + 1, math.exp(l_sum / n), time.time() - start))
            for prefix in prefixes:
                print(' -', predict_rnn(
                    prefix, pred_len, rnn, params, init_rnn_state,
                    num_hiddens, vocab_size, device, idx_to_char, char_to_idx))


def train_and_predict_rnn_nn(model, num_hiddens, vocab_size, device,
                            corpus_indices, idx_to_char, char_to_idx,
                            num_epochs, num_steps, lr, clipping_theta,
                            batch_size, pred_period, pred_len, prefixes):
    """Train an PyTorch RNN model and predict the next item in the sequence."""
    loss = nn.CrossEntropyLoss()
    model.to(device)
    params_init(model, init=nn.init.normal_, mean=0.01)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0, weight_decay=0)
    
    for epoch in range(num_epochs):
        l_sum, n, start = 0.0, 0, time.time()
        data_iter = data_iter_consecutive(
            corpus_indices, batch_size, num_steps, device)
        state = None
        for X, Y in data_iter:
            optimizer.zero_grad()
            if not state is None:
                if isinstance(state, tuple):
                    for s in state:
                        s.detach_()
                else:
                    state.detach_()
            (output, state) = model(X, state)
            y = Y.long().t().flatten()
            l = loss(output, y)
            l.backward()
            # 梯度裁剪
            nn.utils.clip_grad_norm_(model.parameters(), clipping_theta)
            optimizer.step()
            l_sum += l.data.mean().item() * torch.numel(y.data)
            n += torch.numel(y.data)
        
        if (epoch + 1) % pred_period == 0:
            print('epoch %d, perplexity %f, time %.2f sec' % (
                epoch + 1, math.exp(l_sum / n), time.time() - start))
            for prefix in prefixes:
                print(' -', predict_rnn_nn(
                    prefix, pred_len, model, vocab_size, device, idx_to_char,
                    char_to_idx))


def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size,
              params=None, lr=None, optimizer=None):
    """Train and evaluate a model with CPU."""
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y).sum()
            l.backward()
            if optimizer is None:
                if isinstance(loss, nn.Module):
                    sgd(params, lr, 1)
                else:
                    sgd(params, lr, batch_size)
            else:
                optimizer.step()
                optimizer.zero_grad()
                
            train_l_sum += l.data.item()
            train_acc_sum += (y_hat.data.argmax(dim=1) == y).sum().item()
            n += y.size(0)
        
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))


def train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs):
    """Train and evaluate a model with CPU or GPU."""
    print('training on', device)
    net.to(device)
    loss = nn.CrossEntropyLoss()
    
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for X, y in train_iter:
            net.zero_grad()
            
            X, y = X.to(device), y.to(device)
                
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            
            train_l_sum += l.data.item()
            train_acc_sum += (torch.argmax(y_hat.data, dim=1) == y.data).sum().item()
            n += y.size(0)
    
        test_acc = evaluate_accuracy(test_iter, net, device)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc, time.time() - start))


# def train_ch7(trainer_fn, states, hyperparams, features, labels, batch_size=10,
#               num_epochs=2):
#     """Train a linear regression model."""
#     net, loss = linreg, squared_loss
#     w, b = nd.random.normal(scale=0.01, shape=(features.shape[1], 1)), nd.zeros(1)
#     w.attach_grad()
#     b.attach_grad()

#     def eval_loss():
#         return loss(net(features, w, b), labels).mean().asscalar()

#     ls = [eval_loss()]
#     data_iter = gdata.DataLoader(
#         gdata.ArrayDataset(features, labels), batch_size, shuffle=True)
#     for _ in range(num_epochs):
#         start = time.time()
#         for batch_i, (X, y) in enumerate(data_iter):
#             with autograd.record():
#                 l = loss(net(X, w, b), y).mean()
#             l.backward()
#             trainer_fn([w, b], states, hyperparams)
#             if (batch_i + 1) * batch_size % 100 == 0:
#                 ls.append(eval_loss())
#     print('loss: %f, %f sec per epoch' % (ls[-1], time.time() - start))
#     set_figsize()
#     plt.plot(np.linspace(0, num_epochs, len(ls)), ls)
#     plt.xlabel('epoch')
#     plt.ylabel('loss')


# def train_gluon_ch7(trainer_name, trainer_hyperparams, features, labels,
#                     batch_size=10, num_epochs=2):
#     """Train a linear regression model with a given Gluon trainer."""
#     net = nn.Sequential()
#     net.add(nn.Dense(1))
#     net.initialize(init.Normal(sigma=0.01))
#     loss = gloss.L2Loss()

#     def eval_loss():
#         return loss(net(features), labels).mean().asscalar()

#     ls = [eval_loss()]
#     data_iter = gdata.DataLoader(
#         gdata.ArrayDataset(features, labels), batch_size, shuffle=True)
#     trainer = gluon.Trainer(net.collect_params(),
#                             trainer_name, trainer_hyperparams)
#     for _ in range(num_epochs):
#         start = time.time()
#         for batch_i, (X, y) in enumerate(data_iter):
#             with autograd.record():
#                 l = loss(net(X), y)
#             l.backward()
#             trainer.step(batch_size)
#             if (batch_i + 1) * batch_size % 100 == 0:
#                 ls.append(eval_loss())
#     print('loss: %f, %f sec per epoch' % (ls[-1], time.time() - start))
#     set_figsize()
#     plt.plot(np.linspace(0, num_epochs, len(ls)), ls)
#     plt.xlabel('epoch')
#     plt.ylabel('loss')


# def try_all_gpus():
#     """Return all available GPUs, or [mx.cpu()] if there is no GPU."""
#     ctxes = []
#     try:
#         for i in range(16):
#             ctx = mx.gpu(i)
#             _ = nd.array([0], ctx=ctx)
#             ctxes.append(ctx)
#     except mx.base.MXNetError:
#         pass
#     if not ctxes:
#         ctxes = [mx.cpu()]
#     return ctxes


def try_gpu():
    """If GPU is available, return mx.gpu(0); else return mx.cpu()."""
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def use_svg_display():
    """Use svg format to display plot in jupyter"""
    display.set_matplotlib_formats('svg')


# def voc_label_indices(colormap, colormap2label):
#     """Assign label indices for Pascal VOC2012 Dataset."""
#     colormap = colormap.astype('int32')
#     idx = ((colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256
#            + colormap[:, :, 2])
#     return colormap2label[idx]


# def voc_rand_crop(feature, label, height, width):
#     """Random cropping for images of the Pascal VOC2012 Dataset."""
#     feature, rect = image.random_crop(feature, (width, height))
#     label = image.fixed_crop(label, *rect)
#     return feature, label


# class VOCSegDataset(gdata.Dataset):
#     """The Pascal VOC2012 Dataset."""
#     def __init__(self, is_train, crop_size, voc_dir, colormap2label):
#         self.rgb_mean = nd.array([0.485, 0.456, 0.406])
#         self.rgb_std = nd.array([0.229, 0.224, 0.225])
#         self.crop_size = crop_size
#         data, labels = read_voc_images(root=voc_dir, is_train=is_train)
#         self.data = [self.normalize_image(im) for im in self.filter(data)]
#         self.labels = self.filter(labels)
#         self.colormap2label = colormap2label
#         print('read ' + str(len(self.data)) + ' examples')

#     def normalize_image(self, data):
#         return (data.astype('float32') / 255 - self.rgb_mean) / self.rgb_std

#     def filter(self, images):
#         return [im for im in images if (
#             im.shape[0] >= self.crop_size[0] and
#             im.shape[1] >= self.crop_size[1])]

#     def __getitem__(self, idx):
#         data, labels = voc_rand_crop(self.data[idx], self.labels[idx],
#                                      *self.crop_size)
#         return (data.transpose((2, 0, 1)),
#                 voc_label_indices(labels, self.colormap2label))

#     def __len__(self):
#         return len(self.data)
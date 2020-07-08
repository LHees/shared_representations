# Author: Laurens Hees

import numpy as np
import matplotlib.pyplot as plt
import math
import copy
from skimage.transform import resize
from skimage.metrics import structural_similarity as ssim
from dbm import RBM, DBM, BimodalDBM


def read(file):
    """ Read a .txt pattern file.

    Return:
        data: a 2D ndarray of binary data. Each row is a pattern.
    """
    with open(file) as f:
        contents = f.readlines()
    data = np.asarray([np.asarray([int(c) for c in line if c != '\n'])
                       for line in contents])
    return data


def plot_mnist(image):
    """ Plot a mnist image represented as a vector.

    Args:
        image: a vector with a mnist image representation.
    """

    dim = int(math.sqrt(len(image)))
    plot = np.reshape(image, (dim, dim))
    plt.imshow(plot, cmap='gray')
    plt.show()


def plot_rgb(r, g, b):
    dim = int(math.sqrt(len(r)))
    fig, axes = plt.subplots(1, 3, figsize=(12, 12))
    fig.suptitle("Red, green and blue components:")
    axes = axes.flatten()

    axes[0].imshow(r.reshape((dim, dim)), cmap="gray")
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    axes[1].imshow(g.reshape((dim, dim)), cmap="gray")
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    axes[2].imshow(b.reshape((dim, dim)), cmap="gray")
    axes[2].set_xticks([])
    axes[2].set_yticks([])

    plt.show()


def plot_ori_recon(ori, recon):
    """ Plot a mnist image represented as a vector and its reconstruction.

    Args:
        ori: a vector with a mnist image representation.
        recon: a vector with a reconstructed representation of the same image.
    """

    dim = int(math.sqrt(len(ori)))
    fig, axes = plt.subplots(1, 2, figsize=(16, 16))
    fig.suptitle("Original image and reconstruction:")
    axes = axes.flatten()
    axes[0].imshow(ori.reshape((dim, dim)), cmap="gray")
    axes[1].imshow(recon.reshape((dim, dim)), cmap="gray")
    plt.show()


def plot_three(one, two, three, title):
    """ Plot a MNIST image, a mixture of this image with another and a
    reconstruction of the original image.

    Args:
        one: a vector with e.g. a mnist image representation.
        two: e.g. a mix of the original image and another.
        three: e.g. a vector with a reconstructed representation of the
        original image.
    """

    dim = int(math.sqrt(len(one)))
    fig, axes = plt.subplots(1, 3, figsize=(16, 16))
    fig.suptitle(title)
    axes = axes.flatten()
    axes[0].imshow(one.reshape((dim, dim)), cmap="gray")
    axes[1].imshow(two.reshape((dim, dim)), cmap="gray")
    axes[2].imshow(three.reshape((dim, dim)), cmap="gray")
    plt.show()


def plot_hidden(W):
    """ Plot the first few feature detectors of a hidden layer.

    Args:
        W: the weights from an input layer to the layer to be plotted.
    """
    dim = int(math.sqrt(W.shape[0]))
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    fig.suptitle("Feature detectors in the hidden layer:")
    axes = axes.flatten()
    for i in range(16):
        try:
            axes[i].imshow(W[:, i].reshape((dim, dim)), cmap="gray")
            axes[i].set_xticks([])
            axes[i].set_yticks([])
        except IndexError:
            break
    plt.show()


def plot_hidden_2(W1, W2):
    dim = int(math.sqrt(W1.shape[0]))
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    fig.suptitle("Feature detectors in the second hidden layer:")
    axes = axes.flatten()
    for i2 in range(16):
        try:
            plot = np.zeros(dim**2)
            for i1, w2 in enumerate(W2[:, i2]):  # index of w1 and its weight to w2
                plot += W1[i1] * w2
            axes[i2].imshow(plot.reshape((dim, dim)), cmap="gray")
            axes[i2].set_xticks([])
            axes[i2].set_yticks([])
        except IndexError:
            break
    plt.show()


def small(X):
    X_small = np.zeros((len(X), 14*14))
    for i, im in enumerate(X):
        im = np.reshape(im, (28, 28))
        im = resize(im, (14, 14))
        im = np.reshape(im, (14 * 14))
        X_small[i] = im
    return X_small


def preprocess(data, mean=None, sd=None):
    if mean is None:
        mean = np.mean(data, 0)
    if sd is None:
        sd = np.std(data, 0)
    data -= mean
    data = np.divide(data, sd, out=copy.deepcopy(data), where=sd != 0)
    return data, mean, sd


def reprocess(data, mean, sd):
    data = np.multiply(data, sd, out=copy.deepcopy(data), where=sd != 0)
    data += mean
    return data


def save_params(network, path):
    if isinstance(network, DBM):
        bot_params, top_params = network.parameters
        bot_W, bot_b_vis, bot_b_hid = bot_params
        top_W, top_b_vis, top_b_hid = top_params
        np.savez(path, bot_W=bot_W, bot_b_vis=bot_b_vis, bot_b_hid=bot_b_hid,
                 top_W=top_W, top_b_vis=top_b_vis, top_b_hid=top_b_hid)

    elif isinstance(network, BimodalDBM):
        l_params, r_params, W_l, W_r, b_joint = network.parameters
        l_bot_params, l_top_params = l_params
        r_bot_params, r_top_params = r_params
        l_bot_W, l_bot_b_vis, l_bot_b_hid = l_bot_params
        l_top_W, l_top_b_vis, l_top_b_hid = l_top_params
        r_bot_W, r_bot_b_vis, r_bot_b_hid = r_bot_params
        r_top_W, r_top_b_vis, r_top_b_hid = r_top_params
        np.savez(path, l_bot_W=l_bot_W, l_bot_b_vis=l_bot_b_vis,
                 l_bot_b_hid=l_bot_b_hid, l_top_W=l_top_W,
                 l_top_b_vis=l_top_b_vis, l_top_b_hid=l_top_b_hid,
                 r_bot_W=r_bot_W, r_bot_b_vis=r_bot_b_vis,
                 r_bot_b_hid=r_bot_b_hid, r_top_W=r_top_W,
                 r_top_b_vis=r_top_b_vis, r_top_b_hid=r_top_b_hid, W_l=W_l,
                 W_r=W_r, b_joint=b_joint)

    else:
        raise ValueError("Type of network is not understood.")


def mse(ori: np.ndarray, recon: np.ndarray):
    return np.mean(np.square(ori - recon))


def ssi(data, recon):
    ssis = np.asarray([ssim(ori, rec) for ori, rec in zip(data, recon)])
    return np.mean(ssis), np.std(ssis)

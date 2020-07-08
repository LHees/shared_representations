# Author: Laurens Hees

import numpy as np
import copy
from mlxtend.data import loadlocal_mnist
from dbm import DBM, RBMBin, RBMGaus, RBMSoftmax, BimodalDBM
import util


def run():
    train_path = 'data/mnist/train-images.idx3-ubyte'
    train_label_path = 'data/mnist/train-labels.idx1-ubyte'
    test_path = 'data/mnist/t10k-images.idx3-ubyte'
    test_label_path = 'data/mnist/t10k-labels.idx1-ubyte'
    n_cls = 10  # 1 - 10
    data_train, labels_train = loadlocal_mnist(train_path, train_label_path)
    data_train = data_train[np.isin(labels_train, range(n_cls))]
    labels_train = labels_train[np.isin(labels_train, range(n_cls))]
    data_train = util.small(data_train)
    _, n_feat = data_train.shape
    data_test, labels_test = loadlocal_mnist(test_path, test_label_path)
    data_test = data_test[np.isin(labels_test, range(n_cls))]
    labels_test = labels_test[np.isin(labels_test, range(n_cls))]
    data_test = util.small(data_test)

    # test_dbm(data_train, data_test, n_feat, labels_test=labels_test, vis=True)
    test_mdbm_lh(data_train, labels_train, data_test, labels_test, n_feat)
    # test_mdbm_lrgb(data_train, labels_train, data_test, labels_test, n_feat)
    # test_mdbm_lc(data_train, labels_train, data_test, labels_test, n_feat)
    # test_mdbm_le(data_train, data_test, labels_test, n_feat)


def test_dbm(data_train, data_test, n_feat, labels_test=None, vis=False):
    """ Perform the normal reconstruction, mixture and missing modality tasks
    with a (unimodal) DBM.

    IMPORTANT: use only one reconstruction task per run. Otherwise the data is
    reprocessed more than once, leading to false results. TODO: fix this

    Args:
        data_train: training data for the dbm.
        data_test: test data for the dbm.
        n_feat: the number of features in the data.
        labels_test: the labels that go with the test data (optional).
        vis: whether some examples should be plotted (default False).
    """
    data_train, mean, sd = util.preprocess(data_train)

    # Modelling
    dbm = DBM(RBMGaus(n_feat, n_feat), RBMBin(n_feat, n_feat))
    dbm.parameters = np.load('models/final/dbm_100_100.npz')
    # dbm.pretrain(data_train, nE=100)
    # dbm.train(data_train, nE=100)
    # util.save_params(dbm, 'models/final/dbm_100_100.npz')


    # Testing
    data_test, _, _ = util.preprocess(data_test, mean, sd)

    # weights
    # TODO

    # # normal reconstruction
    # recon_test = dbm.sample_vis(data_test)
    # data_test = util.reprocess(data_test, mean, sd)
    # recon_test = util.reprocess(recon_test, mean, sd)
    # mse = util.mse(data_test, recon_test)
    # print('left mse: ', mse)
    # ssi, sd = util.ssi(data_test, recon_test)
    # print('ssi: ', ssi, ', sd: ', sd)
    #
    # if vis:
    #     for _ in range(3):
    #         i = np.random.randint(0, len(data_test))
    #         print('index: ', i, '\tclass: ',
    #               labels_test[i] if labels_test is not None else 'unknown')
    #         util.plot_ori_recon(data_test[i], recon_test[i])

    # # mixture
    # mixtures = create_mixtures(data_test)
    # recon_test = dbm.sample_vis(mixtures)
    # mixtures = util.reprocess(mixtures, mean, sd)
    # recon_test = util.reprocess(recon_test, mean, sd)
    # data_test = util.reprocess(data_test, mean, sd)
    # mse = util.mse(data_test, recon_test)
    # print('mse: ', mse)
    # ssi, sd = util.ssi(data_test, recon_test)
    # print('ssi: ', ssi, ', sd: ', sd)
    #
    # if vis:
    #     for _ in range(3):
    #         i = np.random.randint(0, len(mixtures))
    #         print('index: ', i, '\tclass: ',
    #               labels_test[i] if labels_test is not None else 'unknown')
    #         util.plot_three(data_test[i], mixtures[i], recon_test[i],
    #                         "Original image, mixture and reconstruction:")

    # missing modality
    missing = np.zeros((len(data_test), n_feat))
    missing, _, _ = util.preprocess(missing, mean, sd)
    recon_test = dbm.sample_vis(missing)
    missing = util.reprocess(missing, mean, sd)
    recon_test = util.reprocess(recon_test, mean, sd)
    data_test = util.reprocess(data_test, mean, sd)
    mse = util.mse(data_test, recon_test)
    print('mse: ', mse)
    ssi, sd = util.ssi(data_test, recon_test)
    print('ssi: ', ssi, ', sd: ', sd)

    if vis:
        for temp in range(3):
            i = np.random.randint(0, len(missing))
            print('index: ', i, '\tclass: ',
                  labels_test[i] if labels_test is not None else 'unknown')
            util.plot_three(data_test[i], missing[i], recon_test[i],
                            "Original image, blank input and reconstruction:")


def test_mdbm_lh(data_train, labels_train, data_test, labels_test, n_feat):
    h_train = repr_h(data_train, labels_train)
    data_train, mean_l, sd_l = util.preprocess(data_train)
    h_train, mean_h, sd_h = util.preprocess(h_train)

    # Modelling
    left = DBM(RBMGaus(n_feat, n_feat), RBMBin(n_feat, n_feat))
    right = DBM(RBMGaus(n_feat, n_feat), RBMBin(n_feat, n_feat))
    mdbm = BimodalDBM(left, right, n_feat * 2)
    # mdbm.parameters = np.load('models/final/mnisth_100_100.npz')
    mdbm.pretrain(data_train, h_train, nE=1000)
    mdbm.train(data_train, h_train, nE=100)
    util.save_params(mdbm, 'models/mnisth_100_100_10classes_morepretraining.npz')


    # Testing
    h_test = repr_h(data_test, labels_test)
    data_test, _, _ = util.preprocess(data_test, mean_l, sd_l)
    h_test, _, _ = util.preprocess(h_test, mean_h, sd_h)

    # weights
    # util.plot_hidden(mdbm.left.bot.W)
    # util.plot_hidden(mdbm.right.bot.W)
    # util.plot_hidden(mdbm.left.top.W)
    # util.plot_hidden(mdbm.right.top.W)
    # util.plot_hidden(mdbm.W_left)
    # util.plot_hidden(mdbm.W_right)

    # normal reconstruction
    # norm_rec(mdbm, data_test, mean_l, sd_l, h_test, mean_h, sd_h,
    #          plot_col=True, labels=labels_test)

    # mixture
    mixture(mdbm, data_test, mean_l, sd_l, h_test, labels=labels_test)

    # missing modality
    # miss_mod(mdbm, data_test, mean_l, sd_l, h_test, n_feat, labels_test)


def test_mdbm_lrgb(data_train, labels_train, data_test, labels_test, n_feat):
    rgb_train = repr_rgb(labels_train)
    data_train, mean_l, sd_l = util.preprocess(data_train)
    rgb_train, mean_rgb, sd_rgb = util.preprocess(rgb_train)

    # Modelling
    left = DBM(RBMGaus(n_feat, n_feat), RBMBin(n_feat, n_feat))
    right = DBM(RBMGaus(3, 3), RBMBin(3, 3))
    mdbm = BimodalDBM(left, right, n_feat * 2)
    mdbm.parameters = np.load('models/final/mnistrgb_100_100.npz')
    # mdbm.pretrain(data_train, rgb_train, nE=100)
    # mdbm.train(data_train, rgb_train, nE=100)
    # util.save_params(mdbm, 'models/final/mnistrgb_100_100.npz')


    # Testing
    rgb_test = repr_rgb(labels_test)
    data_test, _, _ = util.preprocess(data_test, mean_l, sd_l)
    rgb_test, _, _ = util.preprocess(rgb_test, mean_rgb, sd_rgb)

    # # weights
    # util.plot_hidden(mdbm.left.bot.W)
    # util.plot_hidden(mdbm.left.top.W)
    # util.plot_hidden_2(mdbm.left.bot.W, mdbm.left.top.W)

    # normal reconstruction
    norm_rec(mdbm, data_test, mean_l, sd_l, rgb_test, mean_rgb, sd_rgb,
             labels=labels_test)

    # mixture
    mixture(mdbm, data_test, mean_l, sd_l, rgb_test, labels=labels_test)

    # missing modality
    miss_mod(mdbm, data_test, mean_l, sd_l, rgb_test, n_feat, labels_test)


def test_mdbm_lc(data_train, labels_train, data_test, labels_test, n_feat):
    cat_train = repr_cat(labels_train)
    data_train, mean_l, sd_l = util.preprocess(data_train)

    # Modelling
    left = DBM(RBMGaus(n_feat, n_feat), RBMBin(n_feat, n_feat))
    right = DBM(RBMSoftmax(10, 10), RBMBin(10, 10))
    mdbm = BimodalDBM(left, right, n_feat * 2)
    mdbm.parameters = np.load('models/final/mnistc_100_100.npz')
    # mdbm.pretrain(data_train, cat_train, nE=100)
    # mdbm.train(data_train, cat_train, nE=100)
    # util.save_params(mdbm, 'models/final/mnistc_100_100.npz')


    # Testing
    cat_test = repr_cat(labels_test)
    data_test, _, _ = util.preprocess(data_test, mean_l, sd_l)

    # # weights
    # util.plot_hidden(mdbm.left.bot.W)
    # util.plot_hidden(mdbm.left.top.W)
    # util.plot_hidden_2(mdbm.left.bot.W, mdbm.left.top.W)

    # normal reconstruction
    norm_rec(mdbm, data_test, mean_l, sd_l, cat_test, None, None,
             labels=labels_test)

    # mixture
    mixture(mdbm, data_test, mean_l, sd_l, cat_test, labels=labels_test)

    # missing modality
    miss_mod(mdbm, data_test, mean_l, sd_l, cat_test, n_feat, labels_test)


def test_mdbm_le(data_train, data_test, labels_test, n_feat):
    empty_train = np.zeros((len(data_train), n_feat))
    data_train, mean, sd = util.preprocess(data_train)

    # Modelling
    left = DBM(RBMGaus(n_feat, n_feat), RBMBin(n_feat, n_feat))
    right = DBM(RBMGaus(n_feat, n_feat), RBMBin(n_feat, n_feat))
    mdbm = BimodalDBM(left, right, n_feat * 2)
    mdbm.parameters = np.load('models/final/mniste_100_100.npz')
    # mdbm.pretrain(data_train, empty_train, nE=100)
    # mdbm.train(data_train, empty_train, nE=100)
    # util.save_params(mdbm, 'models/final/mniste_100_100.npz')

    # Testing
    empty_test = np.zeros((len(data_test), n_feat))
    data_test, _, _ = util.preprocess(data_test, mean, sd)

    # # weights
    # util.plot_hidden(mdbm.left.bot.W)
    # util.plot_hidden(mdbm.left.top.W)
    # util.plot_hidden_2(mdbm.left.bot.W, mdbm.left.top.W)

    # normal reconstruction
    norm_rec(mdbm, data_test, mean, sd, empty_test, None, None,
             labels=labels_test)

    # mixture
    mixture(mdbm, data_test, mean, sd, empty_test, labels=labels_test)

    # missing modality
    miss_mod(mdbm, data_test, mean, sd, empty_test, n_feat, labels_test)


def repr_h(data, labels):
    hues = np.arange(360, step=36)
    hues[0] = 360
    h = copy.deepcopy(data)
    for i, img in enumerate(h):
        img[img > 0] = hues[labels[i]]
    return h


def norm_rec(mdbm: BimodalDBM, data, mean_l, sd_l, col, mean_c=None, sd_c=None,
             plot_col: bool=False, labels=None, vis=False):
    """ Perform the normal reconstruction task and assess the results.

    Args:
        mdbm: a trained bimodal DBM.
        data: preprocessed original data for the missing modality.
        mean_l: mean of the left modality data.
        sd_l: standard deviation of the left modality data.
        col: preprocessed original data for the color modality.
        mean_c: mean of the color modality data (optional).
        sd_c: standard deviation of the color modality data (optional).
        plot_col: whether the color modality reconstruction should be plotted.
        Alternatively is is printed (default False).
        labels: the labels that go with the missing modality data (optional).
        vis: whether some examples should be plotted (default False).
    """
    recon_test, colrecon_test = mdbm.sample_vis(data, col)
    data_test = util.reprocess(data, mean_l, sd_l)
    col = util.reprocess(col, mean_c, sd_c) if mean_c is not None else col
    recon_test = util.reprocess(recon_test, mean_l, sd_l)
    colrecon_test = util.reprocess(colrecon_test, mean_c, sd_c) if \
        mean_c is not None else colrecon_test
    mse_l = util.mse(data_test, recon_test)
    print('left mse: ', mse_l)
    ssi, sd = util.ssi(data_test, recon_test)
    print('ssi: ', ssi, ', sd: ', sd)
    mse_r = util.mse(col, colrecon_test)
    print('right mse: ', mse_r)

    if vis:
        for _ in range(3):
            i = np.random.randint(0, len(data_test))
            print('index: ', i, '\tclass: ',
                  labels[i] if labels is not None else 'unknown')
            util.plot_ori_recon(data_test[i], recon_test[i])
            if plot_col:
                util.plot_ori_recon(col[i], colrecon_test[i])
            else:
                print(col[i])
                print(colrecon_test[i])


def mixture(mdbm: BimodalDBM, data, mean, sd, col, labels=None, vis=False):
    """ Perform the mixture task and assess the results.

    Args:
        mdbm: a trained bimodal DBM.
        data: preprocessed original data for the missing modality.
        mean: mean of the left modality data.
        sd: standard deviation of the left modality data.
        col: preprocessed original data for the color modality.
        labels: the labels that go with the missing modality data (optional).
        vis: whether some examples should be plotted (default False).
    """
    mixtures = create_mixtures(data)
    recon_test, _ = mdbm.sample_vis(mixtures, col)
    mixtures = util.reprocess(mixtures, mean, sd)
    recon_test = util.reprocess(recon_test, mean, sd)
    data_test = util.reprocess(data, mean, sd)
    mse = util.mse(data_test, recon_test)
    print('mse: ', mse)
    ssi, sd = util.ssi(data_test, recon_test)
    print('ssi: ', ssi, ', sd: ', sd)

    if vis:
        for _ in range(5):
            i = np.random.randint(0, len(mixtures))
            print('index: ', i, '\tclass: ',
                  labels[i] if labels is not None else 'unknown')
            util.plot_three(data_test[i], mixtures[i], recon_test[i],
                            "Original image, mixture and reconstruction:")


def miss_mod(mdbm: BimodalDBM, data, mean, sd, col, n_feat, labels=None,
             vis=False):
    """ Perform the missing modality task and assess the results.

    Args:
        mdbm: a trained bimodal DBM.
        data: preprocessed original data for the missing modality.
        mean: mean of the left modality data.
        sd: standard deviation of the left modality data.
        col: preprocessed original data for the color modality.
        n_feat: the number of features in the missing modality data.
        labels: the labels that go with the missing modality data (optional).
        vis: whether some examples should be plotted (default False).
    """
    missing = np.zeros((len(data), n_feat))
    missing, _, _ = util.preprocess(missing, mean, sd)
    recon_test, _ = mdbm.sample_vis(missing, col)
    missing = util.reprocess(missing, mean, sd)
    recon_test = util.reprocess(recon_test, mean, sd)
    data_test = util.reprocess(data, mean, sd)
    mse = util.mse(data_test, recon_test)
    print('mse: ', mse)
    ssi, sd = util.ssi(data_test, recon_test)
    print('ssi: ', ssi, ', sd: ', sd)

    if vis:
        for temp in range(5):
            i = np.random.randint(0, len(missing))
            print('index: ', i, '\tclass: ',
                  labels[i] if labels is not None else 'unknown')
            util.plot_three(data_test[i], missing[i], recon_test[i],
                            "Original image, blank input and reconstruction:")


def repr_rgb(labels):
    colors = [[255, 0, 0], [255, 153, 0], [204, 255, 0], [51, 255, 0],
              [0, 255, 102], [0, 255, 255], [0, 102, 255], [51, 0, 255],
              [204, 0, 255], [255, 0, 153]]
    rgb = np.asarray([colors[l] for l in labels], dtype=float)
    return rgb


def repr_cat(labels):
    categories = np.zeros((10, 10), dtype=int)
    np.fill_diagonal(categories, 1)
    cat = np.asarray([categories[l] for l in labels])
    return cat


def create_mixtures(data):
    mixtures = []
    for i, example in enumerate(data):
        j = np.random.randint(0, len(data))
        intruder = data[j]
        mixtures.append((example + intruder) / 2)
    return np.asarray(mixtures)


if __name__ == '__main__':
    run()

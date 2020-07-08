# Author: Laurens Hees
# Inspired by code of Michael Deistler and Yagmur Yener
# Data source: http://yann.lecun.com/exdb/mnist/

import numpy as np
import copy
from abc import ABC, abstractmethod


class RBM(ABC):
    def __init__(self, n_vis, n_hid, debug=False):
        self.debug = debug
        np_rng = np.random.RandomState(1234)
        self.n_vis = n_vis
        self.n_hid = n_hid
        self.W = np.asarray(np_rng.uniform(
            low=-0.2 * np.sqrt(6. / (n_hid + n_vis)),
            high=0.2 * np.sqrt(6. / (n_hid + n_vis)),
            size=(n_vis, n_hid)))
        self.b_vis = np.zeros(n_vis)
        self.b_hid = np.zeros(n_hid)

    @property
    def parameters(self):
        return self.W, self.b_vis, self.b_hid

    @parameters.setter
    def parameters(self, npz: dict):
        self.W = npz['W']
        self.b_vis = npz['b_vis']
        self.b_hid = npz['b_hid']

    def train(self, data, nE=1000, bS = 10, lR=0.01, k=1, m_bu=1, m_td=1):
        """ Train the RBM using the CDk algorithm. The RBM's weights and biases
        will be updated.

        Args:
            data: a 2D ndarray in which each row is a data vector.
            nE: the number of training epochs (default 1000).
            bS: the size of the minibatches used (default 10).
            lR: the learning rate (default 0.01).
            k: the number of sampling steps used by the algorithm (default 1).
            m_bu: multiplier for the weights in bottom-up passes (default 1).
            m_td: multiplier for the weights in top-down passes.
        """
        for e in range(nE):
            print(f'RBM epoch: ', e)
            rows = np.random.permutation(len(data))
            for b in range(1, int(len(data) / bS) + 2):
                batch = data[rows[(b - 1) * bS:b * bS]]
                act_bS = len(batch)  # data length may not be divisible by bS
                if act_bS == 0:
                    break
                vis = batch
                hid_p, hid = self.to_hid(vis, w_mult=m_bu)

                # Copy the hidden state and run k Gibbs sample steps.
                vis_m = None
                vis_m_p = None
                hid_m_p = None
                hid_m = copy.deepcopy(hid)
                for _ in range(k):
                    vis_m_p, vis_m = self.to_vis(hid_m, w_mult=m_td)
                    hid_m_p, hid_m = self.to_hid(vis_m, w_mult=m_bu)

                # Update weights and biases.
                self.update_W(vis, hid_p, vis_m_p, vis_m, hid_m_p,
                              bS=act_bS, lR=lR)
                self.update_b_vis(vis, vis_m_p, vis_m, bS=act_bS, lR=lR)
                self.update_b_hid(hid_p, hid_m_p, bS=act_bS, lR=lR)

    def hid_act(self, vis, b=True, w_mult=1):
        """ Compute the activation of the hidden nodes.

        Args:
            vis: the state of the visual nodes.
            b: whether the bias should be included (default True).
            w_mult: a multiplier for the weights (default 1). Can be set to
            different values for pretraining scenarios.

        Returns:
            act: the activation of the hidden nodes.
        """
        act = np.dot(vis, self.W * w_mult)
        if b:
            act += self.b_hid
        return act

    def vis_act(self, hid, b=True, w_mult=1):
        """ Compute the activation of the visible nodes.

        Args:
            hid: the state of the hidden nodes.
            b: whether the bias should be included (default True).
            w_mult: a multiplier for the weights (default 1). Can be set to
            different values for pretraining scenarios.

        Returns:
            act: the activation of the visible nodes.
        """
        act = np.dot(hid, self.W.T * w_mult)
        if b:
            act += self.b_vis
        return act

    @abstractmethod
    def to_vis(self, hid, w_mult=1):
        """ Compute the output of the visible nodes.
        Compute the activation, then call an activation function over it and
        sample if appropriate.

        Args:
            hid: the state of the hidden nodes.
            w_mult: a multiplier for the incoming weights (default 1).

        Returns:
            vis_p: the result of calling the activation function over the
            activation. None if these values are not to be interpreted as
            probabilities.
            vis: the output of the visible nodes.
        """
        pass

    def to_hid(self, vis, w_mult=1):
        """ Compute the output of the hidden nodes.
        Compute the activation, then call the sigmoid activation function
        over it and sample from the resulting distribution.

        Args:
            vis: the state of the visible nodes.
            w_mult: a multiplier for the incoming weights (default 1).

        Returns:
            hid_p: the result of calling the sigmoid activation function
            over the activation. These are probabilities.
            hid: the output of the hidden nodes.
        """
        hid_p = sigmoid(self.hid_act(vis, w_mult=w_mult))
        hid = (hid_p > np.random.rand(self.n_hid)).astype(int)
        return hid_p, hid

    @abstractmethod
    def update_W(self, vis, hid_p, vis_m_p, vis_m, hid_m_p, bS, lR):
        """ Update the weights of the RBM based on statistics obtained with
        the CDk algorithm.

        Args:
            vis: the values of the visible nodes with clamped input.
            hid_p: the probabilities of the hidden nodes with clamped input.
            vis_m_p: the probabilities of the visible nodes without clamped
            input.
            vis_m: the values of the visible nodes without clamped input.
            hid_m_p: the probabilities of the hidden nodes without clamped
            input.
            bS: the size of the minibatches used.
            lR: the learning rate used.
        """
        pass

    @abstractmethod
    def update_b_vis(self, vis, vis_m_p, vis_m, bS, lR):
        """ Update the visible node biases of the RBM based on statistics
        obtained with the CDk algorithm.

        Args:
            vis: the values of the visible nodes with clamped input.
            vis_m_p: the probabilities of the visible nodes without clamped
            input.
            vis_m: the values of the visible nodes without clamped input.
            bS: the size of the minibatches used.
            lR: the learning rate used.
        """
        pass

    def update_b_hid(self, hid_p, hid_m_p, bS, lR):
        """ Update the hidden node biases of the RBM based on statistics
        obtained with the CDk algorithm.

        Args:
            hid_p: the probabilities of the hidden nodes with clamped input.
            hid_m_p: the probabilities of the hidden nodes without clamped
            input.
            bS: the size of the minibatches used.
            lR: the learning rate used.
        """
        self.b_hid += lR/bS * np.sum(hid_p - hid_m_p, axis=0)

    def sample_vis(self, example, nC=1):
        """ Generate a data reconstruction.

        Args:
            example: a real data vector.
            nC: the number of Gibbs sampling cycles to take (default 1).

        Returns:
            vis: a reconstructed data vector.
        """
        vis = example
        for _ in range(nC):
            _, hid = self.to_hid(vis)
            _, vis = self.to_vis(hid)
        return vis

    def pass_through(self, data):
        """ Pass input data through the RBM and return the output.

        Args:
            data: input data vectors.

        Returns:
            hidden_states: the data's corresponding hidden states.
        """
        hidden_states = []
        for vector in data:
            _, hid = self.to_hid(vector)
            hidden_states.append(hid)
        return np.asarray(hidden_states)


class RBMBin(RBM):
    def to_vis(self, hid, w_mult=1):
        vis_p = sigmoid(self.vis_act(hid, w_mult=w_mult))
        vis = (vis_p > np.random.rand(self.n_vis)).astype(int)
        return vis_p, vis

    def update_W(self, vis, hid_p, vis_m_p, vis_m, hid_m_p, lR, bS):
        self.W += lR/bS * (np.dot(vis.T, hid_p) - np.dot(vis_m_p.T, hid_m_p))

    def update_b_vis(self, vis, vis_m_p, vis_m, lR, bS):
        self.b_vis += lR/bS * np.sum(vis - vis_m_p, axis=0)


class RBMGaus(RBM):
    def to_vis(self, hid, w_mult=1):
        vis = self.vis_act(hid, w_mult=w_mult)
        return None, vis

    def update_W(self, vis, hid_p, vis_m_p, vis_m, hid_m_p, lR, bS):
        self.W += lR/bS * (np.dot(vis.T, hid_p) - np.dot(vis_m.T, hid_m_p))

    def update_b_vis(self, vis, vis_m_p, vis_m, lR, bS):
        self.b_vis += lR/bS * np.sum(vis - vis_m, axis=0)


class RBMSoftmax(RBM):
    def to_vis(self, hid, w_mult=1):
        vis_a = self.vis_act(hid, w_mult=w_mult)
        vis_p = softmax(vis_a)
        if vis_p.ndim > 1:
            vis = np.ma.masked_where(vis_p != np.resize(vis_p.max(axis=1),
                                        [vis_p.shape[0], 1]), vis_p).filled(0)
            vis[vis > 0] = 1
        else:
            vis = vis_p == max(vis_p)
        vis = vis.astype(int)
        return vis_p, vis

    def update_W(self, vis, hid_p, vis_m_p, vis_m, hid_m_p, bS, lR):
        self.W += lR/bS * (np.dot(vis.T, hid_p) - np.dot(vis_m_p.T, hid_m_p))

    def update_b_vis(self, vis, vis_m_p, vis_m, bS, lR):
        self.b_vis += lR/bS * np.sum(vis - vis_m_p, axis=0)


class DBM:
    """ Two layer Deep Boltzmann Machine."""

    def __init__(self, rbm_bot: RBM, rbm_top: RBMBin):
        self.bot = rbm_bot
        self.top = rbm_top
        self.top.b_vis = self.bot.b_hid

    @property
    def parameters(self):
        """ Shape: bot params, top params"""
        return self.bot.parameters, self.top.parameters

    @parameters.setter
    def parameters(self, npz: dict):
        self.bot.parameters = {key[4:]: npz[key] for key in npz.keys()
                               if key.startswith('bot')}
        self.top.parameters = {key[4:]: npz[key] for key in npz.keys()
                               if key.startswith('top')}

    def pretrain(self, data, nE=100, bS=10, lR=0.01, k=1):
        """ Pretrain the DBM by training its constituent RBMS sequentially.

        Args:
            data: a 2D ndarray in which each row is a data vector.
            nE: the number of training epochs (default 100).
            bS: the size of the minibatches used (default 10).
            lR: the learning rate (default 0.01).
            k: the number of sampling steps used by the algorithm
            (default 1).
        """
        self.bot.train(data, nE=nE, bS=bS, lR=lR, k=k, m_bu=3)
        h1 = self.bot.pass_through(data)
        self.top.train(h1, nE=nE, bS=bS, lR=lR, k=k, m_bu=2, m_td=3)
        del h1

    def train(self, data, nE=1000, bS = 10, lR=0.01, k=1):
        """ Train the DBM using the CDk algorithm over all layers
        simultaneously. The DBM's weights and biases will be updated.

        Args:
            data: a 2D ndarray in which each row is a data vector.
            nE: the number of training epochs (default 1000).
            bS: the size of the minibatches used (default 10).
            lR: the learning rate (default 0.01).
            k: the number of sampling steps used by the algorithm
            (default 1).
        """
        for e in range(nE):
            print(f'DBM epoch: ', e)
            rows = np.random.permutation(len(data))
            for b in range(1, int(len(data) / bS) + 2):
                batch = data[rows[(b - 1) * bS:b * bS]]
                act_bS = len(batch)  # data length may not be divisible by bS
                if act_bS == 0:
                    break
                vis = batch

                # Initialize hid2 with a bottom-up pass.
                _, hid1 = self.bot.to_hid(vis, w_mult=2)
                hid2_p, hid2 = self.top.to_hid(hid1)

                # Compute hid1.
                hid1_p, hid1 = self.to_mid(vis, hid2)

                # Copy the hidden state and run k Gibbs sample steps.
                hid1_m = copy.deepcopy(hid1)
                vis_m_p = None
                vis_m = None
                hid1_m_p = None
                hid2_m_p = None
                for _ in range(k):
                    vis_m_p, vis_m = self.bot.to_vis(hid1_m)
                    hid2_m_p, hid2_m = self.top.to_hid(hid1_m)
                    hid1_m_p, hid1_m = self.to_mid(vis_m, hid2_m)

                # Update weights and biases.
                self.bot.update_W(vis, hid1_p, vis_m_p, vis_m,
                                  hid1_m_p, bS=act_bS, lR=lR)
                self.top.update_W(hid1, hid2_p, hid1_m_p, hid1_m,
                                  hid2_m_p, bS=act_bS, lR=lR)
                self.bot.update_b_vis(vis, vis_m_p, vis_m, bS=act_bS, lR=lR)
                self.bot.update_b_hid(hid1_p, hid1_m_p, bS=act_bS, lR=lR)
                self.top.b_vis = self.bot.b_hid
                self.top.update_b_hid(hid2_p, hid2_m_p, bS=act_bS, lR=lR)

    def to_mid(self, vis, hid2):
        """ Compute the output of the middle layer hidden nodes.
            Compute the activation, then call the sigmoid activation function
            over it and sample from the resulting distribution.

        Args:
            vis: the state of the visible nodes.
            hid2: the state of the top layer hidden nodes.

        Returns:
            hid1_p: the result of calling the sigmoid activation function
            over the activation. These are probabilities.
            hid1: the output of the middle layer hidden nodes.
        """
        hid1_a = self.bot.hid_act(vis) + self.top.vis_act(hid2, b=False)
        hid1_p = sigmoid(hid1_a)
        hid1 = (hid1_p > np.random.rand(self.bot.n_hid)).astype(int)
        return hid1_p, hid1

    def sample_vis(self, example, nC=100):
        """ Generate a data reconstruction.

        Args:
            example: a real data vector.
            nC: the number of Gibbs sampling cycles to take for the hidden
            layers to settle (default 100).

        Returns:
            vis: a reconstructed data vector.
        """
        vis = example
        hid2 = np.random.randint(0, 2, self.top.n_hid)
        _, hid1 = self.to_mid(vis, hid2)
        for i in range(nC):
            _, hid2 = self.top.to_hid(hid1)
            _, hid1 = self.to_mid(vis, hid2)
        _, vis = self.bot.to_vis(hid1)
        return vis


class BimodalDBM:
    """ Bimodal Deep Boltzmann Machine.
    Contains two DBMs and an additional joint layer.
    """
    def __init__(self, dbm1: DBM, dbm2: DBM, n_joint):
        np_rng = np.random.RandomState(1234)
        self.left = dbm1
        self.right = dbm2
        self.n_joint = n_joint
        self.W_left = np.asarray(np_rng.uniform(
            low=-0.2 * np.sqrt(6. / (n_joint + self.left.top.n_hid)),
            high=0.2 * np.sqrt(6. / (n_joint + self.left.top.n_hid)),
            size=(self.left.top.n_hid, n_joint)))
        self.W_right = np.asarray(np_rng.uniform(
            low=-0.2 * np.sqrt(6. / (n_joint + self.right.top.n_hid)),
            high=0.2 * np.sqrt(6. / (n_joint + self.right.top.n_hid)),
            size=(self.right.top.n_hid, n_joint)))
        self.b_joint = np.zeros(n_joint)

    @property
    def parameters(self):
        """ Shape: ((left params), (right params), left W, right W, joint b)"""
        return self.left.parameters, self.right.parameters,\
               self.W_left, self.W_right, self.b_joint

    @parameters.setter
    def parameters(self, npz: dict):
        self.left.parameters = {key[2:]: npz[key] for key in npz.keys()
                                if key.startswith('l_')}
        self.right.parameters = {key[2:]: npz[key] for key in npz.keys()
                                 if key.startswith('r_')}
        self.W_left = npz['W_l']
        self.W_right = npz['W_r']
        self.b_joint = npz['b_joint']

    def pretrain(self, mod1, mod2, nE=100, bS=10, lR=0.01, k=1):
        """ Pretrain the bimodal DBM by training its constituent DBMS and then
        the joint layer separately.

        Args:
            mod1: a 2D ndarray in which each row is a data vector of the
            first modality.
            mod2: a 2D ndarray in which each row is a data vector of the
            second modality.
            nE: the number of training epochs (default 100).
            bS: the size of the minibatches used (default 10).
            lR: the learning rate (default 0.01).
            k: the number of sampling steps used by the algorithm
            (default 1).
        """
        if len(mod1) != len(mod2):
            raise ValueError("The amount of data for each modality differs.")

        # Left
        self.left.bot.train(mod1, nE=nE, bS=bS, lR=lR, k=k, m_bu=2)
        hid1_l = self.left.bot.pass_through(mod1)
        self.left.top.train(hid1_l, nE=nE, bS=bS, lR=lR, k=k, m_bu=2, m_td=2)
        hid2_l = self.left.top.pass_through(hid1_l)
        del hid1_l

        # Right
        self.right.bot.train(mod2, nE=nE, bS=bS, lR=lR, k=k, m_bu=2)
        hid1_r = self.right.bot.pass_through(mod2)
        self.right.top.train(hid1_r, nE=nE, bS=bS, lR=lR, k=k, m_bu=2, m_td=2)
        hid2_r = self.right.top.pass_through(hid1_r)
        del hid1_r

        # Joint
        for e in range(nE):
            print(f'Joint epoch: ', e)
            rows = np.random.permutation(len(mod1))
            for b in range(1, int(len(hid2_l) / bS) + 2):
                batch_l = hid2_l[rows[(b - 1) * bS:b * bS]]
                batch_r = hid2_r[rows[(b - 1) * bS:b * bS]]
                act_bS = len(batch_l)  # data length may not be divisible by bS
                if act_bS == 0:
                    break
                vis_l = batch_l  # vis in the context of this part of the model
                vis_r = batch_r
                joint_p, joint = self.to_joint(vis_l, vis_r)
                joint_m = copy.deepcopy(joint)
                vis_l_m_p = joint_m_p = vis_r_m_p = vis_l_m = vis_r_m = None
                for _ in range(k):
                    vis_l_m_p, vis_l_m, vis_r_m_p, vis_r_m = \
                        self.to_hid2_pre(joint_m)
                    joint_m_p, joint_m = self.to_joint(vis_l_m, vis_r_m)
                self.update_Ws(vis_l, vis_r, joint_p, vis_l_m_p,
                               vis_r_m_p, joint_m_p, bS=act_bS, lR=lR)
                self.left.top.b_hid += lR/act_bS *\
                                                np.sum(vis_l - vis_l_m, axis=0)
                self.right.top.b_hid += lR/act_bS * \
                                                np.sum(vis_r - vis_r_m, axis=0)
                self.update_b_joint(joint_p, joint_m_p, bS=act_bS, lR=lR)
        del hid2_l, hid2_r

    def train(self, mod1, mod2, nE=1000, bS = 10, lR=0.01, k=1):
        """ Train the bimodal DBM using the CDk algorithm over all layers
        simultaneously. The model's weights and biases will be updated.

        Args:
            mod1: a 2D ndarray in which each row is a data vector of the
            first modality.
            mod2: a 2D ndarray in which each row is a data vector of the
            second modality.
            nE: the number of training epochs (default 1000).
            bS: the size of the minibatches used (default 10).
            lR: the learning rate (default 0.01).
            k: the number of sampling steps used by the algorithm
            (default 1).
        """
        # TODO: annealing?
        if len(mod1) != len(mod2):
            raise ValueError("The amount of data for each modality differs.")
        for e in range(nE):
            print(f'mDBM epoch: ', e)
            rows = np.random.permutation(len(mod1))
            for b in range(1, int(len(mod1) / bS) + 2):
                batch_l = mod1[rows[(b - 1) * bS:b * bS]]
                batch_r = mod2[rows[(b - 1) * bS:b * bS]]
                act_bS = len(batch_l)  # data length may not be divisible by bS
                if act_bS == 0:
                    break
                vis_l = batch_l
                vis_r = batch_r

                # Initialize hidden layers with a bottom-up pass.
                _, hid1_l = self.left.bot.to_hid(vis_l, w_mult=2)
                _, hid2_l = self.left.top.to_hid(hid1_l, w_mult=2)
                _, hid1_r = self.right.bot.to_hid(vis_r, w_mult=2)
                _, hid2_r = self.right.top.to_hid(hid1_r, w_mult=2)
                joint_p, joint = self.to_joint(hid2_l, hid2_r)

                # Compute hid1 and hid2 for the left and right DBMs.  # TODO: repeat until convergence?
                hid1_l_p, hid1_l = self.left.to_mid(vis_l, hid2_l)
                hid1_r_p, hid1_r = self.right.to_mid(vis_r, hid2_r)
                hid2_l_p, hid2_l, hid2_r_p, hid2_r = \
                    self.to_hid2(hid1_l, hid1_r, joint)

                # Copy the hidden states and run k Gibbs sample steps.
                hid1_l_m = copy.deepcopy(hid1_l)
                hid1_r_m = copy.deepcopy(hid1_r)
                joint_m = copy.deepcopy(joint)
                vis_l_m_p = vis_l_m = hid1_l_m_p = hid2_l_m_p = vis_r_m_p = \
                    vis_r_m = hid1_r_m_p = hid2_r_m_p = joint_m_p = None
                for _ in range(k):
                    # Compute odd layers.
                    vis_l_m_p, vis_l_m = self.left.bot.to_vis(hid1_l_m)
                    vis_r_m_p, vis_r_m = self.right.bot.to_vis(hid1_r_m)
                    hid2_l_m_p, hid2_l_m, hid2_r_m_p, hid2_r_m = \
                        self.to_hid2(hid1_l_m, hid1_r_m, joint_m)

                    # Compute even layers.
                    hid1_l_m_p, hid1_l_m = self.left.to_mid(vis_l_m, hid2_l_m)
                    hid1_r_m_p, hid1_r_m = self.right.to_mid(vis_r_m, hid2_r_m)
                    joint_m_p, joint_m = self.to_joint(hid2_l_m, hid2_r_m)

                # Update weights and biases.  # TODO: abstract update_W method in DBM / all models
                self.left.bot.update_W(vis_l, hid1_l_p, vis_l_m_p,
                                       vis_l_m, hid1_l_m_p, bS=act_bS, lR=lR)
                self.left.top.update_W(hid1_l, hid2_l_p, hid1_l_m_p,
                                       hid1_l_m, hid2_l_m_p, bS=act_bS, lR=lR)
                self.left.bot.update_b_vis(vis_l, vis_l_m_p, vis_l_m,
                                           bS=act_bS, lR=lR)
                self.left.bot.update_b_hid(hid1_l_p, hid1_l_m_p,
                                           bS=act_bS, lR=lR)
                self.left.top.b_vis = self.left.bot.b_hid
                self.left.top.update_b_hid(hid2_l_p, hid2_l_m_p,
                                           bS=act_bS, lR=lR)
                self.right.bot.update_W(vis_r, hid1_r_p, vis_r_m_p,
                                        vis_r_m, hid1_r_m_p, bS=act_bS, lR=lR)
                self.right.top.update_W(hid1_r, hid2_r_p, hid1_r_m_p,
                                        hid1_r_m, hid2_r_m_p, bS=act_bS, lR=lR)
                self.right.bot.update_b_vis(vis_r, vis_r_m_p, vis_r_m,
                                            bS=act_bS, lR=lR)
                self.right.bot.update_b_hid(hid1_r_p, hid1_r_m_p,
                                            bS=act_bS, lR=lR)
                self.right.top.b_vis = self.right.bot.b_hid
                self.right.top.update_b_hid(hid2_r_p, hid2_r_m_p,
                                            bS=act_bS, lR=lR)
                self.update_Ws(hid2_l_p, hid2_r_p, joint_p, hid2_l_m_p,
                               hid2_r_m_p, joint_m_p, bS=act_bS, lR=lR)
                self.update_b_joint(joint_p, joint_m_p, bS=act_bS, lR=lR)

    def joint_act(self, hid_l, hid_r):
        """ Compute the activation of the joint nodes.

        Args:
            hid_l: the state of the left DBM's top layer nodes.
            hid_r: the state of the right DBM's top layer nodes.

        Returns:
            act: the activation of the hidden nodes.
        """
        act = np.dot(hid_l, self.W_left) + np.dot(hid_r, self.W_right) + \
              self.b_joint
        return act

    def to_joint(self, hid_l, hid_r):
        """ Compute the output of the joint layer hidden nodes.
        Compute the activation, then call the sigmoid activation function
        over it and sample from the resulting distribution.

        Args:
            hid_l: the state of the left DBM's top layer nodes.
            hid_r: the state of the right DBM's top layer nodes.

        Returns:
            joint_p: the result of calling the sigmoid activation function
            over the activation. These are probabilities.
            joint: the output of the joint layer hidden nodes.
        """
        joint_a = self.joint_act(hid_l, hid_r)
        joint_p = sigmoid(joint_a)
        joint = (joint_p > np.random.rand(self.n_joint)).astype(int)
        return joint_p, joint

    def to_hid2(self, hid1_l, hid1_r, joint):
        """ Compute the output of the both DBMs' top layer hidden nodes.
        Compute their activations, then call the sigmoid activation function
        over those and sample from the resulting distributions.

        Args:
            hid1_l: the state of the left DBM's middle layer nodes.
            hid1_r: the state of the right DBM's middle layer nodes.
            joint: the state of the joint layer nodes.

        Returns:
            hid2_l_p: the result of calling the sigmoid activation function
            over the left DBM's top layer activation. These are probabilities.
            hid2_l: the output of the left DBM's top layer hidden nodes.
            hid2_r_p: the result of calling the sigmoid activation function
            over the right DBM's top layer activation. These are probabilities.
            hid2_r: the output of the right DBM's top layer hidden nodes.
        """
        # TODO: one side only? Through boolean or two functions.
        hid2_l_a = np.dot(hid1_l, self.left.top.W) + \
                   np.dot(joint, self.W_left.T) + self.left.top.b_hid
        hid2_l_p = sigmoid(hid2_l_a)
        hid2_l = (hid2_l_p > np.random.rand(self.left.top.n_hid)).astype(int)
        hid2_r_a = np.dot(hid1_r, self.right.top.W) + \
                   np.dot(joint, self.W_right.T) + self.right.top.b_hid
        hid2_r_p = sigmoid(hid2_r_a)
        hid2_r = (hid2_r_p > np.random.rand(self.right.top.n_hid)).astype(int)
        return hid2_l_p, hid2_l, hid2_r_p, hid2_r

    def to_hid2_pre(self, joint):
        """ Compute the states of the DBMs' top hidden layers only from the
        joint layer. Used in pretraining.

        Args:
            joint: the state of the joint layer nodes.

        Returns:
            hid2_l_p: the result of calling the sigmoid activation function
            over the left DBM's top layer activation. These are probabilities.
            hid2_l: the output of the left DBM's top layer hidden nodes.
            hid2_r_p: the result of calling the sigmoid activation function
            over the right DBM's top layer activation. These are probabilities.
            hid2_r: the output of the right DBM's top layer hidden nodes.
        """
        hid2_l_a = np.dot(joint, self.W_left.T * 2) + self.left.top.b_hid
        hid2_l_p = sigmoid(hid2_l_a)
        hid2_l = (hid2_l_p > np.random.rand(self.left.top.n_hid)).astype(int)
        hid2_r_a = np.dot(joint, self.W_right.T * 2) + self.right.top.b_hid
        hid2_r_p = sigmoid(hid2_r_a)
        hid2_r = (hid2_r_p > np.random.rand(self.right.top.n_hid)).astype(int)
        return hid2_l_p, hid2_l, hid2_r_p, hid2_r

    def update_Ws(self, hid2_l, hid2_r, joint_p, hid2_l_m_p,
                  hid2_r_m_p, joint_m_p, bS, lR):
        """ Update the weights of the bimodal DBM based on statistics obtained
        with the CDk algorithm.

        Args:
            hid2_l: the values of the left DBM's top layer hidden nodes with
            clamped input.
            hid2_r: the values of the right DBM's top layer hidden nodes with
            clamped input.
            joint_p: the probabilities of the joint nodes with clamped input.
            hid2_l_m_p: the probabilities of the left DBM's top layer hidden
            nodes without clamped input.
            hid2_r_m_p: the probabilities of the right DBM's top layer hidden
            nodes without clamped input.
            joint_m_p: the probabilities of the joint nodes without clamped
            input.
            bS: the size of the minibatches used.
            lR: the learning rate used.
        """
        self.W_left += lR/bS * (np.dot(hid2_l.T, joint_p) -
                                np.dot(hid2_l_m_p.T, joint_m_p))
        self.W_right += lR/bS * (np.dot(hid2_r.T, joint_p) -
                                 np.dot(hid2_r_m_p.T, joint_m_p))

    def update_b_joint(self, joint_p, joint_m_p, bS, lR):
        """ Update the joint layer biases of the bimodal DBM based on
        statistics obtained with the CDk algorithm.

        Args:
            joint_p: the probabilities of the joint nodes with clamped input.
            joint_m_p: the probabilities of the joint nodes without clamped
            input.
            bS: the size of the minibatches used.
            lR: the learning rate used.
        """
        self.b_joint += lR/bS * np.sum(joint_p - joint_m_p, axis=0)

    def sample_vis(self, mod1, mod2, nC=100):
        """ Generate a data reconstruction.

        Args:
            mod1: a real data vector of one modality.
            mod2: a corresponding data vector of another modality.
            nC: the number of Gibbs sampling cycles to take for the hidden
            layers to settle (default 100).

        Returns:
            vis_l: a reconstructed data vector in one modality.
            vis_r: a reconstructed data vector in another modality.
        """
        vis_l = mod1
        vis_r = mod2

        # Initialize all hidden layers randomly
        hid1_l = np.random.randint(0, 2, self.left.bot.n_hid)
        hid2_l = np.random.randint(0, 2, self.left.top.n_hid)
        hid1_r = np.random.randint(0, 2, self.right.bot.n_hid)
        hid2_r = np.random.randint(0, 2, self.right.top.n_hid)
        joint = np.random.randint(0, 2, self.n_joint)

        _, hid1_l = self.left.to_mid(vis_l, hid2_l)
        _, hid1_r = self.right.to_mid(vis_r, hid2_r)
        for _ in range(nC):
            _, hid2_l, _, hid2_r = self.to_hid2(hid1_l, hid1_r, joint)
            _, joint = self.to_joint(hid2_l, hid2_r)
            _, hid2_l, _, hid2_r = self.to_hid2(hid1_l, hid1_r, joint)
            _, hid1_l = self.left.to_mid(vis_l, hid2_l)
            _, hid1_r = self.right.to_mid(vis_r, hid2_r)
        _, vis_l = self.left.bot.to_vis(hid1_l)
        _, vis_r = self.right.bot.to_vis(hid1_r)

        return vis_l, vis_r


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


def softmax(x):
    exp = np.exp(x)
    if x.ndim > 1:
        div = np.sum(exp, axis=1)
        div = np.reshape(div, (len(div), 1))
    else:
        div = sum(exp)
    return exp / div

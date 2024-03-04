# @author: H.Zhu
# @date: 2018/10/24 21:24

import math

import torch
import torch.nn.functional as F


def log_sum_exp(x, axis=None):
    x_max = torch.max(x, axis)[0]
    y = torch.log((torch.exp(x - x_max)).sum(axis)) + x_max
    return y


def raise_measure_error(measure):
    supported_measures = ['GAN', 'JSD', 'X2', 'KL', 'RKL', 'DV', 'H2', 'W1']
    raise NotImplementedError(
        'Measure `{}` not supported. Supported: {}'.format(measure,
                                                           supported_measures))


def get_positive_expectation(p_samples, measure, average=True):
    log_2 = math.log(2.)

    if measure == 'GAN':
        Ep = - F.softplus(-p_samples)
    elif measure == 'JSD':
        Ep = log_2 - F.softplus(- p_samples)
    elif measure == 'X2':
        Ep = p_samples ** 2
    elif measure == 'KL':
        Ep = p_samples + 1.
    elif measure == 'RKL':
        Ep = -torch.exp(-p_samples)
    elif measure == 'DV':
        Ep = p_samples
    elif measure == 'H2':
        Ep = 1. - torch.exp(-p_samples)
    elif measure == 'W1':
        Ep = p_samples

    else:
        raise_measure_error(measure)
        return

    if average:
        return Ep.mean()
    else:
        return Ep


def get_negative_expectation(q_samples, measure, average=True):
    log_2 = math.log(2.)

    if measure == 'GAN':
        Eq = F.softplus(-q_samples) + q_samples
    elif measure == 'JSD':
        Eq = F.softplus(-q_samples) + q_samples - log_2
    elif measure == 'X2':
        Eq = -0.5 * ((torch.sqrt(q_samples ** 2) + 1.) ** 2)
    elif measure == 'KL':
        Eq = torch.exp(q_samples)
    elif measure == 'RKL':
        Eq = q_samples - 1.
    elif measure == 'DV':
        Eq = log_sum_exp(q_samples, 0) - math.log(q_samples.size(0))
    elif measure == 'H2':
        Eq = torch.exp(q_samples) - 1.
    elif measure == 'W1':
        Eq = q_samples
    else:
        raise_measure_error(measure)
        return

    if average:
        return Eq.mean()
    else:
        return Eq


class MINE(object):
    def __init__(self, discriminator,  measure='GAN'):
        super(MINE, self).__init__()

        self.discriminator = discriminator
        # if optim is None:
        optim = torch.optim.Adam(self.discriminator.parameters(), lr=0.001)
        self.optim = optim
        self.measure = measure

    def score(self, X_P, Z_P, X_Q, Z_Q, grad=False):
        P_samples = self.discriminator(X_P, Z_P)
        Q_samples = self.discriminator(X_Q, Z_Q)
        # print(P_samples.size(),Q_samples.size(),Z_P.size())
        # raise RuntimeError
        if not grad:
            P_samples = P_samples.detach().data
            Q_samples = Q_samples.detach().data

        E_pos = get_positive_expectation(P_samples, self.measure)
        E_neg = get_negative_expectation(Q_samples, self.measure)

        return E_pos, E_neg, P_samples, Q_samples

    def update(self, X_real, Z_real, X_fake, Z_fake):
        self.optim.zero_grad()
        X_P, Z_P = X_real, Z_real
        X_Q, Z_Q = X_fake, Z_fake

        E_pos, E_neg, P_samples, Q_samples = self.score(X_P, Z_P, X_Q, Z_Q, grad=True)
        difference = E_pos - E_neg
        loss = -difference

        loss.backward()
        self.optim.step()

        return difference.detach().data



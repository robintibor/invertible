import numpy as np
import torch as th
from invertible.gaussian import get_gauss_samples, \
    get_mixture_gaussian_log_probs
import torch.nn.functional as F
from invertible.gaussian import get_gaussian_log_probs
from torch import nn


class PerDimWeightedMix(nn.Module):
    def __init__(self, n_classes, n_mixes, n_dims, init_std=1e-1,
                 **mix_dist_kwargs):
        super().__init__()
        self.n_classes = n_classes
        self.n_mixes = n_mixes
        self.n_dims = n_dims
        self.mix_dist = NClassIndependentDist(n_classes=n_mixes, n_dims=n_dims, **mix_dist_kwargs)
        self.mix_dist.class_means.data.normal_(mean=0, std=init_std)
        self.mix_dist.class_log_stds.data.normal_(mean=0, std=init_std)
        self.weights = nn.Parameter(th.zeros(n_classes, n_mixes, n_dims))

    def forward(self, z, fixed=None):
        fixed = fixed or {}
        _, lps = self.mix_dist(z, fixed=dict(sum_dims=False))
        log_weights = F.log_softmax(self.weights, dim=1)
        lp_weighted = lps.unsqueeze(1) + log_weights.unsqueeze(0)
        # examples x classes x mixtures x dims
        if fixed.get('sum_mixtures', True):
            lp = th.logsumexp(lp_weighted, dim=2)
        if fixed.get('sum_dims', True):
            lp = th.sum(lp, dim=-1)

        if fixed.get('y', None) is not None:
            y = fixed['y']
            if y.ndim > 1:
                # assume one hot encoding
                y = y.argmax(dim=1, keepdim=True)
            else:
                y = y.unsqueeze(1)

            repeats = ()
            while y.ndim < lp.ndim:
                repeats = repeats + (lp.shape[y.ndim],)
                y = y.unsqueeze(-1)
            y = y.repeat((1, 1) + repeats)
            lp = lp.gather(
                dim=1, index=y).squeeze(1)
        return z, lp

    def invert(self, z, fixed=None):
        fixed = fixed or {}
        if z is None:
            assert 'n_samples' in fixed
            if 'y' in fixed:
                i_class = fixed['y']
                assert isinstance(i_class, int)
                z = self.get_samples(i_class, fixed['n_samples'], std_factor=1)
                y = th.zeros(len(z), dtype=th.int64, device=z.device) + i_class
                logdet = self.forward(z, fixed={**fixed, **dict(y=y)})[1]
            else:
                raise ValueError("to be implemented")
                y = self.get_unlabeled_samples(fixed['n_samples'],
                                               std_factor=1)
                logdet = self.log_probs_per_class(y)
        else:
            logdet = self.forward(z, fixed=fixed)[1]
        # compute unconditional logdet
        if fixed.get('y', None) is None:
            #For now
            logdet = th.logsumexp(logdet, dim=1) - np.log(logdet.shape[1])
        return z, logdet

    def get_samples(self, i_class, n_samples, std_factor):
        prob_mixes = th.softmax(self.weights, dim=1)
        selected_components = th.multinomial(
            prob_mixes[i_class].t(), num_samples=n_samples).t()
        selected_means = self.mix_dist.class_means.gather(
            dim=0, index=selected_components)
        selected_log_stds = self.mix_dist.class_log_stds.gather(
            dim=0, index=selected_components)
        selected_stds = th.exp(selected_log_stds)
        normal_samples = th.randn_like(selected_means)
        samples = (normal_samples * selected_stds * std_factor) + selected_means
        return samples



class MergeLogDets(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x, fixed):
        y, logdets = self.module(x, fixed=fixed)
        if fixed['y'] is None:
            n_components = logdets.shape[1]
            logdets = th.logsumexp(logdets, dim=1) - np.log(n_components)
        return y, logdets

    def invert(self, y, fixed):
        return self.module.invert(y, fixed=fixed)


class PerClass(nn.Module):
    def __init__(self, dist):
        super().__init__()
        self.dist = dist

    def forward(self, x, fixed=None):
        logdet = self.dist.log_probs_per_class(x)
        if hasattr(fixed, '__getitem__') and 'y' in fixed and fixed['y'] is not None:
            y = fixed['y']
            logdet = logdet.gather(
                dim=1, index=y.argmax(dim=1, keepdim=True)).squeeze(1)
        return x, logdet

    def invert(self, y, fixed=None):
        if hasattr(fixed, '__getitem__') and 'y' in fixed:
            assert fixed['y'] == None, "other not implemented"
        if y is None:
            assert 'n_samples' in fixed
            y = self.dist.get_unlabeled_samples(fixed['n_samples'],
                                                std_factor=1)
        logdet = self.dist.log_probs_per_class(y)
        return y, logdet


class Unlabeled(nn.Module):
    def __init__(self, dist):
        super().__init__()
        self.dist = dist

    def forward(self, x, fixed=None):
        logdet = self.dist.log_prob_unlabeled(x)
        return x, logdet

    def invert(self, y, fixed=None):
        if y is None:
            assert 'n_samples' in fixed
            y = self.dist.get_unlabeled_samples(fixed['n_samples'],
                                                std_factor=1)
        logdet = self.dist.log_prob_unlabeled(y)
        return y, logdet


class ZeroDist(nn.Module):
    def log_prob_unlabeled(self, x):
        return 0


class NClassIndependentDist(nn.Module):
    def __init__(self, n_classes=None, n_dims=None, optimize_mean=True,
                 optimize_std=True,
                 truncate_to=None,
                 means=None, log_stds=None):
        super().__init__()
        if means is not None:
            assert log_stds is not None
            self.class_means = means
            self.class_log_stds = log_stds
        else:
            if optimize_mean:
                self.class_means = nn.Parameter(
                    th.zeros(n_classes, n_dims, requires_grad=True))
            else:
                self.register_buffer('class_means',
                                     th.zeros(n_classes, n_dims, ))
            if optimize_std:
                self.class_log_stds = nn.Parameter(
                    th.zeros(n_classes, n_dims, requires_grad=True))
            else:
                self.register_buffer('class_log_stds', th.zeros(n_classes, n_dims, ))

        self.truncate_to = truncate_to

    def forward(self, x, fixed=None):
        fixed = fixed or {}
        logdet = self.log_probs_per_class(x, sum_dims=fixed.get('sum_dims', True))
        if 'y' in fixed and fixed['y'] is not None:
            y = fixed['y']
            if y.ndim > 1:
                # assume one hot encoding
                y = y.argmax(dim=1, keepdim=True)
            else:
                y = y.unsqueeze(1)
            logdet = logdet.gather(
                dim=1, index=y).squeeze(1)
        return x, logdet

    def invert(self, y, fixed=None):
        if y is None:
            assert 'n_samples' in fixed
            if hasattr(fixed, '__getitem__') and 'y' in fixed:
                i_class = fixed['y']
                assert isinstance(i_class, int)
                y = self.get_samples(i_class, fixed['n_samples'], std_factor=1)
                logdet = self.log_prob_class(i_class, y)
            else:
                y = self.get_unlabeled_samples(fixed['n_samples'],
                                                std_factor=1)
                logdet = self.log_probs_per_class(y)

        else:
            if (fixed is not None) and (fixed.get('y', None) is not None):
                i_class = self.fixed['y']
                logdet = self.log_prob_class(i_class, y)
            else:
                logdet = self.log_prob_unlabeled(y)
        return y, logdet

    def get_mean_std(self, i_class):
        cur_mean, cur_log_std = self.get_mean_log_std(i_class)
        return cur_mean, th.exp(cur_log_std)

    def get_mean_log_std(self, i_class):
        cur_mean = self.class_means[i_class]
        cur_log_std = self.class_log_stds[i_class]
        return cur_mean, cur_log_std

    def get_samples(self, i_class, n_samples, std_factor=1):
        cur_mean, cur_std = self.get_mean_std(i_class)
        samples = get_gauss_samples(
            n_samples, cur_mean, cur_std * std_factor,
            truncate_to=self.truncate_to
        )
        return samples

    def get_unlabeled_samples(self, n_samples, std_factor=1):
        choices = np.random.choice(range(len(self.class_means)),
            size=n_samples,)
        bincounts = np.bincount(choices)
        all_samples = th.cat([self.get_samples(
            i_mixture, bincounts[i_mixture], std_factor=std_factor)
            for i_mixture in np.flatnonzero(bincounts)], dim=0)
        return all_samples

    def change_to_other_class(self, outs, i_class_from, i_class_to, eps=1e-6):
        mean_from, std_from = self.get_mean_std(i_class_from)
        mean_to, std_to = self.get_mean_std(i_class_to)
        normed = (outs - mean_from.unsqueeze(0)) / (std_from.unsqueeze(0) + eps)
        transformed = (normed * std_to.unsqueeze(0)) + mean_to.unsqueeze(0)
        return transformed

    def log_prob_class(self, i_class, outs, clamp_max_sigma=None):
        mean, log_std = self.get_mean_log_std(i_class)
        log_probs = get_gaussian_log_probs(mean, log_std, outs,
                                           clamp_max_sigma=clamp_max_sigma)
        return log_probs

    def log_probs_per_class(self, y, clamp_max_sigma=None, sum_dims=True):
        log_probs = get_mixture_gaussian_log_probs(
            self.class_means, self.class_log_stds, y,
            clamp_max_sigma=clamp_max_sigma, sum_dims=sum_dims)
        return log_probs

    def log_probs_per_weighted_class(self, y, clamp_max_sigma=None):
        n_classes = len(self.class_means)
        log_probs = get_mixture_gaussian_log_probs(
            self.class_means, self.class_log_stds, y,
            clamp_max_sigma=clamp_max_sigma) - np.log(n_classes)
        return log_probs

    def log_prob_unlabeled(self, outs, clamp_max_sigma=None):
        weighted_log_probs = self.log_probs_per_weighted_class(
            outs, clamp_max_sigma=clamp_max_sigma)
        return th.logsumexp(weighted_log_probs, dim=-1)

    def set_mean_std(self, i_class, mean, std):
        if mean is not None:
            self.class_means.data[i_class] = mean.data
        if std is not None:
            self.class_log_stds.data[i_class] = th.log(std).data

    def log_softmax(self, outs):
        log_probs = self.log_probs_per_weighted_class(
            outs, clamp_max_sigma=None)
        log_softmaxed = F.log_softmax(log_probs, dim=-1)
        return log_softmaxed

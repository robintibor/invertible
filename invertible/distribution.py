import warnings

import numpy as np
import torch as th
from .gaussian import get_gauss_samples, get_mixture_gaussian_log_probs
import torch.nn.functional as F
from torch import nn

from .gaussian import get_gaussian_log_probs


class MaskedMixDist(nn.Module):
    def __init__(self, dist, n_dims):
        super().__init__()
        self.alphas = nn.Parameter(th.zeros(n_dims))
        self.dist = dist

    def forward(self, z, fixed=None):
        fixed = fixed or {}
        _, lps = self.dist(z, fixed=dict(sum_dims=False))
        # first compute mask and (1-mask) * prob for every dim and class
        weighted_lps_pos = th.nn.functional.logsigmoid(self.alphas).view(1, 1, -1) + lps
        weighted_lps_neg = (
            th.nn.functional.logsigmoid(-self.alphas).view(1, 1, -1) + lps
        )

        # then add them together for every class pair
        eps = 1e-20
        mixed = th.log(
            (
                weighted_lps_pos.unsqueeze(2).exp()
                + weighted_lps_neg.unsqueeze(1).exp()
            ).clamp_min(eps)
        )

        # first multiply across dimensions
        # then add across classes
        lp = th.logsumexp(th.sum(mixed, dim=-1), dim=2) - np.log(mixed.shape[1])

        if not fixed.get("sum_dims", True):
            # hack for now
            warnings.warn("sum dims False not properly possible")
            lp = lp.unsqueeze(-1)

        if fixed.get("y", None) is not None:
            y = fixed["y"]
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
            lp = lp.gather(dim=1, index=y).squeeze(1)
        return z, lp

    def invert(self, z, fixed=None):
        fixed = fixed or {}
        if z is None:
            warnings.warn("Ignoring alphas in current implementation")
            z, _ = self.dist.invert(None, fixed=fixed)

        warnings.warn("Ignoring fixed in logdet computation of invert")
        logdet = self.forward(z, fixed=None)[1]
        return z, logdet


class PerDimWeightedMix(nn.Module):
    def __init__(self, n_classes, n_mixes, n_dims, init_std=1e-1, **mix_dist_kwargs):
        super().__init__()
        self.n_classes = n_classes
        self.n_mixes = n_mixes
        self.n_dims = n_dims
        self.mix_dist = NClassIndependentDist(
            n_classes=n_mixes, n_dims=n_dims, **mix_dist_kwargs
        )
        self.mix_dist.class_means.data.normal_(mean=0, std=init_std)
        self.mix_dist.class_log_stds.data.normal_(mean=0, std=init_std)
        self.weights = nn.Parameter(th.zeros(n_classes, n_mixes, n_dims))

    def forward(self, z, fixed=None):
        fixed = fixed or {}
        _, lps = self.mix_dist(z, fixed=dict(sum_dims=False))
        log_weights = F.log_softmax(self.weights, dim=1)
        lp_weighted = lps.unsqueeze(1) + log_weights.unsqueeze(0)
        # examples x classes x mixtures x dims
        if fixed.get("sum_mixtures", True):
            lp = th.logsumexp(lp_weighted, dim=2)
        if fixed.get("sum_dims", True):
            lp = th.sum(lp, dim=-1)

        if fixed.get("y", None) is not None:
            y = fixed["y"]
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
            lp = lp.gather(dim=1, index=y).squeeze(1)
        return z, lp

    def invert(self, z, fixed=None):
        fixed = fixed or {}
        if z is None:
            assert "n_samples" in fixed
            if "y" in fixed:
                i_class = fixed["y"]
                assert isinstance(i_class, int)
                z = self.get_samples(i_class, fixed["n_samples"], std_factor=1)
                y = th.zeros(len(z), dtype=th.int64, device=z.device) + i_class
                logdet = self.forward(z, fixed={**fixed, **dict(y=y)})[1]
            else:
                raise ValueError("to be implemented")
                y = self.get_unlabeled_samples(fixed["n_samples"], std_factor=1)
                logdet = self.log_probs_per_class(y)
        else:
            logdet = self.forward(z, fixed=fixed)[1]
        return z, logdet

    def get_samples(self, i_class, n_samples, std_factor):
        prob_mixes = th.softmax(self.weights, dim=1)
        selected_components = th.multinomial(
            prob_mixes[i_class].t(), num_samples=n_samples
        ).t()
        selected_means = self.mix_dist.class_means.gather(
            dim=0, index=selected_components
        )
        selected_log_stds = self.mix_dist.class_log_stds.gather(
            dim=0, index=selected_components
        )
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
        if fixed["y"] is None:
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
        if hasattr(fixed, "__getitem__") and "y" in fixed and fixed["y"] is not None:
            y = fixed["y"]
            logdet = logdet.gather(dim=1, index=y.argmax(dim=1, keepdim=True)).squeeze(
                1
            )
        return x, logdet

    def invert(self, y, fixed=None):
        if hasattr(fixed, "__getitem__") and "y" in fixed:
            assert fixed["y"] == None, "other not implemented"
        if y is None:
            assert "n_samples" in fixed
            y = self.dist.get_unlabeled_samples(fixed["n_samples"], std_factor=1)
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
            assert "n_samples" in fixed
            y = self.dist.get_unlabeled_samples(fixed["n_samples"], std_factor=1)
        logdet = self.dist.log_prob_unlabeled(y)
        return y, logdet


class ZeroDist(nn.Module):
    def log_prob_unlabeled(self, x):
        return 0


class NClassIndependentDist(nn.Module):
    def __init__(
        self,
        n_classes=None,
        n_dims=None,
        optimize_mean=True,
        optimize_std=True,
        truncate_to=None,
        means=None,
        log_stds=None,
    ):
        super().__init__()
        if means is not None:
            assert log_stds is not None
            self.class_means = means
            self.class_log_stds = log_stds
        else:
            if optimize_mean:
                self.class_means = nn.Parameter(
                    th.zeros(n_classes, n_dims, requires_grad=True)
                )
            else:
                self.register_buffer(
                    "class_means",
                    th.zeros(
                        n_classes,
                        n_dims,
                    ),
                )

            if optimize_std:
                self.class_log_stds = nn.Parameter(
                    th.zeros(n_classes, n_dims, requires_grad=True)
                )
            else:
                self.register_buffer(
                    "class_log_stds",
                    th.zeros(
                        n_classes,
                        n_dims,
                    ),
                )

        self.truncate_to = truncate_to

    def forward(self, x, fixed=None):
        fixed = fixed or {}
        logdet = self.log_probs_per_class(x, sum_dims=fixed.get("sum_dims", True))
        if "y" in fixed and fixed["y"] is not None:
            y = fixed["y"]
            if y.ndim > 1:
                # assume one hot encoding
                y = y.argmax(dim=1, keepdim=True)
            else:
                y = y.unsqueeze(1)
            logdet = logdet.gather(dim=1, index=y).squeeze(1)
        return x, logdet

    def invert(self, z, fixed=None):
        fixed = fixed or {}
        if z is None:
            assert "n_samples" in fixed
            if "y" in fixed:
                i_class = fixed["y"]
                assert isinstance(i_class, int)
                z = self.get_samples(i_class, fixed["n_samples"], std_factor=1)
                y = th.zeros(len(z), dtype=th.int64, device=z.device) + i_class
                logdet = self.forward(z, fixed={**fixed, **dict(y=y)})[1]
            else:
                raise ValueError("to be implemented")
                y = self.get_unlabeled_samples(fixed["n_samples"], std_factor=1)
                logdet = self.log_probs_per_class(y)
        else:
            logdet = self.forward(z, fixed=fixed)[1]
        return z, logdet

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
            n_samples, cur_mean, cur_std * std_factor, truncate_to=self.truncate_to
        )
        return samples

    def get_unlabeled_samples(self, n_samples, std_factor=1):
        choices = np.random.choice(
            range(len(self.class_means)),
            size=n_samples,
        )
        bincounts = np.bincount(choices)
        all_samples = th.cat(
            [
                self.get_samples(i_mixture, bincounts[i_mixture], std_factor=std_factor)
                for i_mixture in np.flatnonzero(bincounts)
            ],
            dim=0,
        )
        return all_samples

    def change_to_other_class(self, outs, i_class_from, i_class_to, eps=1e-6):
        mean_from, std_from = self.get_mean_std(i_class_from)
        mean_to, std_to = self.get_mean_std(i_class_to)
        normed = (outs - mean_from.unsqueeze(0)) / (std_from.unsqueeze(0) + eps)
        transformed = (normed * std_to.unsqueeze(0)) + mean_to.unsqueeze(0)
        return transformed

    def log_prob_class(self, i_class, outs, clamp_max_sigma=None):
        mean, log_std = self.get_mean_log_std(i_class)
        log_probs = get_gaussian_log_probs(
            mean, log_std, outs, clamp_max_sigma=clamp_max_sigma
        )
        return log_probs

    def log_probs_per_class(self, y, clamp_max_sigma=None, sum_dims=True):
        log_probs = get_mixture_gaussian_log_probs(
            self.class_means,
            self.class_log_stds,
            y,
            clamp_max_sigma=clamp_max_sigma,
            sum_dims=sum_dims,
        )
        return log_probs

    def log_probs_per_weighted_class(self, y, clamp_max_sigma=None):
        n_classes = len(self.class_means)
        log_probs = get_mixture_gaussian_log_probs(
            self.class_means, self.class_log_stds, y, clamp_max_sigma=clamp_max_sigma
        ) - np.log(n_classes)
        return log_probs

    def log_prob_unlabeled(self, outs, clamp_max_sigma=None):
        weighted_log_probs = self.log_probs_per_weighted_class(
            outs, clamp_max_sigma=clamp_max_sigma
        )
        return th.logsumexp(weighted_log_probs, dim=-1)

    def set_mean_std(self, i_class, mean, std):
        if mean is not None:
            self.class_means.data[i_class] = mean.data
        if std is not None:
            self.class_log_stds.data[i_class] = th.log(std).data

    def log_softmax(self, outs):
        log_probs = self.log_probs_per_weighted_class(outs, clamp_max_sigma=None)
        log_softmaxed = F.log_softmax(log_probs, dim=-1)
        return log_softmaxed


class ClassWeightedGaussianMixture(nn.Module):
    def __init__(self, n_classes, n_mixes, n_dims, init_std=1e-1, **mix_dist_kwargs):
        super().__init__()
        self.n_classes = n_classes
        self.n_mixes = n_mixes
        self.n_dims = n_dims
        self.mix_dist = NClassIndependentDist(
            n_classes=n_mixes, n_dims=n_dims, **mix_dist_kwargs
        )
        self.mix_dist.class_means.data.normal_(mean=0, std=init_std)
        self.mix_dist.class_log_stds.data.normal_(mean=0, std=init_std)
        self.weights = nn.Parameter(
            th.zeros(
                n_classes,
                n_mixes,
            )
        )

    def forward(self, z, fixed=None):
        fixed = fixed or {}
        mixture_reduce = fixed.get("mixture_reduce", "sum")
        sum_dims = fixed.get("sum_dims", True)
        assert sum_dims or mixture_reduce == "none"
        lp_per_mixture = self.mix_dist.log_probs_per_class(z, sum_dims=sum_dims)
        # examples x mixtures (x dims)
        log_weights = th.log_softmax(self.weights, dim=1)
        if not sum_dims:
            log_weights = log_weights.unsqueeze(-1) / lp_per_mixture.shape[-1]
        # classes x mixtures (x 1)
        # add:
        # examples x 1 x mixtures x (dims)
        # 1 x classes x mixtures (x 1)
        lp_weighted = lp_per_mixture.unsqueeze(1) + log_weights.unsqueeze(0)
        # examples x classes x mixtures (x dims)
        if mixture_reduce == "sum":
            lp_weighted = th.logsumexp(lp_weighted, dim=2)
        elif mixture_reduce == "max":
            lp_weighted = th.max(lp_weighted, dim=2)[0]
        else:
            assert mixture_reduce == "none"
        return z, lp_weighted

    def invert(self, z, fixed=None):
        fixed = fixed or {}
        if z is None:
            raise ValueError("Not implemented")
        else:
            logdet = self.forward(z, fixed=fixed)[1]
        return z, logdet


class ClassWeightedPerDimGaussianMixture(nn.Module):
    def __init__(self, n_classes, n_mixes, n_dims, init_std=1e-1, **mix_dist_kwargs):
        super().__init__()
        self.n_classes = n_classes
        self.n_mixes = n_mixes
        self.n_dims = n_dims
        self.mix_dist = NClassIndependentDist(
            n_classes=n_mixes, n_dims=n_dims, **mix_dist_kwargs
        )
        self.mix_dist.class_means.data.normal_(mean=0, std=init_std)
        self.mix_dist.class_log_stds.data.normal_(mean=0, std=init_std)
        self.weights = nn.Parameter(
            th.zeros(
                n_classes,
                n_mixes,
            )
        )

    def forward(self, z, fixed=None):
        fixed = fixed or {}
        mixture_reduce = fixed.get("mixture_reduce", "sum")
        sum_dims = fixed.get("sum_dims", True)
        lp_per_mixture = self.mix_dist.log_probs_per_class(z, sum_dims=False)
        # examples x mixtures x dims
        log_weights = th.log_softmax(self.weights, dim=1)
        log_weights = log_weights.unsqueeze(-1)
        # classes x mixtures x 1
        # add:
        # examples x 1       x mixtures x dims
        # 1        x classes x mixtures x 1
        lp_weighted = lp_per_mixture.unsqueeze(1) + log_weights.unsqueeze(0)
        # examples x classes x mixtures x dims

        # npw compute mix across dims
        if mixture_reduce == "sum":
            lp_weighted = th.logsumexp(lp_weighted, dim=2)
        elif mixture_reduce == "max":
            lp_weighted = th.max(lp_weighted, dim=2)[0]
        else:
            assert mixture_reduce == "none"
        if sum_dims:
            lp_weighted = lp_weighted.sum(dim=-1)
        return z, lp_weighted

    def invert(self, z, fixed=None):
        fixed = fixed or {}
        if z is None:
            assert "n_samples" in fixed or "y" in fixed

            if "y" not in fixed:
                n_classes = self.class_weights.shape[0]
                n_samples = fixed["n_samples"]
                y = th.multinomial(
                    th.ones(n_classes) / n_classes, n_samples, replacement=True
                )
            else:
                y = fixed["y"]
            softmaxed_weights = th.softmax(self.weights, dim=1)
            # class x components

            desired_weights = softmaxed_weights[y]
            # examples x components

            desired_components = th.multinomial(
                desired_weights, num_samples=2688, replacement=True
            )
            # examples x dims

            desired_means = self.mix_dist.class_means.gather(
                dim=0, index=desired_components
            )
            # examples x dims
            desired_stds = th.exp(
                self.mix_dist.class_log_stds.gather(dim=0, index=desired_components)
            )
            # examples x dims

            randn_sample = th.randn_like(desired_means)

            z = randn_sample * desired_stds + desired_means
        logdet = self.forward(z, fixed=fixed)[1]
        return z, logdet


class ClassWeightedHierarchicalGaussianMixture(nn.Module):
    def __init__(
        self,
        n_classes,
        n_overall_mixes,
        n_dim_mixes,
        n_dims,
        reduce_per_dim,
        reduce_overall_mix,
        init_weight_std,
        init_mean_std,
        init_std_std,
        **mix_dist_kwargs
    ):
        super().__init__()
        self.n_classes = n_classes
        self.n_dim_mixes = n_dim_mixes
        self.n_dims = n_dims

        self.mix_dist = NClassIndependentDist(
            n_classes=n_dim_mixes, n_dims=n_dims, **mix_dist_kwargs
        )
        self.mix_dist.class_means.data.normal_(mean=0, std=init_mean_std)
        self.mix_dist.class_log_stds.data.normal_(mean=0, std=init_std_std)
        self.weights_per_dim = nn.Parameter(
            th.zeros(n_overall_mixes, n_dim_mixes, n_dims)
        )
        self.weights_per_dim.data.normal_(mean=0, std=init_weight_std)
        self.class_weights = nn.Parameter(th.zeros(n_classes, n_overall_mixes))
        # self.class_weights.data.normal(mean=0, std=init_std)
        self.reduce_per_dim = reduce_per_dim
        self.reduce_overall_mix = reduce_overall_mix

    def forward(self, z, fixed=None):
        fixed = fixed or {}
        sum_dims = fixed.get("sum_dims", True)

        lp_per_mixture = self.mix_dist.log_probs_per_class(z, sum_dims=False)
        # examples x mix_per_dims x dims

        # now we add:
        # examples x 1             x mix_per_dims x dims
        # 1        x overall_mixes x mix_per_dims x dims
        log_weights_per_dim = th.log_softmax(self.weights_per_dim, dim=1)
        weighted_per_mix = lp_per_mixture.unsqueeze(1) + log_weights_per_dim.unsqueeze(
            0
        )
        reduce_per_dim = fixed.get("reduce_per_dim", self.reduce_per_dim)
        if reduce_per_dim == "logsumexp":
            # compute across per-dim-components
            reduced_per_mix = th.logsumexp(weighted_per_mix, dim=2)
        elif reduce_per_dim == "max":
            # compute across per-dim-components
            reduced_per_mix = th.max(weighted_per_mix, dim=2)[0]
        elif reduce_per_dim == "gumbel_softmax":
            hard_max = F.gumbel_softmax(weighted_per_mix, dim=2, hard=True)
            reduced_per_mix = th.sum(weighted_per_mix * hard_max, dim=2)
        else:
            assert reduce_per_dim == "none"
            reduced_per_mix = weighted_per_mix
        # now examples x overall_mixes x dims

        if sum_dims:
            reduced_per_mix = th.sum(reduced_per_mix, dim=-1)
            # examples x overall_mixes

        log_class_weights = th.log_softmax(self.class_weights, dim=1)
        if not sum_dims:
            # need to also remember we will sum all log class weights later
            # so we just add average to each dim
            log_class_weights = (
                log_class_weights.unsqueeze(-1) / lp_per_mixture.shape[-1]
            )
        if reduce_per_dim == "none":
            # TODO: what do do here? seems unclear
            log_class_weights = log_class_weights.unsqueeze(-1)
            warnings.warn(
                "Results will be incorrect. we need to add class weights later to be correct",
            )
        # reduced_per_mix is examples x overall mixes (x mix per dims) x(dims)
        # so we add in basic case:
        # examples x 1       x overall_mixes
        # 1        x classes x overall_mixes
        weighted_per_class = reduced_per_mix.unsqueeze(1) + log_class_weights.unsqueeze(
            0
        )

        reduce_overall_mix = fixed.get("reduce_overall_mix", self.reduce_overall_mix)
        if reduce_overall_mix == "logsumexp":
            reduced_per_class = th.logsumexp(weighted_per_class, dim=2)
        elif reduce_overall_mix == "max":
            reduced_per_class = th.max(weighted_per_class, dim=2)[0]
        elif reduce_overall_mix == "gumbel_softmax":
            hard_max = F.gumbel_softmax(weighted_per_class, dim=2, hard=True)
            reduced_per_class = th.sum(weighted_per_class * hard_max, dim=2)
        else:
            assert reduce_overall_mix == "none"
            reduced_per_class = weighted_per_class
        return z, reduced_per_class

    def invert(self, z, fixed=None):
        fixed = fixed or {}
        if z is None:
            n_classes = self.class_weights.shape[0]
            assert "n_samples" in fixed or "y" in fixed

            if "y" not in fixed:
                n_samples = fixed["n_samples"]
                y = th.multinomial(
                    th.ones(n_classes) / n_classes, n_samples, replacement=True
                )
            else:
                y = fixed["y"]
                n_samples = len(y)

            softmaxed_class_weights = th.softmax(self.class_weights, dim=1)
            # classes x components

            desired_weights = softmaxed_class_weights[y]
            # examples x components
            component_selection = fixed.get("component_selection", "multinomial")
            if component_selection == "multinomial":
                desired_components = th.multinomial(
                    desired_weights, num_samples=1
                ).squeeze(1)
            else:
                assert component_selection == "max"
                desired_components = th.max(desired_weights, dim=1)[1]

            # examples

            softmaxed_dim_weights = th.softmax(self.weights_per_dim, dim=1)
            # components x dim-components x dims
            desired_dim_weights = softmaxed_dim_weights[desired_components]
            # examples x dim-components x dims

            if component_selection == "multinomial":
                # have to move to 2d first then back to 3d
                desired_dim_mixes = th.multinomial(
                    desired_dim_weights.permute(0, 2, 1).reshape(
                        -1, desired_dim_weights.shape[1]
                    ),
                    num_samples=1,
                ).squeeze(1)
                # (examples * dims)
                desired_dim_mixes = desired_dim_mixes.reshape(
                    desired_dim_weights.shape[0], desired_dim_weights.shape[2]
                )
                # examples x dims
            else:
                assert component_selection == "max"
                desired_dim_mixes = th.max(desired_dim_weights, dim=1)[1]

            # equal to: b = th.cat([dist.mix_dist.class_means.gather(dim=0, index=desired_dim_mixes[i:i+1]) for i in range(len(desired_dim_mixes))])
            desired_means = self.mix_dist.class_means.gather(
                dim=0, index=desired_dim_mixes
            )
            # examples x dims
            desired_stds = th.exp(
                self.mix_dist.class_log_stds.gather(dim=0, index=desired_dim_mixes)
            )
            # examples x dims

            randn_sample = th.randn_like(desired_means)

            z = randn_sample * desired_stds + desired_means
        logdet = self.forward(z, fixed=fixed)[1]
        return z, logdet


ClassWeightedHierachicalGaussianMixture = ClassWeightedHierarchicalGaussianMixture


class PerClassHierarchical(nn.Module):
    def __init__(
        self,
        n_classes,
        n_overall_mixes,
        n_dim_mixes,
        n_dims,
        reduce_per_dim,
        reduce_overall_mix,
        init_weight_std,
        init_mean_std,
        init_std_std,
        **mix_dist_kwargs
    ):
        super().__init__()
        self.dists_per_class = nn.ModuleList(
            [
                ClassWeightedHierarchicalGaussianMixture(
                    1,
                    n_overall_mixes=n_overall_mixes,
                    n_dim_mixes=n_dim_mixes,
                    n_dims=n_dims,
                    reduce_per_dim=reduce_per_dim,
                    reduce_overall_mix=reduce_overall_mix,
                    init_weight_std=init_weight_std,
                    init_mean_std=init_mean_std,
                    init_std_std=init_std_std,
                    **mix_dist_kwargs
                )
                for _ in range(n_classes)
            ]
        )

    def forward(self, z, fixed=None):
        lps = th.cat(
            [d.forward(z, fixed=fixed)[1] for d in self.dists_per_class], dim=1
        )
        return z, lps

    def invert(self, z, fixed=None):
        fixed = fixed or {}
        if z is None:
            assert "n_samples" in fixed or "y" in fixed
            n_classes = len(self.dists_per_class)
            if "y" not in fixed:
                n_samples = fixed["n_samples"]
                y = th.multinomial(
                    th.ones(n_classes) / n_classes, n_samples, replacement=True
                ).to(self.dists_per_class.weights_per_dim.device)
            else:
                y = fixed["y"]

            z = th.zeros(
                len(y),
                self.dists_per_class[0].weights_per_dim.shape[-1],
                device=y.device,
            )
            for i_class in range(n_classes):
                this_z = self.dists_per_class[i_class].invert(
                    None,
                    fixed=dict(
                        y=y[y == i_class] * 0,
                        **{k: fixed[k] for k in fixed if k != "y"}
                    ),
                )[0]
                z[y == i_class] = this_z
        logdet = self.forward(z, fixed=fixed)[1]
        return z, logdet


class PerClassDists(nn.Module):
    def __init__(
        self,
        dists_per_class,
    ):
        super().__init__()
        self.dists_per_class = nn.ModuleList(dists_per_class)

    def forward(self, z, fixed=None):
        lps = th.cat(
            [d.forward(z, fixed=fixed)[1] for d in self.dists_per_class], dim=1
        )
        return z, lps

    def invert(self, z, fixed=None):
        fixed = fixed or {}
        if z is None:
            assert "n_samples" in fixed or "y" in fixed
            n_classes = len(self.dists_per_class)
            if "y" not in fixed:
                n_samples = fixed["n_samples"]
                y = th.multinomial(
                    th.ones(n_classes) / n_classes, n_samples, replacement=True
                ).to(self.dists_per_class.weights_per_dim.device)
            else:
                y = fixed["y"]

            z = th.zeros(
                len(y),
                self.dists_per_class[0].weights_per_dim.shape[-1],
                device=y.device,
            )
            for i_class in range(n_classes):
                this_z = self.dists_per_class[i_class].invert(
                    None,
                    fixed=dict(
                        y=y[y == i_class] * 0,
                        **{k: fixed[k] for k in fixed if k != "y"}
                    ),
                )[0]
                z[y == i_class] = this_z
        logdet = self.forward(z, fixed=fixed)[1]
        return z, logdet


class SomeClassIndependentDimsDist(nn.Module):
    def __init__(
        self,
        n_classes,
        n_dims,
        n_class_independent_dims,
        optimize_std,
        truncate_to=None,
    ):
        super().__init__()
        n_shared_dims = n_dims - n_class_independent_dims
        self.shared_mean = nn.Parameter(th.zeros(n_shared_dims, requires_grad=True))
        self.class_means = nn.Parameter(
            th.zeros(n_classes, n_class_independent_dims, requires_grad=True)
        )
        if optimize_std:
            self.shared_log_std = nn.Parameter(th.zeros(n_shared_dims, requires_grad=True))
            self.class_log_stds = nn.Parameter(
                th.zeros(n_classes, n_class_independent_dims, requires_grad=True)
            )
        else:
            self.register_buffer(
                "shared_log_std",
                th.zeros(
                    n_shared_dims,
                ),
            )
            self.register_buffer(
                "class_log_stds",
                th.zeros(
                    n_classes,
                    n_class_independent_dims,
                ),
            )
        randperm = th.randperm(n_dims)
        shared_indices = th.sort(randperm[n_class_independent_dims:])[0]
        independent_indices = th.sort(randperm[:n_class_independent_dims])[0]
        self.register_buffer("shared_indices", shared_indices)
        self.register_buffer("independent_indices", independent_indices)
        self.truncate_to = truncate_to

    def forward(self, x, fixed=None):
        dist = self.create_n_class_dist()
        return dist.forward(x, fixed=fixed)

    def invert(self, x, fixed=None):
        dist = self.create_n_class_dist()
        return dist.invert(x, fixed=fixed)

    def create_n_class_dist(self):
        n_dims = self.shared_mean.shape[0] + self.class_means.shape[1]
        n_classes = len(self.class_means)
        merged_means = th.zeros(
            len(self.class_means), n_dims, device=self.class_means.device
        )
        merged_means = th.scatter(
            merged_means,
            1,
            self.shared_indices.unsqueeze(0).repeat(n_classes, 1),
            self.shared_mean.unsqueeze(0).repeat(n_classes, 1),
        )
        merged_means = th.scatter(
            merged_means,
            1,
            self.independent_indices.unsqueeze(0).repeat(n_classes, 1),
            self.class_means,
        )
        merged_log_stds = th.zeros(
            len(self.class_means), n_dims, device=self.class_means.device
        )
        merged_log_stds = th.scatter(
            merged_log_stds,
            1,
            self.shared_indices.unsqueeze(0).repeat(n_classes, 1),
            self.shared_log_std.unsqueeze(0).repeat(n_classes, 1),
        )
        merged_log_stds = th.scatter(
            merged_log_stds,
            1,
            self.independent_indices.unsqueeze(0).repeat(n_classes, 1),
            self.class_log_stds,
        )
        dist = NClassIndependentDist(
            means=merged_means, log_stds=merged_log_stds, truncate_to=self.truncate_to
        )
        return dist

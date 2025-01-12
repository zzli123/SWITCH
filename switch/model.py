import collections
from abc import abstractmethod
from typing import Optional, Tuple

import torch
import numpy as np
import torch.distributions as D
import torch.nn.functional as F
from torch.autograd import Function
from torch_geometric.nn import GCNConv
from .gat_conv import GATConv
from torch.optim.lr_scheduler import LambdaLR, StepLR
from torch.distributions.utils import broadcast_all, lazy_property, logits_to_probs
from torch.distributions import Distribution, Gamma, constraints
from torch.distributions import Poisson as PoissonTorch

EPS = 1e-7

# Cite from https://github.com/gao-lab/GLUE
class GraphConv(torch.nn.Module):

    def forward(
            self, input: torch.Tensor, eidx: torch.Tensor,
            enorm: torch.Tensor, esgn: torch.Tensor
    ) -> torch.Tensor:
        r"""
        Forward propagation

        Parameters
        ----------
        input
            Input data (:math:`n_{vertices} \times n_{features}`)
        eidx
            Vertex indices of edges (:math:`2 \times n_{edges}`)
        enorm
            Normalized weight of edges (:math:`n_{edges}`)
        esgn
            Sign of edges (:math:`n_{edges}`)

        Returns
        -------
        result
            Graph convolution result (:math:`n_{vertices} \times n_{features}`)
        """
        sidx, tidx = eidx  # source index and target index
        message = input[sidx] * (esgn * enorm).unsqueeze(1)  # n_edges * n_features
        res = torch.zeros_like(input)
        tidx = tidx.unsqueeze(1).expand_as(message)  # n_edges * n_features
        res.scatter_add_(0, tidx, message)
        return res

# Cite from https://github.com/gao-lab/GLUE
class GraphEncoder(torch.nn.Module):

    r"""
    Graph encoder

    Parameters
    ----------
    vnum
        Number of vertices
    out_features
        Output dimensionality
    """

    def __init__(
            self, vnum: int, out_features: int
    ) -> None:
        super().__init__()
        self.vrepr = torch.nn.Parameter(torch.zeros(vnum, out_features))
        self.conv = GraphConv()
        self.loc = torch.nn.Linear(out_features, out_features)
        self.std_lin = torch.nn.Linear(out_features, out_features)

    def forward(
            self, eidx: torch.Tensor, enorm: torch.Tensor, esgn: torch.Tensor
    ) -> D.Normal:
        ptr = self.conv(self.vrepr, eidx, enorm, esgn)
        loc = self.loc(ptr)
        std = F.softplus(self.std_lin(ptr)) + EPS
        return D.Normal(loc, std)

# Cite from https://github.com/gao-lab/GLUE
class GraphDecoder(torch.nn.Module):

    r"""
    Graph decoder
    """

    def forward(
            self, v: torch.Tensor, eidx: torch.Tensor, esgn: torch.Tensor
    ) -> D.Bernoulli:
        sidx, tidx = eidx  # Source index and target index
        logits = esgn * (v[sidx] * v[tidx]).sum(dim=1)
        return D.Bernoulli(logits=logits)

class DataEncoder(torch.nn.Module):
    def __init__(self, in_features, out_features, conv, h_depth=2, h_dim=256, dropout=0.2, ):
        super().__init__()

        assert conv in ["GCN","GAT","LIN"]
        self.h_depth = h_depth
        self.conv_layers = torch.nn.ModuleList()
        self.act_layers = torch.nn.ModuleList()
        self.bn_layers = torch.nn.ModuleList()
        self.dropout_layers = torch.nn.ModuleList()
        self.conv = conv

        ptr_dim = in_features

        for layer in range(self.h_depth):
            if(conv=="GCN"):
                self.conv_layers.append(GCNConv(ptr_dim, h_dim, bias=False))
            elif(conv=="GAT"):
                self.conv_layers.append(GATConv(ptr_dim, h_dim, add_self_loops=False, bias=False, 
                                                concat=False))
            elif(conv=="LIN"):
                self.conv_layers.append(torch.nn.Linear(ptr_dim, h_dim))
            self.act_layers.append(torch.nn.LeakyReLU(negative_slope=0.2))
            self.bn_layers.append(torch.nn.BatchNorm1d(h_dim))
            self.dropout_layers.append(torch.nn.Dropout(p=dropout))
            ptr_dim = h_dim
        
        self.loc = torch.nn.Linear(ptr_dim, out_features)
        self.std_lin = torch.nn.Linear(ptr_dim, out_features)

        self.TOTAL_COUNT = 1e4

    def compute_l(self, x: torch.Tensor) -> torch.Tensor:
        return x.sum(dim=1, keepdim=True)

    def normalize(
            self, x: torch.Tensor, l: torch.Tensor
    ) -> torch.Tensor:
        return (x * (self.TOTAL_COUNT / l)).log1p()
    
    def clr_normalize(self, x: torch.Tensor):
        l = torch.log1p(x).sum(dim=1, keepdim=True)
        exp = torch.exp(l / x.size(1)) + EPS
        clr_x = torch.log1p(x / exp.view(-1, 1))

        return clr_x, x.sum(dim=1, keepdim=True)
    
    def forward(self, x, edge_index, minibatch=False, batch_size=None, normalize="log"):
        
        if(normalize=="clr"):# for protein
            ptr, l = self.clr_normalize(x)
        elif(normalize=="log"):
                l = self.compute_l(x)
                ptr = self.normalize(x, l)
        else:
            raise ValueError("Invalid normalize method")
                
        if(minibatch):
            if(len(edge_index)!=len(self.conv_layers)):
                raise ValueError(f"Length of sample size must be equal to `h_deepth`.")
            
            for i, (edge_idx, _, size) in enumerate(edge_index):
                ptr_target = ptr[:size[1]]
                ptr = self.conv_layers[i]((ptr, ptr_target), edge_idx)
                ptr = self.bn_layers[i](ptr)   
                ptr = self.act_layers[i](ptr)
                ptr = self.dropout_layers[i](ptr) 
        else:
            for layer in range(0, self.h_depth):
                if(self.conv in ["GAT","GCN"]):
                    ptr = self.conv_layers[layer](ptr, edge_index)
                else:
                    ptr = self.conv_layers[layer](ptr)
                ptr = self.bn_layers[layer](ptr)   
                ptr = self.act_layers[layer](ptr)
                ptr = self.dropout_layers[layer](ptr)
        
        # print(torch.any(torch.isnan(ptr)))
        

        loc = self.loc(ptr, )
        std = F.softplus(self.std_lin(ptr, )) + EPS
        if(batch_size!=None):
            return D.Normal(loc, std), l[:batch_size]
        else:
            return D.Normal(loc, std), l

class NBDataDecoder(torch.nn.Module):
    def __init__(self, n_features, n_batches=1,):
        super().__init__()
        
        self.scale_lin = torch.nn.Parameter(torch.nn.init.xavier_uniform_(torch.zeros(n_batches, n_features)))
        self.bias = torch.nn.Parameter(torch.nn.init.xavier_uniform_(torch.zeros(n_batches, n_features)))
        self.log_theta = torch.nn.Parameter(torch.nn.init.xavier_uniform_(torch.zeros(n_batches, n_features)))

    def forward(self, u: torch.Tensor, v: torch.Tensor, b: torch.Tensor, l: torch.Tensor, edge_index: torch.Tensor = None):

        scale = F.softplus(self.scale_lin[b])
        logit_mu = scale * (u @ v.t()) + self.bias[b]
        mu = F.softmax(logit_mu, dim=1)  * l
        log_theta = self.log_theta[b]
        feature_dist = D.NegativeBinomial(
            log_theta.exp(),
            logits=(mu + EPS).log() - log_theta
        )
        adj_rec = None

        return (feature_dist, adj_rec)

class NormalDataDecoder(torch.nn.Module):

    def __init__(self, out_features: int, n_batches: int = 1) -> None:
        super().__init__()
        self.scale_lin = torch.nn.Parameter(torch.zeros(n_batches, out_features))
        self.bias = torch.nn.Parameter(torch.zeros(n_batches, out_features))
        self.std_lin = torch.nn.Parameter(torch.zeros(n_batches, out_features))

    def forward(
            self, u: torch.Tensor, v: torch.Tensor, b: torch.Tensor, l: Optional[torch.Tensor], edge_index: torch.Tensor = None
    ) -> D.Normal:
        scale = F.softplus(self.scale_lin[b])
        loc = scale * (u @ v.t()) + self.bias[b]
        std = F.softplus(self.std_lin[b]) + EPS

        adj_rec = None
        return (D.Normal(loc, std), adj_rec)

class BerDataDecoder(torch.nn.Module):
    def __init__(self, n_features, n_batches=1):
        super().__init__()
        
        self.scale_lin = torch.nn.Parameter(torch.nn.init.xavier_uniform_(torch.zeros(n_batches, n_features)))
        self.bias = torch.nn.Parameter(torch.nn.init.xavier_uniform_(torch.zeros(n_batches, n_features)))

    def forward(self, u: torch.Tensor, v: torch.Tensor, b: torch.Tensor, l: torch.Tensor, edge_index: torch.Tensor = None):

        scale = F.softplus(self.scale_lin[b])
        logits = scale * (u @ v.t())  + self.bias[b]

        adj_rec = None

        return (D.Bernoulli(logits = logits), adj_rec)

class PoisDataDecoder(torch.nn.Module):
    def __init__(self, n_features, n_batches=1, h_dim=50):
        super().__init__()
        
        self.scale_lin = torch.nn.Parameter(torch.zeros(n_batches, n_features))
        self.bias = torch.nn.Parameter(torch.zeros(n_batches, n_features))

    def forward(self, u: torch.Tensor, v: torch.Tensor, b: torch.Tensor, l: torch.Tensor, edge_index: torch.Tensor = None):

        scale = F.softplus(self.scale_lin[b])
        rate = scale * (u @ v.t()) + self.bias[b]
        rate = F.softmax(rate, dim=1)  * l
        
        feature_dist = D.Poisson(
            rate=rate
        )
        adj_rec = None

        return (feature_dist, adj_rec)

# modified from https://github.com/gao-lab/GLUE
class ZINB(D.NegativeBinomial):

    def __init__(
            self, zi_logits: torch.Tensor,
            total_count: torch.Tensor, logits: torch.Tensor = None
    ) -> None:
        super().__init__(total_count, logits=logits)
        self.zi_logits = zi_logits

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        raw_log_prob = super().log_prob(value)
        zi_log_prob = torch.empty_like(raw_log_prob)
        z_mask = value.abs() < EPS
        z_zi_logits, nz_zi_logits = self.zi_logits[z_mask], self.zi_logits[~z_mask]
        zi_log_prob[z_mask] = (
            raw_log_prob[z_mask].exp() + z_zi_logits.exp() + EPS
        ).log() - F.softplus(z_zi_logits)
        zi_log_prob[~z_mask] = raw_log_prob[~z_mask] - F.softplus(nz_zi_logits)
        return zi_log_prob

class ZINBDataDecoder(NBDataDecoder):

    def __init__(self, out_features: int, n_batches: int = 1) -> None:
        super().__init__(out_features, n_batches=n_batches)
        self.zi_logits = torch.nn.Parameter(torch.zeros(n_batches, out_features))

    def forward(self, u: torch.Tensor, v: torch.Tensor, b: torch.Tensor, l: torch.Tensor, edge_index=None):
        scale = F.softplus(self.scale_lin[b])
        logit_mu = scale * (u @ v.t()) + self.bias[b]
        mu = F.softmax(logit_mu, dim=1) * l
        log_theta = self.log_theta[b]
        feature_dist = ZINB(
            self.zi_logits[b].expand_as(mu),
            log_theta.exp(),
            logits=(mu + EPS).log() - log_theta
        )
        adj_rec = None

        return (feature_dist, adj_rec)

# modified from https://github.com/scverse/scvi-tools
class NegativeBinomialMixture(Distribution):

    def __init__(
        self,
        mu1: torch.Tensor,
        mu2: torch.Tensor,
        theta1: torch.Tensor,
        mixture_logits: torch.Tensor,
        validate_args: bool = False,
    ):
        (
            self.mu1,
            self.theta1,
            self.mu2,
            self.mixture_logits,
        ) = broadcast_all(mu1, theta1, mu2, mixture_logits)
        self.on_mps = (
            mu1.device.type == "mps"
        )  # TODO: This is used until torch will solve the MPS issues
        super().__init__(validate_args=validate_args)

    @property
    def mean(self) -> torch.Tensor:
        pi = self.mixture_probs
        return pi * self.mu1 + (1 - pi) * self.mu2

    @lazy_property
    def mixture_probs(self) -> torch.Tensor:
        return logits_to_probs(self.mixture_logits, is_binary=True)

    def _gamma(self, theta: torch.Tensor, mu: torch.Tensor, on_mps: bool = False) -> Gamma:
        concentration = theta
        rate = theta / mu
        # Important remark: Gamma is parametrized by the rate = 1/scale!
        gamma_d = (
            Gamma(concentration=concentration.to("cpu"), rate=rate.to("cpu"))
            if on_mps  # TODO: NEED TORCH MPS FIX for 'aten::_standard_gamma'
            else Gamma(concentration=concentration, rate=rate)
        )
        return gamma_d
    
    def torch_lgamma_mps(self, x: torch.Tensor) -> torch.Tensor:
        """Used in mac Mx devices while broadcasting a tensor

        Parameters
        ----------
        x
            Data

        Returns
        -------
        lgamma tensor that perform on a copied version of the tensor
        """
        return torch.lgamma(x.contiguous())

    @torch.inference_mode()
    def sample(
        self,
        sample_shape: torch.Size = None,
    ) -> torch.Tensor:
        """Sample from the distribution."""
        sample_shape = sample_shape or torch.Size()
        pi = self.mixture_probs
        mixing_sample = D.Bernoulli(pi).sample()
        mu = self.mu1 * mixing_sample + self.mu2 * (1 - mixing_sample)
        if self.theta2 is None:
            theta = self.theta1
        else:
            theta = self.theta1 * mixing_sample + self.theta2 * (1 - mixing_sample)
        gamma_d = self._gamma(theta, mu, self.on_mps)  # TODO: TORCH MPS FIX - DONE ON CPU CURRENTLY
        p_means = gamma_d.sample(sample_shape)

        # Clamping as distributions objects can have buggy behaviors when
        # their parameters are too high
        l_train = torch.clamp(p_means, max=1e8)
        counts = PoissonTorch(l_train).sample()  # Shape : (n_samples, n_cells_batch, n_features)
        return counts

    def log_mixture_nb(
        self,
        x: torch.Tensor,
        mu_1: torch.Tensor,
        mu_2: torch.Tensor,
        theta_1: torch.Tensor,
        pi_logits: torch.Tensor,
        eps: float = 1e-8,
        log_fn: callable = torch.log,
        lgamma_fn: callable = torch.lgamma,
    ) -> torch.Tensor:
        """Log likelihood (scalar) of a minibatch according to a mixture nb model.

        pi_logits is the probability (logits) to be in the first component.
        For totalVI, the first component should be background.

        Parameters
        ----------
        x
            Observed data
        mu_1
            Mean of the first negative binomial component (has to be positive support) (shape:
            minibatch x features)
        mu_2
            Mean of the second negative binomial (has to be positive support) (shape: minibatch x
            features)
        theta_1
            First inverse dispersion parameter (has to be positive support) (shape: minibatch x
            features)
        theta_2
            Second inverse dispersion parameter (has to be positive support) (shape: minibatch x
            features). If None, assume one shared inverse dispersion parameter.
        pi_logits
            Probability of belonging to mixture component 1 (logits scale)
        eps
            Numerical stability constant
        log_fn
            log function
        lgamma_fn
            log gamma function
        """
        log = log_fn
        lgamma = lgamma_fn
        theta = theta_1
        if theta.ndimension() == 1:
            theta = theta.view(1, theta.size(0))  # In this case, we reshape theta for broadcasting

        log_theta_mu_1_eps = log(theta + mu_1 + eps)
        log_theta_mu_2_eps = log(theta + mu_2 + eps)
        lgamma_x_theta = lgamma(x + theta)
        lgamma_theta = lgamma(theta)
        lgamma_x_plus_1 = lgamma(x + 1)

        log_nb_1 = (
            theta * (log(theta + eps) - log_theta_mu_1_eps)
            + x * (log(mu_1 + eps) - log_theta_mu_1_eps)
            + lgamma_x_theta
            - lgamma_theta
            - lgamma_x_plus_1
        )
        log_nb_2 = (
            theta * (log(theta + eps) - log_theta_mu_2_eps)
            + x * (log(mu_2 + eps) - log_theta_mu_2_eps)
            + lgamma_x_theta
            - lgamma_theta
            - lgamma_x_plus_1
        )

        logsumexp = torch.logsumexp(torch.stack((log_nb_1, log_nb_2 - pi_logits)), dim=0)
        softplus_pi = F.softplus(-pi_logits)

        log_mixture_nb_res = logsumexp - softplus_pi

        return log_mixture_nb_res

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """Log probability."""

        lgamma_fn = self.torch_lgamma_mps if self.on_mps else torch.lgamma  # TODO: TORCH MPS FIX
        return self.log_mixture_nb(
            value,
            self.mu1,
            self.mu2,
            self.theta1,
            self.mixture_logits,
            eps=1e-08,
            lgamma_fn=lgamma_fn,
        )

    def __repr__(self) -> str:
        param_names = [k for k, _ in self.arg_constraints.items() if k in self.__dict__]
        args_string = ", ".join(
            [
                f"{p}: "
                f"{self.__dict__[p] if self.__dict__[p].numel() == 1 else self.__dict__[p].size()}"
                for p in param_names
                if self.__dict__[p] is not None
            ]
        )
        return self.__class__.__name__ + "(" + args_string + ")"

class MixtureNBDecoder(torch.nn.Module):
    def __init__(self, n_features, n_batches=1,):
        super().__init__()
        
        self.back_scale_lin = torch.nn.Parameter(torch.nn.init.xavier_uniform_(torch.zeros(n_batches, n_features)))
        self.fore_alpha = torch.nn.Parameter(torch.nn.init.xavier_uniform_(torch.zeros(n_batches, n_features)))
        # self.bias = torch.nn.Parameter(torch.nn.init.xavier_uniform_(torch.zeros(n_batches, n_features)))
        self.log_theta = torch.nn.Parameter(torch.nn.init.xavier_uniform_(torch.zeros(n_batches, n_features)))
        self.mix_prob = torch.nn.Parameter(torch.nn.init.xavier_uniform_(torch.zeros(n_batches, n_features)))
    
    def forward(self, u: torch.Tensor, v: torch.Tensor, b: torch.Tensor, l: torch.Tensor, edge_index: torch.Tensor = None):
        x = u @ v.t()
        back_scale = F.softplus(self.back_scale_lin[b])
        back_logit_mu = back_scale * x # + self.bias[b]
        back_mu = F.softmax(back_logit_mu, dim=1)
        alpha = F.softplus(self.fore_alpha[b])
        alpha = F.relu(alpha * x) + 1 + EPS
        log_theta = self.log_theta[b]
        mix_prob = F.sigmoid(self.mix_prob * x)
        feature_dist = NegativeBinomialMixture(
           mu1 = back_mu,
           mu2 = alpha * back_mu,
           theta1 = log_theta.exp(),
           mixture_logits = mix_prob
        )
        adj_rec = None

        return (feature_dist, adj_rec)

class Discriminator(torch.nn.Sequential):

    r"""
    Modality discriminator

    Parameters
    ----------
    in_features
        Input dimensionality
    out_features
        Output dimensionality
    h_depth
        Hidden layer depth
    h_dim
        Hidden layer dimensionality
    dropout
        Dropout rate
    """

    def __init__(
            self, in_features: int, out_features: int, n_batches: int = 0,
            h_depth: int = 2, h_dim: Optional[int] = 256,
            dropout: float = 0.2
    ) -> None:
        self.n_batches = n_batches
        od = collections.OrderedDict()
        ptr_dim = in_features + self.n_batches
        for layer in range(h_depth):
            od[f"linear_{layer}"] = torch.nn.Linear(ptr_dim, h_dim)
            od[f"bn_layer"] = torch.nn.BatchNorm1d(h_dim)
            od[f"act_{layer}"] = torch.nn.LeakyReLU(negative_slope=0.2)
            od[f"dropout_{layer}"] = torch.nn.Dropout(p=dropout)
            ptr_dim = h_dim
        od["pred"] = torch.nn.Linear(ptr_dim, out_features)
        super().__init__(od)

    def forward(self, x: torch.Tensor, b: torch.Tensor) -> torch.Tensor:  # pylint: disable=arguments-differ
        # x = ReverseLayerF.apply(x, alpha)
        if self.n_batches:
            b_one_hot = F.one_hot(b, num_classes=self.n_batches)
            x = torch.cat([x, b_one_hot], dim=1)
        return super().forward(x)

class Prior(torch.nn.Module):

    r"""
    Prior distribution

    Parameters
    ----------
    loc
        Mean of the normal distribution
    std
        Standard deviation of the normal distribution
    """

    def __init__(
            self, loc: float = 0.0, std: float = 1.0
    ) -> None:
        super().__init__()
        loc = torch.as_tensor(loc, dtype=torch.get_default_dtype())
        std = torch.as_tensor(std, dtype=torch.get_default_dtype())
        self.register_buffer("loc", loc)
        self.register_buffer("std", std)

    def forward(self) -> D.Normal:
        return D.Normal(self.loc, self.std)

class EarlyStopping:
    def __init__(self, patience=200, gen_delta=2e-4, dsc_delta=1e-3, verbose=False, step=50):
    
        self.patience = patience
        self.gen_loss_history = []
        self.avg_loss_history = []
        self.gen_delta = gen_delta
        self.dsc_delta = dsc_delta
        self.verbose = verbose
        self.counter = 0
        self.best_gen_loss = np.inf
        self.best_dsc_loss =  0.693
        self.early_stop = False
        self.step = step

    def __call__(self, val_loss, dsc_loss):

        self.gen_loss_history.append(val_loss)
        self.counter += 1

        if(self.counter % self.step == 0):
            avg_loss = np.mean(self.gen_loss_history)
            self.avg_loss_history.append(avg_loss)
            self.gen_loss_history = []
        
        if(self.counter >= self.patience):
            if(self.best_gen_loss - np.min(self.avg_loss_history) > self.gen_delta):
                self.best_gen_loss = np.min(self.avg_loss_history)
                self.avg_loss_history = []
                self.counter = 0
            elif(np.abs(self.best_dsc_loss - dsc_loss) < self.dsc_delta):
                self.early_stop = True
                if self.verbose:
                    print("Early stopping triggered due to stable discriminator loss.")
            else:
                if self.verbose:
                    print("No improvement; continuing training...")

class WarmUpScheduler():
    def __init__(self, optimizer, warmup_epochs=500, base_lr=2e-4, target_lr=2e-3, step_size=50, gamma=0.9):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.base_lr = base_lr
        self.target_lr = target_lr
        self.step_size = step_size
        self.gamma = gamma
        self.flag = False

        self.decay_scheduler = StepLR(self.optimizer, step_size=self.step_size, gamma=self.gamma)

    def set_lr(self, lr):
        for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            lr = self.base_lr + (self.target_lr - self.base_lr) * (epoch / self.warmup_epochs)
            self.set_lr(lr)
    
        elif(not self.flag):
            self.decay_scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            if current_lr < self.base_lr:
                self.set_lr(self.base_lr)
                self.flag = True

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

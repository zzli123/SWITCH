import collections
from typing import Optional, Tuple

import torch
import numpy as np
import torch.distributions as D
import torch.nn.functional as F
# from torch_geometric.nn import GCNConv
from .gat_conv import GATConv
from torch.optim.lr_scheduler import StepLR

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
        self.vrepr = torch.nn.Parameter((torch.zeros(vnum, out_features)))
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
    """
    DataEncoder class for encoding input features using different convolution types.

    Parameters:
    ----------
    in_features : int
        Number of input features.

    out_features : int
        Number of output features.

    h_depth : int, optional (default=2)
        Number of hidden layers.

    h_dim : int, optional (default=256)
        Dimensionality of hidden layers.

    dropout : float, optional (default=0.2)
        Dropout rate to prevent overfitting.
    """
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 h_depth: int = 2,
                 h_dim: int = 256,
                 dropout: float = 0.2, 
    ):
        super().__init__()

        self.h_depth = h_depth
        self.TOTAL_COUNT = 1e4

    def compute_l(self, x: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the library size for a matrix, which is the sum of each row.

        Parameters:
        ----------
        x : torch.Tensor
            The input matrix (e.g., single-cell matrix) for which the library size is computed.

        Returns:
        -------
        torch.Tensor
            The library size for each row of the input matrix.
        """
        return x.sum(dim=1, keepdim=True)

    def normalize(
            self, x: torch.Tensor, l: torch.Tensor
    ) -> torch.Tensor:
        """
        Normalizes the input matrix using the computed library size.

        Parameters:
        ----------
        x : torch.Tensor
            The input matrix to be normalized.

        l : torch.Tensor
            The computed library size for each row of the matrix.

        Returns:
        -------
        torch.Tensor
            The normalized matrix.
        """
        return (x * (self.TOTAL_COUNT / l)).log1p()
        # return (x * (torch.median(x.sum(dim=1)) / l)).log1p()
    
    def forward(self,
                x: torch.Tensor,
                edge_index: torch.Tensor,
    ):
        """
        Forward pass of the encoder.

        Parameters:
        ----------
        x : torch.Tensor
            The input matrix.

        edge_index : torch.Tensor
            The graph edges, required when convolution type is 'GCN' or 'GAT'.
            
        Returns:
        -------
        torch.Tensor
            The output after applying the forward pass.
        """
        
        raise NotImplementedError

class GATEncoder(DataEncoder):
    """
    DataEncoder class for encoding input features using different convolution types.

    Parameters:
    ----------
    in_features : int
        Number of input features.

    out_features : int
        Number of output features.

    h_depth : int, optional (default=2)
        Number of hidden layers.

    h_dim : int, optional (default=256)
        Dimensionality of hidden layers.

    dropout : float, optional (default=0.2)
        Dropout rate to prevent overfitting.
    """
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 h_depth: int = 2,
                 h_dim: int = 256,
                 dropout: float = 0.2, 
    ):
        super().__init__(in_features, out_features, h_depth, h_dim, dropout)

        self.conv_layers = torch.nn.ModuleList()
        self.act_layers = torch.nn.ModuleList()
        self.bn_layers = torch.nn.ModuleList()
        self.dropout_layers = torch.nn.ModuleList()

        ptr_dim = in_features

        for _ in range(self.h_depth):
            self.conv_layers.append(GATConv(ptr_dim, h_dim, add_self_loops=False,
                                            bias=False, concat=False))
            self.act_layers.append(torch.nn.LeakyReLU(negative_slope=0.2))
            self.bn_layers.append(torch.nn.BatchNorm1d(h_dim))
            self.dropout_layers.append(torch.nn.Dropout(p=dropout))
            ptr_dim = h_dim
        
        self.loc = torch.nn.Linear(ptr_dim, out_features)
        self.std_lin = torch.nn.Linear(ptr_dim, out_features)

    
    def forward(self,
                x: torch.Tensor,
                edge_index: torch.Tensor,
    ):
        """
        Forward pass of the encoder.

        Parameters:
        ----------
        x : torch.Tensor
            The input matrix.

        edge_index : torch.Tensor
            The graph edges, required when convolution type is 'GCN' or 'GAT'.
            
        Returns:
        -------
        torch.Tensor
            The output after applying the forward pass.
        """
        
        l = self.compute_l(x)
        ptr = self.normalize(x, l)

        for layer in range(0, self.h_depth):
            ptr = self.conv_layers[layer](ptr, edge_index)
            # ptr = self.act_layers[layer](ptr)
            ptr = self.bn_layers[layer](ptr)   
            ptr = self.act_layers[layer](ptr)
            ptr = self.dropout_layers[layer](ptr)
        
        loc = self.loc(ptr, )
        std = F.softplus(self.std_lin(ptr, )) + EPS
        
        return D.Normal(loc, std), l

class LinearEncoder(DataEncoder):
    """
    DataEncoder class for encoding input features using different convolution types.

    Parameters:
    ----------
    in_features : int
        Number of input features.

    out_features : int
        Number of output features.

    conv : str
        Type of convolution layer ('GCN', 'GAT', or 'LIN').

    h_depth : int, optional (default=2)
        Number of hidden layers.

    h_dim : int, optional (default=256)
        Dimensionality of hidden layers.

    dropout : float, optional (default=0.2)
        Dropout rate to prevent overfitting.
    """
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 h_depth: int = 2,
                 h_dim: int = 256,
                 dropout: float = 0.2, 
    ):
        super().__init__(in_features, out_features, h_depth, h_dim, dropout)

        self.conv_layers = torch.nn.ModuleList()
        self.act_layers = torch.nn.ModuleList()
        self.bn_layers = torch.nn.ModuleList()
        self.dropout_layers = torch.nn.ModuleList()

        ptr_dim = in_features

        for _ in range(self.h_depth):
            self.conv_layers.append(torch.nn.Linear(ptr_dim, h_dim))
            self.act_layers.append(torch.nn.LeakyReLU(negative_slope=0.2))
            self.bn_layers.append(torch.nn.BatchNorm1d(h_dim))
            self.dropout_layers.append(torch.nn.Dropout(p=dropout))
            ptr_dim = h_dim
        
        self.loc = torch.nn.Linear(ptr_dim, out_features)
        self.std_lin = torch.nn.Linear(ptr_dim, out_features)
    
    def forward(self,
                x: torch.Tensor,
                edge_index: torch.Tensor, # NOT USED
    ):
        """
        Forward pass of the encoder.

        Parameters:
        ----------
        x : torch.Tensor
            The input matrix.

        edge_index : torch.Tensor
            The graph edges, required when convolution type is 'GCN' or 'GAT'.
            
        Returns:
        -------
        torch.Tensor
            The output after applying the forward pass.
        """

        l = self.compute_l(x)
        ptr = self.normalize(x, l)

        for layer in range(0, self.h_depth):
            ptr = self.conv_layers[layer](ptr)
            ptr = self.bn_layers[layer](ptr)   
            ptr = self.act_layers[layer](ptr)
            ptr = self.dropout_layers[layer](ptr)
        
        loc = self.loc(ptr, )
        std = F.softplus(self.std_lin(ptr, )) + EPS
        
        return D.Normal(loc, std), l

class MiniBatchEncoder(GATEncoder):
    """
    DataEncoder class for encoding input features using different convolution types.

    Parameters:
    ----------
    in_features : int
        Number of input features.

    out_features : int
        Number of output features.

    h_depth : int, optional (default=2)
        Number of hidden layers.

    h_dim : int, optional (default=256)
        Dimensionality of hidden layers.

    dropout : float, optional (default=0.2)
        Dropout rate to prevent overfitting.
    """
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 h_depth: int = 2,
                 h_dim: int = 256,
                 dropout: float = 0.2, 
    ):
        super().__init__(in_features, out_features, h_depth, h_dim, dropout)
    
    def forward(self,
                x: torch.Tensor,
                edge_index: torch.Tensor,
    ):
        """
        Forward pass of the encoder.

        Parameters:
        ----------
        x : torch.Tensor
            The input matrix.

        edge_index : torch.Tensor
            The graph edges, required when convolution type is 'GCN' or 'GAT'.
            
        Returns:
        -------
        torch.Tensor
            The output after applying the forward pass.
        """
        
        
        l = self.compute_l(x)
        ptr = self.normalize(x, l)

        # assert len(edge_index)==len(self.conv_layers)
            
        for i, (edge_idx, _, size) in enumerate(edge_index):
            ptr_target = ptr[:size[1]]
            ptr = self.conv_layers[i]((ptr, ptr_target), edge_idx)
            ptr = self.bn_layers[i](ptr)   
            ptr = self.act_layers[i](ptr)
            ptr = self.dropout_layers[i](ptr) 
        
        loc = self.loc(ptr, )
        std = F.softplus(self.std_lin(ptr, )) + EPS
        
        return D.Normal(loc, std), l

class NBDataDecoder(torch.nn.Module):
    """
    Decoder class that returns a negative binomial distribution.

    Parameters:
    ----------
    n_features : int
        The number of output features of the decoder.

    """
    def __init__(self, n_features):
        super().__init__()
        
        self.scale_lin = torch.nn.Parameter(torch.nn.init.xavier_uniform_(torch.zeros(1, n_features)))
        self.bias = torch.nn.Parameter(torch.nn.init.xavier_uniform_(torch.zeros(1, n_features)))
        self.log_theta = torch.nn.Parameter(torch.nn.init.xavier_uniform_(torch.zeros(1, n_features)))
        # self.scale_lin = torch.nn.Parameter(torch.nn.init.kaiming_uniform_(torch.zeros(1, n_features), a=0.2))
        # self.bias = torch.nn.Parameter((torch.randn(1, n_features)))
        # self.log_theta = torch.nn.Parameter(torch.nn.init.kaiming_uniform_(torch.zeros(1, n_features), a=0.2))

    def forward(self, u: torch.Tensor, v: torch.Tensor, l: torch.Tensor
    ):
        """
        Forward pass of the decoder.

        Parameters:
        ----------
        u : torch.Tensor
            The data embedding.

        v : torch.Tensor
            The feature embedding.

        l : torch.Tensor
            The library size of the data.

        Returns:
        -------
        torch.Tensor
          The negative binomial distribution.
        """

        scale = F.softplus(self.scale_lin[0])
        logit_mu = scale * (u @ v.t()) + self.bias[0]
        mu = F.softmax(logit_mu, dim=1)  * l
        log_theta = self.log_theta[0]
        feature_dist = D.NegativeBinomial(
            log_theta.exp(),
            logits=(mu + EPS).log() - log_theta
        )

        return feature_dist

class NormalDataDecoder(torch.nn.Module):
    """
    Decoder class that returns a normal distribution.

    Parameters:
    ----------
    n_features : int
        The number of output features of the decoder.

    """
    def __init__(self, out_features: int,
    ) -> None:
        super().__init__()
        self.scale_lin = torch.nn.Parameter(torch.zeros(1, out_features))
        self.bias = torch.nn.Parameter(torch.zeros(1, out_features))
        self.std_lin = torch.nn.Parameter(torch.zeros(1, out_features))

    def forward(
            self, u: torch.Tensor, v: torch.Tensor, l: Optional[torch.Tensor],
    ) -> D.Normal:
        """
        Forward pass of the decoder.

        Parameters:
        ----------
        u : torch.Tensor
            The data embedding.

        v : torch.Tensor
            The feature embedding.

        l : torch.Tensor
            The library size of the data.

        Returns:
        -------
        torch.Tensor
          The normal distribution.
        """
        scale = F.softplus(self.scale_lin[0])
        loc = scale * (u @ v.t()) + self.bias[0]
        std = F.softplus(self.std_lin[0]) + EPS

        return D.Normal(loc, std)

class BerDataDecoder(torch.nn.Module):
    """
    Decoder class that returns a bernoulli distribution.

    Parameters:
    ----------
    n_features : int
        The number of output features of the decoder.

    """
    def __init__(self, n_features):
        super().__init__()
        
        self.scale_lin = torch.nn.Parameter(torch.nn.init.xavier_uniform_(torch.zeros(1, n_features)))
        self.bias = torch.nn.Parameter(torch.nn.init.xavier_uniform_(torch.zeros(1, n_features)))

    def forward(self, u: torch.Tensor, v: torch.Tensor, l: torch.Tensor
    ) -> D.Bernoulli:
        """
        Forward pass of the decoder.

        Parameters:
        ----------
        u : torch.Tensor
            The data embedding.

        v : torch.Tensor
            The feature embedding.

        l : torch.Tensor
            The library size of the data.

        Returns:
        -------
        torch.Tensor
          The bernoulli distribution.
        """

        scale = F.softplus(self.scale_lin[0])
        logits = scale * (u @ v.t())  + self.bias[0]

        return D.Bernoulli(logits = logits)

class PoisDataDecoder(torch.nn.Module):
    """
    Decoder class that returns a poisson distribution.

    Parameters:
    ----------
    n_features : int
        The number of output features of the decoder.

    """
    def __init__(self, n_features, ):
        super().__init__()
        
        self.scale_lin = torch.nn.Parameter((torch.zeros(1, n_features)))
        self.bias = torch.nn.Parameter((torch.zeros(1, n_features)))

    def forward(self, u: torch.Tensor, v: torch.Tensor, l: torch.Tensor,
    ):
        """
        Forward pass of the decoder.

        Parameters:
        ----------
        u : torch.Tensor
            The data embedding.

        v : torch.Tensor
            The feature embedding.

        l : torch.Tensor
            The library size of the data.

        Returns:
        -------
        torch.Tensor
          The poisson distribution.
        """
        scale = F.softplus(self.scale_lin[0])
        rate = scale * (u @ v.t()) + self.bias[0]
        rate = F.softmax(rate, dim=1)  * l
        rate = torch.clamp(rate, max=1e12, min=1e-12)
        feature_dist = D.Poisson(
            rate=rate
        )

        return feature_dist

class ZINB(D.NegativeBinomial):
    """
    Zero-Inflated Negative Binomial (ZINB) distribution class.

    Parameters:
    ----------
    zi_logits : torch.Tensor
        The logits for the zero-inflation component.

    total_count : torch.Tensor
        The total count (or size) parameter for the negative binomial distribution.

    logits : torch.Tensor, optional
        The logits for the probability of success in the negative binomial distribution (default is None).
    """
    def __init__(
            self, zi_logits: torch.Tensor,
            total_count: torch.Tensor, logits: torch.Tensor = None
    ) -> None:
        super().__init__(total_count, logits=logits)
        self.zi_logits = zi_logits

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """
        Computes the log probability of the given value under the Zero-Inflated Negative Binomial (ZINB) distribution.

        Parameters:
        ----------
        value : torch.Tensor
            The value for which the log probability is computed.

        Returns:
        -------
        torch.Tensor
            The log probability of the given value under the ZINB distribution.
        """
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
    """
    Decoder class that returns a  zero-inflated negative binomial distribution.

    Parameters:
    ----------
    n_features : int
        The number of output features of the decoder.

    """
    def __init__(self, out_features: int ) -> None:
        super().__init__(out_features )
        self.zi_logits = torch.nn.Parameter(torch.zeros(1, out_features))

    def forward(self, u: torch.Tensor, v: torch.Tensor, l: torch.Tensor,):
        """
        Forward pass of the decoder.

        Parameters:
        ----------
        u : torch.Tensor
            The data embedding.

        v : torch.Tensor
            The feature embedding.

        l : torch.Tensor
            The library size of the data.

        Returns:
        -------
        torch.Tensor
          The ZINB distribution.
        """
        scale = F.softplus(self.scale_lin[0])
        logit_mu = scale * (u @ v.t()) + self.bias[0]
        mu = F.softmax(logit_mu, dim=1) * l
        log_theta = self.log_theta[0]
        feature_dist = ZINB(
            self.zi_logits[0].expand_as(mu),
            log_theta.exp(),
            logits=(mu + EPS).log() - log_theta
        )

        return feature_dist

class ZIP(D.Poisson):
    """
    Zero-Inflated Poisson (ZIP) distribution class.

    Parameters:
    ----------
    zi_logits : torch.Tensor
        The logits for the zero-inflation component.

    rate : torch.Tensor
        The rate (lambda) parameter for the Poisson distribution.

    """
    def __init__(
            self, zi_logits: torch.Tensor,
            rate: torch.Tensor
    ) -> None:
        super().__init__(rate)
        self.zi_logits = zi_logits

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """
        Computes the log probability of the given value under the Zero-Inflated Poisson (ZIPP) distribution.

        Parameters:
        ----------
        value : torch.Tensor
            The value for which the log probability is computed.

        Returns:
        -------
        torch.Tensor
            The log probability of the given value under the ZIPP distribution.
        """
        raw_log_prob = super().log_prob(value)
        zi_log_prob = torch.empty_like(raw_log_prob)
        z_mask = value.abs() < EPS
        z_zi_logits, nz_zi_logits = self.zi_logits[z_mask], self.zi_logits[~z_mask]
        zi_log_prob[z_mask] = (
            raw_log_prob[z_mask].exp() + z_zi_logits.exp() + EPS
        ).log() - F.softplus(z_zi_logits)
        zi_log_prob[~z_mask] = raw_log_prob[~z_mask] - F.softplus(nz_zi_logits)
        return zi_log_prob

class ZIPDataDecoder(PoisDataDecoder):
    """
    Decoder class that returns a  zero-inflated poisson distribution.

    Parameters:
    ----------
    n_features : int
        The number of output features of the decoder.

    """
    def __init__(self, out_features: int ) -> None:
        super().__init__(out_features )
        self.zi_logits = torch.nn.Parameter(torch.zeros(1, out_features))

    def forward(self, u: torch.Tensor, v: torch.Tensor, l: torch.Tensor,):
        """
        Forward pass of the decoder.

        Parameters:
        ----------
        u : torch.Tensor
            The data embedding.

        v : torch.Tensor
            The feature embedding.

        l : torch.Tensor
            The library size of the data.

        Returns:
        -------
        torch.Tensor
          The ZIP distribution.
        """
        scale = F.softplus(self.scale_lin[0])
        rate = scale * (u @ v.t()) + self.bias[0]
        rate = F.softmax(rate, dim=1)  * l
        rate = torch.clamp(rate, max=1e12, min=1e-12)
        feature_dist = ZIP(
            self.zi_logits[0].expand_as(rate),
            rate
        )

        return feature_dist

class Discriminator(torch.nn.Sequential):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            h_depth: int = 2,
            h_dim: Optional[int] = 256,
            dropout: float = 0.2,
            norm: Optional[str] = "batch"
    ) -> None:
        """
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
        od = collections.OrderedDict()
        ptr_dim = in_features
        for layer in range(h_depth):
            od[f"linear_{layer}"] = torch.nn.Linear(ptr_dim, h_dim)
            
            if layer == 0 or layer < h_depth-1:
                if norm == "batch":
                    od[f"bn_{layer}"] = torch.nn.BatchNorm1d(h_dim)
                elif norm == "layer":
                    od[f"ln_{layer}"] = torch.nn.LayerNorm(h_dim)
                elif norm == "group":
                    od[f"gn_{layer}"] = torch.nn.GroupNorm(8, h_dim)
            
            od[f"act_{layer}"] = torch.nn.LeakyReLU(negative_slope=0.2)
            od[f"dropout_{layer}"] = torch.nn.Dropout(p=dropout)
            ptr_dim = h_dim
        od["pred"] = torch.nn.Linear(ptr_dim, out_features)
        super().__init__(od)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
    """
    Early stopping mechanism for training, designed to stop the training process if the model performance 
    (losses) does not improve for a certain number of steps.

    Parameters:
    ----------
    patience : int, optional (default=200)
        The number of steps with no improvement after which training will be stopped.

    gen_delta : float, optional (default=2e-4)
        The threshold for considering the generator loss as unchanged.

    dsc_delta : float, optional (default=1e-3)
        The threshold for considering the discriminator loss as unchanged.

    verbose : bool, optional (default=False)
        If True, prints progress messages when the training is stopped.

    step : int, optional (default=50)
        The number of steps after which the average loss is calculated.
    """
    def __init__(self,
                 patience: int = 200,
                 gen_delta: float = 2e-4,
                 dsc_delta: float = 1e-3,
                 verbose: bool = False,
                 step: int = 50
    ) -> None:
    
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

    def __call__(self, val_loss: torch.Tensor, dsc_loss: torch.Tensor):
        """
        Checks if early stopping criteria are met based on the validation loss and discriminator loss.

        Parameters:
        ----------
        val_loss : torch.Tensor
            The current validation loss.

        dsc_loss : torch.Tensor
            The current discriminator loss.

        Returns:
        -------
        bool
            True if early stopping criteria are met, False otherwise.
        """

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
    """
    Learning rate scheduler for warm-up followed by learning rate decay.

    Parameters:
    ----------
    optimizer : torch.optim.Optimizer
        The optimizer for which the learning rate scheduler will be applied.

    warmup_epochs : int, optional (default=500)
        The number of epochs over which the learning rate will be warmed up.

    base_lr : float, optional (default=2e-4)
        The initial (lowest) learning rate before warm-up starts.

    target_lr : float, optional (default=2e-3)
        The target learning rate at the end of the warm-up period.

    step_size : int, optional (default=50)
        The number of steps (or epochs) after which the learning rate will be adjusted.

    gamma : float, optional (default=0.9)
        The learning rate decay factor after warm-up.
    """
    def __init__(self,
                 optimizer,
                 warmup_epochs: int = 500,
                 base_lr: float = 2e-4,
                 target_lr: float = 2e-3,
                 step_size: int = 50,
                 gamma: float = 0.9
    ) -> None:
        
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.base_lr = base_lr
        self.target_lr = target_lr
        self.step_size = step_size
        self.gamma = gamma
        self.flag = False

        self.decay_scheduler = StepLR(self.optimizer, step_size=self.step_size, gamma=self.gamma)

    def set_lr(self, lr: float):
        """
        Sets the learning rate for the optimizer.

        Parameters:
        ----------
        lr : float
            The learning rate to be set.
        """
        for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

    def step(self, epoch: int):
        """
        Updates the learning rate based on the current epoch.

        Parameters:
        ----------
        epoch : int
            The current epoch number.
        """
        if epoch < self.warmup_epochs:
            lr = self.base_lr + (self.target_lr - self.base_lr) * (epoch / self.warmup_epochs)
            # lr = self.target_lr
            self.set_lr(lr)
    
        elif(not self.flag):
            self.decay_scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            if current_lr < self.base_lr:
                self.set_lr(self.base_lr)
                self.flag = True

    def get_lr(self):
        """
        Returns the current learning rate.

        Returns:
        -------
        float
            The current learning rate.
        """
        return self.optimizer.param_groups[0]['lr']
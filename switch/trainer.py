import itertools
from typing import List, Mapping, Tuple
import numpy as np

import torch
import torch.distributions as D
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader
from torch_geometric.data import Data


from . import model
from .nn import SWITCH_nn
from .utils import normalize_edges


DataTensors = Tuple[
    Mapping[str, torch.Tensor],  # x (data)
    Mapping[str, torch.Tensor],  # xbch (data batch)
    Mapping[str, torch.Tensor],  # xflag (modality indicator)
    Mapping[str, torch.Tensor],  # snet (Spatial Net)
    torch.Tensor,  # eidx (edge index)
    torch.Tensor,  # ewt (edge weight)
    torch.Tensor  # esgn (edge sign)
]  

EPS = 1e-7

class Trainer():
    """
    Trainer class for the SWITCH model.

    Parameters:
    ----------
    net : SWITCH_nn
        The SWITCH_nn model to be trained.

    lam_data : float, optional
        The weight for the data reconstruction loss.

    lam_graph : float, optional
        The weight for the graph reconstruction loss.

    lam_adv : float, optional
        The weight for the adversarial loss.

    lam_cycle : float, optional
        The weight for the cycle consistency loss.

    lam_align : float, optional
        The weight for the pseudo-pair alignment loss.

    lam_kl : float, optional
        The weight for the KL divergence loss.

    optim : str, optional
        The optimizer to be used (e.g., 'Adam', 'SGD').

    lr : float, optional
        The learning rate for the optimizer.

    TTUR : float, optional (default=1.0)
        The learning rate ratio between the discriminator and VAE.

    **kwargs
        Additional parameters to be passed to the optimizer.
    """
    def __init__(
            self,
            net: SWITCH_nn,
            lam_data: float=None,
            lam_graph: float=None,
            lam_adv: float=None,
            lam_cycle: float=None,
            lam_align: float=None,
            lam_kl: float=None,
            optim: str = None, 
            lr: float = None,
            TTUR: float=1.0,
            **kwargs
    ) -> None:
        
        required_kwargs = ("lam_kl", "lam_graph", "lam_adv","lam_data", "lam_align",
                           "lam_cycle","optim", "lr")
        
        for required_kwarg in required_kwargs:
            if locals()[required_kwarg] is None:
                raise ValueError(f"`{required_kwarg}` must be specified!")
            
        self.net = net

        self.lam_kl = lam_kl
        self.lam_graph = lam_graph
        self.lam_align = lam_align
        self.lam_iden = lam_data
        self.lam_cycle = lam_cycle
        self.lam_adv = lam_adv

        self.pretrained = False

        self.lr = lr
        self.TTUR = TTUR
        self.vae_optim = getattr(torch.optim, optim)(
            itertools.chain(
                self.net.g2v.parameters(),
                self.net.v2g.parameters(),
                self.net.x2u.parameters(),
                self.net.u2x.parameters(),
            ), lr=self.lr ,  **kwargs
        )
        self.dsc_optim = getattr(torch.optim, optim)(
            itertools.chain(
                self.net.du.parameters()
            ), lr=self.lr*TTUR, **kwargs
        )

        self.pretrain_epoch = 0
        self.train_epoch = 0

    def format_data(
            self,
            data: List[np.array],
            graph_data: List[torch.Tensor],
            mini_batch: bool = False,
    ) -> List[torch.Tensor]:
        """
        Formats the data into a usable format for training.

        Parameters:
        ----------
        data : List[np.array]
            A list of data matrices.

        graph_data : List[torch.Tensor]
            A list of feature graph data.

        mini_batch : bool, optional (default=False)
            Whether to perform mini-batch training. If True, all data is temporarily moved to CPU.

        Returns:
        -------
        List[torch.Tensor]
            The formatted data ready for training.
        """
        device = self.net.device
        keys = self.net.keys
        K = len(keys)
        x, xbch, snet, edge_type = \
            data[0:K], data[K:2*K], data[2*K:3*K], data[3*K:4*K]
        (eidx, ewt, esgn) = graph_data
        temp_device = 'cpu' if mini_batch else device
    
        x = {
            k: torch.as_tensor(x[i], device=temp_device)
            for i, k in enumerate(keys)
        }
        xbch = {
            k: torch.as_tensor(xbch[i], device=temp_device)
            for i, k in enumerate(keys)
        }
        xflag = {
            k: torch.as_tensor(
                i, dtype=torch.int64, device=temp_device
            ).expand(x[k].shape[0])
            for i, k in enumerate(keys)
        }
        snet = {
            k: torch.as_tensor(snet[i], device=temp_device)
            for i, k in enumerate(keys)
        }
        # edge_type = {
        #     k: torch.as_tensor(edge_type[i], device=temp_device)
        #     for i, k in enumerate(keys)
        # }
        enorm = torch.as_tensor(
            normalize_edges(eidx, ewt),
            device=device
        )
        eidx = torch.as_tensor(eidx, device=device)
        ewt = torch.as_tensor(ewt, device=device)
        esgn = torch.as_tensor(esgn, device=device)
        return x, xbch,  xflag, snet, eidx, ewt, esgn, enorm

    def sample_neighbor(
            self,
            data: List[np.array],
            graph_data: List[torch.Tensor],
            iteration: int=1,
            sizes: int=10
    ) -> List:
        """
        Samples neighbors for mini-batch training in a graph neural network.

        Parameters:
        ----------
        data : List[np.array]
            A list of data matrices.

        graph_data : List[torch.Tensor]
            A list of feature graph data.

        iteration : int, optional (default=1)
            The number of iterations for mini-batch training.

        sizes : int, optional (default=10)
            The number of neighbors to sample.

        Returns:
        -------
        List
            The sampled neighbor data for mini-batch training.
        """

        data = self.format_data(data, graph_data, mini_batch=True)
        x, xbch, xflag, snet, eidx, ewt, esgn, enorm = data
        keys = self.net.keys
        NeighborLoaders = dict()
        for key in keys:
            train_data = Data(x=x[key], edge_index =snet[key], xbch=xbch[key], xflag=xflag[key])
            h_depth = 2
            batch_size = int(x[key].shape[0]/iteration)
            loader = NeighborLoader(train_data, num_neighbors=[sizes] * h_depth, batch_size=batch_size)
            NeighborLoaders[key] = loader

        return NeighborLoaders, eidx, ewt, esgn, enorm
        
    def compute_losses(
            self,
            data: DataTensors,
            dsc_only: bool = False,
            pretrain: bool = False, 
            cycle_key: List = [],
            normalize_methods: dict={}
    ) -> Mapping[str, torch.Tensor]:
        """
        Computes the losses for the model during training.

        Parameters:
        ----------
        data : DataTensors
            The data formatted and preprocessed for training.

        dsc_only : bool, optional (default=False)
            Whether to train only the discriminator.

        pretrain : bool, optional (default=False)
            Whether it's pretraining (i.e., excluding cycle mapping loss and pseudo-pair alignment loss).

        cycle_key : List, optional (default=[])
            The modality used for calculating the cycle mapping loss and pseudo-pair alignment loss.

        normalize_methods : dict, optional (default={})
            A dictionary of normalization methods for different modalities.

        Returns:
        -------
        Mapping[str, torch.Tensor]
            A mapping of loss names to their computed loss values.
        """

        net = self.net
        x, xbch, xflag, snet, eidx, ewt, esgn, enorm = data
        u, l = {}, {}
        keys = net.keys
        A = keys[0]
        B = keys[1]

        # Encodding
        for k in net.keys:
            u[k], l[k] = net.x2u[k](x[k], snet[k], normalize=normalize_methods[k])

        usamp = {k: u[k].rsample() for k in net.keys}
        prior = net.prior()

        u_cat = torch.cat([u[k].mean for k in net.keys])
        xbch_cat = torch.cat([xbch[k] for k in net.keys])
        xflag_cat = torch.cat([xflag[k] for k in net.keys])
        
        # GAN loss
        # alpha =  (1 - np.exp(-0.005 * epoch))
        dsc_loss = F.cross_entropy(net.du(u_cat, xbch_cat), xflag_cat, reduction="mean")

        if dsc_only:
            return {"dsc_loss":  self.lam_adv * dsc_loss}
                
        v = net.g2v(eidx, enorm, esgn)
        vsamp = v.rsample()
        g_nll = -net.v2g(vsamp, eidx, esgn).log_prob(ewt)
        pos_mask = (ewt != 0).to(torch.int64)
        n_pos = pos_mask.sum().item()
        n_neg = pos_mask.numel() - n_pos
        g_nll_pn = torch.zeros(2, dtype=g_nll.dtype, device=g_nll.device)
        g_nll_pn.scatter_add_(0, pos_mask, g_nll)
        avgc = (n_pos > 0) + (n_neg > 0)
        g_nll = (g_nll_pn[0] / max(n_neg, 1) + g_nll_pn[1] / max(n_pos, 1)) / avgc
        g_kl = D.kl_divergence(v, prior).sum(dim=1).mean() / vsamp.shape[0]
        g_elbo = g_nll + self.lam_kl * g_kl

        ## Recon loss
        recon_data = {
            A: net.u2x[A](
                usamp[A], vsamp[getattr(net, f"{A}_idx")], xbch[A], l[A],
            ),
            B: net.u2x[B](
                usamp[B], vsamp[getattr(net, f"{B}_idx")], xbch[B], l[B],
            )
        }
        recon_loss = sum([-recon_data[k][0].log_prob(x[k]).mean() for k in net.keys])

        for k in net.keys:
            del recon_data[k]
        
        ## KL loss
        x_kl = {
            k: D.kl_divergence(
                u[k], prior
            ).sum(dim=1).mean() / x[k].shape[1]
            for k in net.keys
        }
    
        kl_loss = sum([x_kl[k] for k in net.keys])

        gen_loss = self.lam_kl * kl_loss + self.lam_iden * recon_loss \
            + self.lam_graph * len(net.keys) * g_elbo \
            - self.lam_adv * dsc_loss
        
        losses = {
            "dsc_loss": dsc_loss, "gen_loss": gen_loss, "kl_loss": kl_loss, "iden_loss": recon_loss
        }

        if(not pretrain):

            fake_data = dict()
            if(A in cycle_key):
                fake_data[B] = net.u2x[B](
                    usamp[A], vsamp[getattr(net, f"{B}_idx")], xbch[A],
                    torch.as_tensor(1.0, device = net.device)
                )[0].mean
                fakeB_u = net.x2u[B](fake_data[B], 
                                     snet[A], 
                                     normalize=normalize_methods[B])
            
                del fake_data[B]

            if(B in cycle_key):
                fake_data[A] = net.u2x[A](
                    usamp[B], vsamp[getattr(net, f"{A}_idx")], xbch[B],
                    torch.as_tensor(1.0, device = net.device),
                )[0].mean
                fakeA_u = net.x2u[A](fake_data[A], 
                                     snet[B],
                                     normalize=normalize_methods[A])
            
                del fake_data[A]

            ## Cycle loss
            if(A in cycle_key):
                cycle_loss_A = -net.u2x[A](
                    fakeB_u[0].mean, vsamp[getattr(net, f"{A}_idx")], xbch[A], l[A],
                    )[0].log_prob(x[A])
            else:
                cycle_loss_A = torch.tensor(0.0, device=net.device)
            if(B in cycle_key):
                cycle_loss_B = -net.u2x[B](
                    fakeA_u[0].mean, vsamp[getattr(net, f"{B}_idx")], xbch[B], l[B],
                    )[0].log_prob(x[B])
            else:
                cycle_loss_B = torch.tensor(0.0, device=net.device)

            ## Align loss
            if(A in cycle_key):
                align_lossA = torch.exp(-torch.mean(F.cosine_similarity(u[A].mean, fakeB_u[0].mean, dim=1)))
            else:
                align_lossA = torch.tensor(0.0, device=net.device)
            if(B in cycle_key):
                align_lossB = torch.exp(-torch.mean(F.cosine_similarity(u[B].mean, fakeA_u[0].mean, dim=1)))
            else:
                align_lossB = torch.tensor(0.0, device=net.device)
        
            cycle_loss = cycle_loss_A.mean() + cycle_loss_B.mean()
            align_loss = align_lossA + align_lossB


            gen_loss = gen_loss + self.lam_cycle * cycle_loss + self.lam_align * align_loss

            losses["gen_loss"] = gen_loss
            losses["cycle_loss"] = cycle_loss
            losses["align_loss"] = align_loss
    
        return losses

    def reset_lr(
            self, lr: float, optim: str=None, TTUR: float=None
    ):
        """
        Resets the learning rate for VAE and discriminator.

        Parameters:
        ----------
        lr : float
            The new learning rate to be set.

        optim : optional
            The optimizer to be used.

        TTUR : float, optional
            The learning rate ratio between the discriminator and VAE.
        """

        optim = optim or ["dsc_optim","vae_optim"]
        TTUR = TTUR or self.TTUR
        if(isinstance(optim, str)):
            assert optim in ["dsc_optim","vae_optim"]
            opt = getattr(self, optim)
            if(optim=="dsc_optim"):
                for params in opt.param_groups:                        
                    params['lr'] = lr * TTUR
            else:
                for params in opt.param_groups:                        
                    params['lr'] = lr
        elif(isinstance(optim, list)):
            for t in optim:
                if(t=="dsc_optim"):
                    opt = getattr(self, t)
                    for params in opt.param_groups:                        
                        params['lr'] = lr * TTUR
                else:
                    opt = getattr(self, t)
                    for params in opt.param_groups:                        
                        params['lr'] = lr

    def pretrain(
            self,
            data: List[torch.Tensor],
            graph_data: List[torch.Tensor],
            max_epochs: int=None, 
            mini_batch: bool=False,
            iteration: int=1,
            dsc_k: int=None,
            early_stop : bool=False,
            warmup: bool=False,
            log_step : int=100,
            early_stop_kwargs: dict={'gen_delta':2e-4, 'dsc_delta':2e-3, 'patience':250, 'verbose':False, 'step':50},
            warmup_kwargs : dict={'warmup_epochs':None, 'base_lr': None, 'max_lr':2e-3,'step_size':100, 'gamma':0.9}
    ) -> Mapping[str, torch.Tensor]:
        """
        Pre-trains the SWITCH model, excluding cycle mapping and pseudo-pair alignment losses.

        Parameters:
        ----------
        data : List[torch.Tensor]
            The training data in tensor format.

        graph_data : List[torch.Tensor]
            The feature graph data.

        max_epochs : int, optional (default=None)
            The maximum number of epochs for training.

        mini_batch : bool, optional (default=False)
            Whether to use mini-batch training.

        iteration : int, optional (default=1)
            The number of iterations for mini-batch training.

        dsc_k : int, optional (default=None)
            The number of times the discriminator is trained per VAE training step.

        early_stop : bool, optional (default=False)
            Whether to apply early stopping during training.

        warmup : bool, optional (default=False)
            Whether to perform warmup during training.

        log_step : int, optional (default=100)
            The number of epochs after which to log the loss.

        early_stop_kwargs : dict, optional (default={'gen_delta': 2e-4, 'dsc_delta': 2e-3, 'patience': 250, 'verbose': False, 'step': 50})
            Additional parameters to be passed to the early stop function.

        warmup_kwargs : dict, optional (default={'warmup_epochs': None, 'base_lr': None, 'max_lr': 2e-3, 'step_size': 100, 'gamma': 0.9})
            Additional parameters to be passed to the warmup function.

        Returns:
        -------
        Mapping[str, torch.Tensor]
            A mapping of loss names to their computed values.
        """
        
        if(early_stop):
            early_stop_default = {'gen_delta':2e-4, 'dsc_delta':2e-3, 'patience':250, 'verbose':False, 'step':100}
            for key in early_stop_default.keys():
                if not key in early_stop_kwargs:
                    early_stop_kwargs[key] = early_stop_default[key]
            self.earlystop = model.EarlyStopping(gen_delta=early_stop_kwargs['gen_delta'], 
                                                dsc_delta=early_stop_kwargs['dsc_delta'],
                                                patience=early_stop_kwargs['patience'],
                                                verbose=early_stop_kwargs['verbose'],
                                                step=early_stop_kwargs['step'])
        
        if(warmup):
            warmup_default = {'warmup_epochs':None, 'base_lr':None, 'max_lr':2e-3,'step_size':100, 'gamma':0.9}
            for key in warmup_default.keys():
                if not key in warmup_kwargs:
                    warmup_kwargs[key] = warmup_default[key]
            warmup_kwargs["base_lr"] = warmup_kwargs["base_lr"] or self.lr
            warmup_kwargs["warmup_epochs"] = warmup_kwargs["warmup_epochs"] or (0.2 * max_epochs)
            self.vae_warmup = model.WarmUpScheduler(self.vae_optim, warmup_kwargs["warmup_epochs"],
                                                    warmup_kwargs["base_lr"], warmup_kwargs["max_lr"],
                                                    warmup_kwargs["step_size"], warmup_kwargs["gamma"])
            self.dsc_warmup = model.WarmUpScheduler(self.dsc_optim, warmup_kwargs["warmup_epochs"],
                                                    warmup_kwargs["base_lr"] * self.TTUR, warmup_kwargs["max_lr"] * self.TTUR,
                                                    warmup_kwargs["step_size"], warmup_kwargs["gamma"])
            # last_lr = self.vae_warmup.get_lr()
            
        normalize_methods = self.net.normalize_methods

        if(not mini_batch):
            self.net.logger.info(f"Pretraining with full batch.")
            data = self.format_data(data, graph_data)
            self.net.train()
            cur_epoch = 0
            while(cur_epoch < max_epochs):
                cur_epoch += 1
                self.pretrain_epoch += 1

                for _ in range(dsc_k):
                    losses = self.compute_losses(data, dsc_only=True, pretrain=True, normalize_methods=normalize_methods)
                    self.net.zero_grad(set_to_none=True)
                    losses["dsc_loss"].backward()
                    self.dsc_optim.step()
                
                losses = self.compute_losses(data,  pretrain=True, normalize_methods=normalize_methods)
                self.net.zero_grad(set_to_none=True)
                losses["gen_loss"].backward()
                self.vae_optim.step()

                if(warmup):
                    self.vae_warmup.step(self.pretrain_epoch)
                    self.dsc_warmup.step(self.pretrain_epoch)

                dsc_loss = float(losses["dsc_loss"].detach().cpu().numpy())
                gen_loss = float(losses["gen_loss"].detach().cpu().numpy())

                if(cur_epoch % log_step == 0):
                    self.net.logger.info(f"Epoch {int(cur_epoch/log_step)} : dsc_loss={round(dsc_loss, 3)}, gen_loss={round(gen_loss, 3)}")
                    
                if(early_stop):
                    self.earlystop(val_loss=gen_loss, dsc_loss=dsc_loss)
                    if(self.earlystop.early_stop):
                        self.net.logger.info(f"Earlystop at epoch {int(cur_epoch/log_step)} : dsc_loss={round(dsc_loss, 3)}, gen_loss={round(gen_loss, 3)}")
                        break
    
            self.net.eval()
            self.pretrained=True
        else:
            ## mini-batch
            self.net.logger.info(f"Pretraining with mini-batch, iteration = {iteration}.")
            device = self.net.device
            NeighborLoaders, eidx, ewt, esgn, enorm = self.sample_neighbor(data, graph_data, iteration=iteration)
            self.net.train()
            cur_epoch = 0
            while(cur_epoch < max_epochs):
                cur_epoch += 1
                self.pretrain_epoch += 1

                for _ in range(iteration):
                    x, snet, xbch, xflag =  {}, {}, {}, {}
                    for key in self.net.keys:
                        sample_data = next(iter(NeighborLoaders[key]))
                        x[key] = sample_data.x.to(device)
                        snet[key] = sample_data.edge_index.to(device)
                        xbch[key] = sample_data.xbch.to(device)
                        xflag[key] = sample_data.xflag.to(device)

                    data = x,  xbch, xflag, snet, eidx, ewt, esgn, enorm                  

                    for _ in range(dsc_k):
                        losses = self.compute_losses(data, dsc_only=True, pretrain=True, normalize_methods=normalize_methods)
                        self.net.zero_grad(set_to_none=True)
                        losses["dsc_loss"].backward()
                        self.dsc_optim.step()

                    losses = self.compute_losses(data, dsc_only=False, pretrain=True, normalize_methods=normalize_methods)
                    self.net.zero_grad(set_to_none=True)
                    losses["gen_loss"].backward()
                    self.vae_optim.step()
                
                if(warmup):
                    self.vae_warmup.step(self.pretrain_epoch)
                    self.dsc_warmup.step(self.pretrain_epoch)
                
                dsc_loss = float(losses["dsc_loss"].detach().cpu().numpy())
                gen_loss = float(losses["gen_loss"].detach().cpu().numpy())

                if(cur_epoch % log_step == 0):
                    self.net.logger.info(f"Epoch {int(cur_epoch/log_step)} : dsc_loss={round(dsc_loss, 3)}, gen_loss={round(gen_loss, 3)}")

                if(early_stop):
                    self.earlystop(val_loss=gen_loss, dsc_loss=dsc_loss)
                    if(self.earlystop.early_stop):
                        self.net.logger.info(f"Earlystop at epoch {int(cur_epoch/log_step)} : dsc_loss={round(dsc_loss, 3)}, gen_loss={round(gen_loss, 3)}")
                        break
    
            self.net.eval()
            self.pretrained=True

    def train(
            self,
            data: List[torch.Tensor],
            graph_data: List[torch.Tensor],
            max_epochs: int=None,
            mini_batch: bool=False,
            iteration: int=1,
            dsc_k: int=None,
            cycle_key: List=[],
            early_stop : bool=False,
            log_step: int = 100,
            warmup: bool = False,
            early_stop_kwargs: dict={'gen_delta':1e-4, 'dsc_delta':2e-3, 'patience':400, 'verbose':False, 'step':50},
            warmup_kwargs : dict= {'warmup_epochs':None, 'base_lr':None, 'max_lr':2e-3,'step_size':50, 'gamma':0.8}
    ) -> Mapping[str, torch.Tensor]:
        """
        Trains the SWITCH model, including cycle mapping and pseudo-pair alignment losses.

        Parameters:
        ----------
        data : List[torch.Tensor]
            The training data in tensor format.

        graph_data : List[torch.Tensor]
            The feature graph data.

        max_epochs : int, optional (default=None)
            The maximum number of epochs for training.

        mini_batch : bool, optional (default=False)
            Whether to use mini-batch training.

        iteration : int, optional (default=1)
            The number of iterations for mini-batch training.

        dsc_k : int, optional (default=None)
            The number of times the discriminator is trained per VAE training step.

        cycle_key : List, optional (default=[])
            The modality used for calculating the cycle mapping loss.

        early_stop : bool, optional (default=False)
            Whether to apply early stopping during training.

        log_step : int, optional (default=100)
            The number of epochs after which to log the loss.

        warmup : bool, optional (default=False)
            Whether to perform warmup during training.

        early_stop_kwargs : dict, optional (default={'gen_delta': 1e-4, 'dsc_delta': 2e-3, 'patience': 400, 'verbose': False, 'step': 50})
            Additional parameters to be passed to the early stop function.

        warmup_kwargs : dict, optional (default={'warmup_epochs': None, 'base_lr': None, 'max_lr': 2e-3, 'step_size': 50, 'gamma': 0.8})
            Additional parameters to be passed to the warmup function.

        Returns:
        -------
        Mapping[str, torch.Tensor]
            A mapping of loss names to their computed values.
        """

        if(early_stop):
            early_stop_default = {'gen_delta':1e-4, 'dsc_delta':2e-3, 'patience':400, 'verbose':False, 'step':50}
            for key in early_stop_default.keys():
                if not key in early_stop_kwargs:
                    early_stop_kwargs[key] = early_stop_default[key]
            self.earlystop = model.EarlyStopping(gen_delta=early_stop_kwargs['gen_delta'], 
                                                dsc_delta=early_stop_kwargs['dsc_delta'],
                                                patience=early_stop_kwargs['patience'],
                                                verbose=early_stop_kwargs['verbose'],
                                                step=early_stop_kwargs['step'])
            
        if(warmup):
            warmup_default = {'warmup_epochs':None, 'base_lr':None, 'max_lr':2e-3,'step_size':50, 'gamma':0.8}
            for key in warmup_default.keys():
                if not key in warmup_kwargs:
                    warmup_kwargs[key] = warmup_default[key]
            warmup_kwargs["base_lr"] = warmup_kwargs["base_lr"] or self.lr
            warmup_kwargs["warmup_epochs"] = warmup_kwargs["warmup_epochs"] or (0.2 * max_epochs)
            self.vae_warmup = model.WarmUpScheduler(self.vae_optim, warmup_kwargs["warmup_epochs"],
                                                    warmup_kwargs["base_lr"], warmup_kwargs["max_lr"],
                                                    warmup_kwargs["step_size"], warmup_kwargs["gamma"])
            self.dsc_warmup = model.WarmUpScheduler(self.dsc_optim, warmup_kwargs["warmup_epochs"],
                                                    warmup_kwargs["base_lr"] * self.TTUR, warmup_kwargs["max_lr"] * self.TTUR,
                                                    warmup_kwargs["step_size"], warmup_kwargs["gamma"])
            # last_lr = self.vae_warmup.get_lr()
            
        normalize_methods = self.net.normalize_methods

        if(not mini_batch):    
            self.net.logger.info(f"Training with full batch.")
            self.net.train()
            data = self.format_data(data, graph_data)
            cur_epoch = 0            
            while(cur_epoch < max_epochs):
                cur_epoch += 1
                self.train_epoch += 1
                
                for _ in range(dsc_k):
                    losses = self.compute_losses(data, dsc_only=True, pretrain=False, normalize_methods=normalize_methods)
                    self.net.zero_grad(set_to_none=True)
                    losses["dsc_loss"].backward()
                    self.dsc_optim.step()


                losses = self.compute_losses(data, pretrain=True, normalize_methods=normalize_methods)
                self.net.zero_grad(set_to_none=True)
                losses["gen_loss"].backward()
                self.vae_optim.step()
                
                losses = self.compute_losses(data, pretrain=False, cycle_key=cycle_key, normalize_methods=normalize_methods)
                self.net.zero_grad(set_to_none=True)
                losses["gen_loss"].backward()
                self.vae_optim.step()

                if(warmup):
                    self.vae_warmup.step(self.train_epoch)
                    self.dsc_warmup.step(self.train_epoch)

                dsc_loss = float(losses["dsc_loss"].detach().cpu().numpy())
                gen_loss = float(losses["gen_loss"].detach().cpu().numpy())
                align_loss = float(losses["align_loss"].detach().cpu().numpy())
                cycle_loss = float(losses["cycle_loss"].detach().cpu().numpy())

                if(cur_epoch % log_step == 0):
                    self.net.logger.info(f"Epoch {int(cur_epoch/log_step)} : dsc_loss={round(dsc_loss, 3)}, " + \
                                         f"gen_loss={round(gen_loss, 3)}, " + \
                                         f"cycle_loss={round(cycle_loss, 3)}, align_loss={round(align_loss, 3)}")
                    
                if(early_stop):
                    self.earlystop(val_loss=gen_loss, dsc_loss=dsc_loss)
                    if(self.earlystop.early_stop):
                        self.net.logger.info(f"Earlystop at epoch {int(cur_epoch/log_step)} : dsc_loss={round(dsc_loss, 3)}, " + \
                                             f"gen_loss={round(gen_loss, 3)}, " + \
                                             f"cycle_loss={round(cycle_loss, 3)}, align_loss={round(align_loss, 3)}")
                        break
                
            self.net.eval()
        else:
            ## mini-batch
            self.net.logger.info(f"Training with mini-batch, iteration = {iteration}.")
            device = self.net.device
            NeighborLoaders, eidx, ewt, esgn, enorm = self.sample_neighbor(data, graph_data, iteration=iteration)
            self.net.train()
            cur_epoch = 0
            
            while(cur_epoch < max_epochs):
                cur_epoch += 1
                self.train_epoch += 1

                for _ in range(iteration):
                    x, snet, xbch, xflag =  {}, {}, {}, {}
                    for key in self.net.keys:
                        sample_data = next(iter(NeighborLoaders[key]))
                        x[key] = sample_data.x.to(device)
                        snet[key] = sample_data.edge_index.to(device)
                        xbch[key] = sample_data.xbch.to(device)
                        xflag[key] = sample_data.xflag.to(device)

                    data = x, xbch, xflag, snet, eidx, ewt, esgn, enorm

                    for _ in range(dsc_k):
                        losses = self.compute_losses(data, dsc_only=True, pretrain=False, normalize_methods=normalize_methods)
                        self.net.zero_grad(set_to_none=True)
                        losses["dsc_loss"].backward()
                        self.dsc_optim.step()
            
                    losses = self.compute_losses(data, pretrain=True, normalize_methods=normalize_methods)
                    self.net.zero_grad(set_to_none=True)
                    losses["gen_loss"].backward()
                    self.vae_optim.step()

                    losses = self.compute_losses(data, pretrain=False, cycle_key=cycle_key, normalize_methods=normalize_methods)
                    self.net.zero_grad(set_to_none=True)
                    losses["gen_loss"].backward()
                    self.vae_optim.step()
                
                if(warmup):
                    self.vae_warmup.step(self.train_epoch)
                    self.dsc_warmup.step(self.train_epoch)

                dsc_loss = float(losses["dsc_loss"].detach().cpu().numpy())
                gen_loss = float(losses["gen_loss"].detach().cpu().numpy())
                align_loss = float(losses["align_loss"].detach().cpu().numpy())
                cycle_loss = float(losses["cycle_loss"].detach().cpu().numpy())

                if(cur_epoch % log_step == 0):
                    self.net.logger.info(f"Epoch {int(cur_epoch/log_step)} : dsc_loss={round(dsc_loss, 3)}, " + \
                                         f"gen_loss={round(gen_loss, 3)}, " + \
                                         f"cycle_loss={round(cycle_loss, 3)}, align_loss={round(align_loss, 3)}")
                    
                if(early_stop):
                    self.earlystop(val_loss=gen_loss, dsc_loss=dsc_loss)
                    if(self.earlystop.early_stop):
                        self.net.logger.info(f"Earlystop at epoch {int(cur_epoch/log_step)} : dsc_loss={round(dsc_loss, 3)}, " + \
                                             f"gen_loss={round(gen_loss, 3)}, " + \
                                             f"cycle_loss={round(cycle_loss, 3)}, align_loss={round(align_loss, 3)}")
                        break

            self.net.eval()

    def save(
            self, path: str
    ) -> None:
        """
        Saves the trained model to the specified path.

        Parameters:
        ----------
        path : str
            The path where the model will be saved.
        """
        torch.save({
        'model_state_dict': self.net.state_dict(),
        'vae_opt_state_dict': self.vae_optim.state_dict(),
        'dsc_opt_state_dict': self.dsc_optim.state_dict(),
        }, path)
    
    def load(
            self, path: str
    )-> None:
        """
        Loads the model from the specified path.

        Parameters:
        ----------
        path : str
            The path from which the model will be loaded.
        """
        checkpoint = torch.load(path)
        self.net.load_state_dict(checkpoint['model_state_dict'])
        self.vae_optim.load_state_dict(checkpoint['vae_opt_state_dict'])
        self.dsc_optim.load_state_dict(checkpoint['dsc_opt_state_dict'])

    def __repr__(self):
        vae_optim = repr(self.vae_optim).replace("    ", "  ").replace("\n", "\n  ")
        dsc_optim = repr(self.dsc_optim).replace("    ", "  ").replace("\n", "\n  ")
        return (
            f"{type(self).__name__}(\n"
            f"  lam_data: {self.lam_iden}\n"
            f"  lam_graph: {self.lam_graph}\n"
            f"  lam_adv: {self.lam_adv}\n"
            f"  lam_cycle: {self.lam_cycle}\n"
            f"  lam_align: {self.lam_align}\n"
            f"  lam_kl: {self.lam_kl}\n"
            f"  vae_optim: {vae_optim}\n"
            f"  dsc_optim: {dsc_optim}\n"
            f")"
        )
import itertools
from typing import List, Mapping, Optional, Tuple, Union
from tqdm import tqdm
import numpy as np

import torch
import torch.distributions as D
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader
from torch_geometric.data import Data, NeighborSampler
import scipy
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances


from . import model
from .nn import SWITCH_nn
from .utils import normalize_edges
import copy 


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

    def __init__(
            self, net: SWITCH_nn, lam_kl: float = None, lam_graph: float = None, lam_adv: float = None, 
            lam_iden = None,  lam_align = None, lam_cycle = None,  optim: str = None, 
            lr: float = None, TTUR: float = 1.0, **kwargs
    ) -> None:
        
        required_kwargs = ("lam_kl", "lam_graph", "lam_adv","lam_iden", "lam_align",
                           "lam_cycle","optim", "lr")
        
        for required_kwarg in required_kwargs:
            if locals()[required_kwarg] is None:
                raise ValueError(f"`{required_kwarg}` must be specified!")
            
        self.net = net

        self.lam_kl = lam_kl
        self.lam_graph = lam_graph
        self.lam_align = lam_align
        self.lam_iden = lam_iden
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
        self.W = None

    def format_data(self, data: List[np.array], graph_data: List[torch.Tensor], mini_batch: bool = False,) -> List[torch.Tensor]:
        r"""
        Format data tensors

        Note
        ----
        The data dataset should contain data arrays for each modality,
        followed by alternative input arrays for each modality,
        in the same order as modality keys of the network.
        """
        device = self.net.device
        keys = self.net.keys
        K = len(keys)
        x, xbch, snet, edge_type = \
            data[0:K], data[K:2*K], data[2*K:3*K], data[3*K:4*K]
        (eidx, ewt, esgn) = graph_data
        temp_device = 'cpu' if mini_batch else device
        # x = {
        #     k: torch.as_tensor(x[i], device=device)
        #     for i, k in enumerate(keys)
        # }
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

    def sample_neighbor(self, data: List[np.array], graph_data: List[torch.Tensor], iteration: int=1, sizes: int=10):

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
            self, data: DataTensors, dsc_only: bool = False, pretrain: bool = False, 
            cycle_key: List = [], normalize_methods: dict={}
    ) -> Mapping[str, torch.Tensor]:
        net = self.net
        x, xbch, xflag, snet, eidx, ewt, esgn, enorm = data
        u, l = {}, {}
        keys = net.keys
        A = keys[0]
        B = keys[1]

        # Encoder
        for k in net.keys:
            u[k], l[k] = net.x2u[k](x[k], snet[k], normalize=normalize_methods[k])

        usamp = {k: u[k].rsample() for k in net.keys}
        prior = net.prior()

        u_cat = torch.cat([u[k].mean for k in net.keys])
        xbch_cat = torch.cat([xbch[k] for k in net.keys])
        xflag_cat = torch.cat([xflag[k] for k in net.keys])
        
        #if pretrain:
        #    if(self.pretrain_epoch < self.align_burnin):
        #        noise = D.Normal(0, u_cat.std(axis=0)).sample((u_cat.shape[0], ))
        #        u_cat = u_cat+self.align_noise*(self.align_burnin-self.pretrain_epoch)/self.align_burnin*noise
        #else:
        #    if(self.train_epoch < self.align_burnin/3):
        #        noise = D.Normal(0, u_cat.std(axis=0)).sample((u_cat.shape[0], ))
        #        u_cat = u_cat+self.align_noise*3*(self.align_burnin/3-self.train_epoch)/self.align_burnin*noise

        
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
                usamp[A], vsamp[getattr(net, f"{A}_idx")], xbch[A], l[A], snet[A]
            ),
            B: net.u2x[B](
                usamp[B], vsamp[getattr(net, f"{B}_idx")], xbch[B], l[B], snet[B]
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
                    usamp[A], vsamp[getattr(net, f"{B}_idx")], xbch[A], #l[A]
                    torch.as_tensor(1.0, device = net.device), snet[A]
                )[0].mean
                fakeB_u = net.x2u[B](fake_data[B], 
                                     snet[A],# torch.arange(x[A].shape[0]).repeat(2, 1).to(net.device), 
                                     normalize=normalize_methods[B])
            
                del fake_data[B]

            if(B in cycle_key):
                fake_data[A] = net.u2x[A](
                    usamp[B], vsamp[getattr(net, f"{A}_idx")], xbch[B],# l[B]
                    torch.as_tensor(1.0, device = net.device), snet[B]
                )[0].mean
                fakeA_u = net.x2u[A](fake_data[A], 
                                     snet[B],# torch.arange(x[B].shape[0]).repeat(2, 1).to(net.device), 
                                     normalize=normalize_methods[A])
            
                del fake_data[A]

            ## Cycle loss
            if(A in cycle_key):
                cycle_loss_A = -net.u2x[A](fakeB_u[0].mean, vsamp[getattr(net, f"{A}_idx")], xbch[A], l[A], snet[A]
                                           )[0].log_prob(x[A])
            else:
                cycle_loss_A = torch.tensor(0.0, device=net.device)
            if(B in cycle_key):
                cycle_loss_B = -net.u2x[B](fakeA_u[0].mean, vsamp[getattr(net, f"{B}_idx")], xbch[B], l[B], snet[B]
                                           )[0].log_prob(x[B])
            else:
                cycle_loss_B = torch.tensor(0.0, device=net.device)

            ## Joint loss
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

            # if(A in cycle_key):
            #     fakeBu_kl = D.kl_divergence(fakeB_u[0], prior).sum(dim=1).mean() / x[A].shape[1]
            # else:
            #     fakeBu_kl = torch.tensor(0.0, device=net.device)
            # if(B in cycle_key):
            #     fakeAu_kl = D.kl_divergence(fakeA_u[0], prior).sum(dim=1).mean() / x[B].shape[1]
            # else:
            #     fakeAu_kl = torch.tensor(0.0, device=net.device)
            # kl_loss = kl_loss +  fakeAu_kl + fakeBu_kl 
    
        return losses

    def reset_lr(self, lr, optim=None, TTUR=None):
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
            self, data: List[torch.Tensor], graph_data: List[torch.Tensor], max_epochs: int = None, 
            mini_batch: bool = False, iteration: int = 1, dsc_k: int = None,
            early_stop : bool = False, warmup: bool = False, log_step : int = 100,
            early_stop_kwargs: dict={'gen_delta':2e-4, 'dsc_delta':2e-3, 'patience':250, 'verbose':False, 'step':50},
            warmup_kwargs : dict= {'warmup_epochs':None, 'base_lr': None, 'max_lr':2e-3,'step_size':100, 'gamma':0.9}
     
    ) -> Mapping[str, torch.Tensor]:
        
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
                    # cur_lr = self.vae_warmup.get_lr()
                    # if(cur_lr < last_lr):
                    #     self.net.logger.debug(f"Learning Rate adjusted to {last_lr:.2e}")
                    # last_lr = cur_lr

                dsc_loss = float(losses["dsc_loss"].detach().cpu().numpy())
                gen_loss = float(losses["gen_loss"].detach().cpu().numpy())

                if(cur_epoch % log_step == 0):
                    self.net.logger.info(f"Epoch {int(cur_epoch/log_step)} : dsc_loss={round(dsc_loss, 3)}, gen_loss={round(gen_loss, 3)}")

                # self.vae_sch.step()
                # self.dsc_sch.step()
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
            self, data: List[torch.Tensor], graph_data: List[torch.Tensor], max_epochs: int = None,
            mini_batch: bool =False, iteration: int=1, dsc_k: int = None, cycle_key: List=[],
            early_stop : bool =False, train_ae : bool = True, log_step: int = 100, warmup: bool = False,
            early_stop_kwargs: dict={'gen_delta':1e-4, 'dsc_delta':2e-3, 'patience':400, 'verbose':False, 'step':50},
            warmup_kwargs : dict= {'warmup_epochs':None, 'base_lr':None, 'max_lr':2e-3,'step_size':50, 'gamma':0.8}
            
    ) -> Mapping[str, torch.Tensor]:
        
        # self.lam_align = self.lam_align * 2

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

                if(train_ae):
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
                    # cur_lr = self.vae_warmup.get_lr()
                    # if(cur_lr < last_lr):
                    #     self.net.logger.debug(f"Learning Rate adjusted to {last_lr:.2e}")
                    # last_lr = cur_lr

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
            
                    if(train_ae):
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
                    
                      
                # self.vae_sch.step()
                # self.dsc_sch.step()
                if(early_stop):
                    self.earlystop(val_loss=gen_loss, dsc_loss=dsc_loss)
                    if(self.earlystop.early_stop):
                        self.net.logger.info(f"Earlystop at epoch {int(cur_epoch/log_step)} : dsc_loss={round(dsc_loss, 3)}, " + \
                                             f"gen_loss={round(gen_loss, 3)}, " + \
                                             f"cycle_loss={round(cycle_loss, 3)}, align_loss={round(align_loss, 3)}")
                        break

            self.net.eval()

        # self.lam_align = self.lam_align * 0.5

    # def procrustes(self, A, B):
    #     """
    #     Find the best orthogonal matrix mapping using the Orthogonal Procrustes problem
    #     https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem
    #     """
    #     M = np.dot(A.T, B)
    #     U, S, V_t = np.linalg.svd(M)
    #     self.W = np.dot(U, V_t)

    # def findMNN(self, A, B, k=10, metric="euclidean"):
    #     if(metric=="euclidean"):
    #         dist_func = euclidean_distances
    #     elif(metric=="cosine"):
    #         dist_func = cosine_distances
    #     else:
    #         raise ValueError(f"Metric must be 'euclidean' or 'cosine'.")
    #     distance_matrix = dist_func(A, B)
    #     A_to_B = np.argmin(distance_matrix, axis=1)  
    #     B_to_A = np.argsort(distance_matrix, axis=0)[:k, :]
    #     mutual_pairs = []
    #     for i, j in enumerate(A_to_B):
    #         if i in B_to_A[:, j]:
    #             mutual_pairs.append((i, j))
    #     matched_A = A[[pair[0] for pair in mutual_pairs]]
    #     matched_B = B[[pair[1] for pair in mutual_pairs]]

    #     return mutual_pairs, matched_A, matched_B

    def save(
            self, path
        )-> None:
        torch.save({
        'model_state_dict': self.net.state_dict(),
        'vae_opt_state_dict': self.vae_optim.state_dict(),
        'dsc_opt_state_dict': self.dsc_optim.state_dict(),
        }, path)
        # self.net.logger.info(f"Model saved to '{path}'")
    
    def load(
            self, path
        )-> None:
        checkpoint = torch.load(path)
        self.net.load_state_dict(checkpoint['model_state_dict'])
        self.vae_optim.load_state_dict(checkpoint['vae_opt_state_dict'])
        self.dsc_optim.load_state_dict(checkpoint['dsc_opt_state_dict'])
        # self.net.logger.info(f"Model loaded from '{path}'")

    def __repr__(self):
        vae_optim = repr(self.vae_optim).replace("    ", "  ").replace("\n", "\n  ")
        dsc_optim = repr(self.dsc_optim).replace("    ", "  ").replace("\n", "\n  ")
        return (
            f"{type(self).__name__}(\n"
            f"  lam_kl: {self.lam_kl}\n"
            f"  lam_graph: {self.lam_graph}\n"
            f"  lam_adv: {self.lam_adv}\n"
            f"  lam_identy: {self.lam_iden}\n"
            f"  lam_align: {self.lam_align}\n"
            f"  lam_cycle: {self.lam_cycle}\n"
            f"  vae_optim: {vae_optim}\n"
            f"  dsc_optim: {dsc_optim}\n"
            f")"
        )
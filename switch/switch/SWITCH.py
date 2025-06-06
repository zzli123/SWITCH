import copy
import os
from typing import List, Mapping, Optional, Union
import scipy.sparse as sp

import networkx as nx
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from anndata import AnnData
from math import ceil

from .model import *
from .nn import SWITCH_nn
from .trainer import Trainer
import logging
from .utils import  extract_data, extract_graph, normalize_edges, eidx_to_adj, seed_everything
from .preprocess import check_graph


logger = logging.getLogger('logger')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter('- %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

DecoderDist = {"POIS" : PoisDataDecoder,
               "BER" : BerDataDecoder,
               "NB" : NBDataDecoder,
               "NORMAL" : NormalDataDecoder,
               "ZINB": ZINBDataDecoder,
               }
               

class SWITCH():
    """
    SWITCH model for spatial multi-modal data integration.

    Parameters:
    ----------
    adatas : Mapping[str, AnnData]
        A dictionary of AnnData objects that need to be integrated.

    vertices : List[str]
        The vertices (nodes) of the feature graph.

    latent_dim : int, optional (default=50)
        The dimensionality of the latent space.

    h_dim : int, optional (default=256)
        The dimensionality of the hidden layers.

    h_depth_enc : int, optional (default=1)
        The number of hidden layers in the encoder.

    h_depth_dsc : int, optional (default=None)
        The number of hidden layers in the discriminator.

    dropout : float, optional (default=0.1)
        The dropout rate.

    dsc_dropout : float, optional (default=None)
        The dropout rate for the discriminator.

    shared_batches : bool, optional (default=False)
        Whether to share batches across different modalities.

    conv_layer : str, optional (default="GAT")
        The type of convolution used in the encoder. Can be one of 'GCN', 'GAT', or 'LIN'.

    seed : int, optional (default=0)
        The random seed for reproducibility.

    normalize_methods : dict, optional (default={})
        A dictionary of normalization methods for different modalities.
        ** CAUTION **, Not implemented.
    """
    def __init__(
            self, adatas: Mapping[str, AnnData], vertices: List[str], 
            latent_dim: int = 50,
            h_dim: int = 256,
            h_depth_enc: int = 1,
            h_depth_dsc: int = None,
            dropout: float = 0.1,
            conv_layer: str = "GAT",
            seed: int=0,
            normalize_methods: dict={}
    ) -> None:
        logger.info(f"Set random seed to {seed}")

        seed_everything(seed)

        if(isinstance(conv_layer, dict)):
            for i in conv_layer.values():
                if(i not in ["GCN", "GAT", "LIN"]):
                    raise ValueError(
                        f"Graph conv layer must be `GCN`, `GAT`, or `LIN`."
                    )
        elif(conv_layer not in ["GCN", "GAT", "LIN"]):
            raise ValueError(
                f"Graph conv layer must be `GCN`, `GAT`, or `LIN`."
            )
        
        self.vertices = pd.Index(vertices)
        self.random_seed = seed

        g2v = GraphEncoder(self.vertices.size, latent_dim)
        v2g = GraphDecoder()
        self.modalities, idx, x2u, u2x, adj = {}, {}, {}, {}, {}

        for k, adata in adatas.items():
            
            if "SWITCH_config" not in adata.uns:
                raise ValueError(
                    f"The '{k}' dataset has not been configured. "
                    f"Please call `configure_dataset` first!"
                )
            data_config = copy.deepcopy(adata.uns["SWITCH_config"])

            if(data_config["use_spatial"]):
                if 'Spatial_Net' not in adata.uns.keys():
                    raise ValueError(
                        f"Spatial_Net is not existed! Run `Cal_Spatial_Net` first!"
                        )
                
                G_df = adata.uns['Spatial_Net'].copy()
                cells = np.array(adata.obs_names)
                cells_id_tran = dict(zip(cells, range(cells.shape[0])))
                G_df['Cell1'] = G_df['Cell1'].map(cells_id_tran)
                G_df['Cell2'] = G_df['Cell2'].map(cells_id_tran)
                G = sp.coo_matrix((np.ones(G_df.shape[0]), (G_df['Cell1'], G_df['Cell2'])), shape=(adata.n_obs, adata.n_obs))
                G = G + sp.eye(G.shape[0])
                edgeList = np.nonzero(G)
                edge_index = np.array([edgeList[0], edgeList[1]], dtype=int)
                adj[k] = eidx_to_adj(edge_index, adata.shape[0])
                adj[k] = torch.as_tensor(np.array(adj[k]))
            else:
                adj[k] = None

            idx[k] = self.vertices.get_indexer(data_config["features"]).astype(np.int64)
            if idx[k].min() < 0:
                raise ValueError("Not all modality features exist in the graph!")
            
            idx[k] = torch.as_tensor(idx[k])

            conv = conv_layer[k] if isinstance(conv_layer,dict) else conv_layer

            logger.debug(f"Use raw feature for the '{k}' dataset.")
            logger.debug(f"Set {h_depth_enc} {conv_layer} conv layer for '{k}' dataset.")
            x2u[k] = DataEncoder(
                len(data_config["features"]), latent_dim, conv=conv,
                h_depth=h_depth_enc, h_dim=h_dim, dropout=dropout
            )

            data_config["batches"] = pd.Index([]) if data_config["batches"] is None \
                else pd.Index(data_config["batches"])
            
            prob_model = data_config["prob_model"].upper()
            if(prob_model in DecoderDist.keys()):
                u2x[k] = DecoderDist[prob_model](len(data_config["features"]), n_batches=max(data_config["batches"].size, 1),)
            else:
                raise ValueError(
                    f"The {data_config['prob_model']} has not been implemented."
                )
            logger.debug(f"Use {prob_model} distribution for the '{k}' dataset.")
            self.modalities[k] = data_config

        h_depth_dsc = h_depth_dsc or h_depth_enc
        logger.debug(f"Set {h_depth_dsc} layers for the discriminator")
        du = Discriminator(
            latent_dim, len(self.modalities), n_batches=0,
            h_depth=h_depth_dsc , h_dim=h_dim, dropout=dropout
        )

        for key in idx.keys():
            if(not key in normalize_methods):
                normalize_methods[key] = "log"
                logger.debug(f"Use {normalize_methods[key]} normalize for the '{key}' dataset.")
            else:
                logger.debug(f"Use {normalize_methods[key]} normalize for the '{key}' dataset.")
                if(not normalize_methods[key] in ["log","clr"]):
                    raise ValueError("Nromalize methods must be `log` or `clr`.")
        
        prior = Prior()

        self._net = SWITCH_nn(g2v, v2g, x2u, u2x, idx, adj, du, prior, logger, normalize_methods)

        self._trainer = None

    def compile(
            self, 
            lam_data: float = 1.0,
            lam_graph: float = 0.2,
            lam_adv: float = 0.02,
            lam_cycle: float = 1.0,
            lam_align: float = 0.1,
            lam_kl: float = 1.0,            
            lr: float = 2e-4,
            **kwargs
    ) -> None:
        """
        Sets the hyperparameters for the SWITCH model.

        Parameters:
        ----------
        lam_data : float, optional (default=1.0)
            The weight for the data reconstruction loss.

        lam_graph : float, optional (default=0.2)
            The weight for the graph reconstruction loss.

        lam_adv : float, optional (default=0.02)
            The weight for the adversarial loss.

        lam_cycle : float, optional (default=1.0)
            The weight for the cycle mapping loss.

        lam_align : float, optional (default=0.1)
            The weight for the pseudo-pair alignment loss.

        lam_kl : float, optional (default=1.0)
            The weight for the KL divergence loss.

        lr : float, optional (default=2e-4)
            The learning rate.

        **kwargs
            Additional parameters to be passed to the trainer.
        """

        if self._trainer:
            self._net.logger.warning(
                "Overwritten previous trainer!"
            )
        self._trainer = Trainer(
            self._net, lam_kl=lam_kl,lam_graph=lam_graph, lam_adv=lam_adv, lam_data=lam_data, 
            lam_cycle=lam_cycle, lam_align=lam_align, optim="RMSprop", lr=lr, **kwargs
        )

    def pretrain(
            self,
            adatas: Mapping[str, AnnData],
            graph: nx.Graph,
            max_epochs: int=None,
            mini_batch: bool=False,
            iteration: int=1,
            dsc_k: int=1,
            warmup: bool=False,
            **kwargs
    ) -> None:
        """
        Pre-trains the SWITCH model without cycle mapping and pseudo-pair alignment losses.

        Parameters:
        ----------
        adatas : Mapping[str, AnnData]
            A dictionary of AnnData objects to be integrated.

        graph : nx.Graph
            A graph representing the feature structure of different modalities.

        max_epochs : int, optional
            The maximum number of training epochs.

        mini_batch : bool, optional (default=False)
            Whether to use mini-batch training.

        iteration : int, optional (default=1)
            The number of iterations per mini-batch.

        dsc_k : int, optional
            The number of discriminator updates per VAE update.

        warmup : bool, optional (default=False)
            Whether to perform warmup during pretraining.

        **kwargs
            Additional parameters to be passed to the training function.
        """

        data_configs = []
        adatas_ordered = []
        self._net.adj_weight = dict()
        self._net.logger.info(f"Prepare data for training.")
        for k in self._net.keys:
            data_config = copy.deepcopy(adatas[k].uns["SWITCH_config"])
            data_configs.append(data_config)
            adata = adatas[k].copy()
            adatas_ordered.append(adata)
        data = extract_data(adatas_ordered, data_configs)
        check_graph(
            graph, adatas.values(),
            cov="ignore", attr="error", loop="warn", sym="warn",
            verbose=False
        )
        graph_data = extract_graph(graph, self.vertices)

        data_size = sum([adata.shape[0] for adata in adatas.values()])
        if(max_epochs is None):
            if(warmup):
                max_epochs = min(1500, 1000 + int(data_size / 1e4) * 500)
            else:
                max_epochs = min(3000, 2000 + int(data_size / 1e4) * 1000)
        self._net.logger.debug(f"Set `dsc_k` = {dsc_k} for pretrain.")
        self._net.logger.debug(f"Set `max_epochs` = {max_epochs} for pretrain.")

        self._trainer.pretrain(data, graph_data, max_epochs=max_epochs, mini_batch=mini_batch,iteration=iteration, 
                               dsc_k=dsc_k, warmup=warmup, **kwargs)
        self._net.logger.info("Model pretrain done.")

    def train(
        self,
        adatas: Mapping[str, AnnData],
        graph: nx.Graph,
        max_epochs: int = None,
        mini_batch: bool=False,
        iteration: int =1,
        dsc_k: int = None, 
        cycle_key = "all",
        warmup: bool=False,
        **kwargs
    )-> None:
        """
        Trains the SWITCH model, including cycle mapping loss and pseudo-pair alignment loss.

        Parameters:
        ----------
        adatas : Mapping[str, AnnData]
            A dictionary of AnnData objects to be integrated.

        graph : nx.Graph
            A graph representing the features of different modalities.

        max_epochs : int, optional (default=None)
            The maximum number of epochs for training.

        mini_batch : bool, optional (default=False)
            Whether to use mini-batch training.

        iteration : int, optional (default=1)
            The number of iterations for mini-batch training.

        dsc_k : int, optional (default=None)
            The number of times the discriminator is trained for each VAE training step.

        cycle_key : str, optional (default="all")
            The key indicating which modality to use for calculating cycle mapping loss.

        warmup : bool, optional (default=False)
            Whether to perform warmup during training.

        **kwargs
            Additional parameters to be passed to the training function.
        """
        # if(not self._trainer.pretrained):
        #     raise ValueError("Model has not been pretrained, call `pretrain()` first.")
        
        data_configs = []
        adatas_ordered = []

        if(isinstance(cycle_key, str)):

            if(cycle_key.lower()=="all"):
                cycle_key = list(adatas.keys())

            elif(not cycle_key in adatas.keys()):
                raise ValueError(
                    f"Please set correct cycle key."
                    )
            else:
                cycle_key = [cycle_key]
        elif(isinstance(cycle_key, list)):
            for i in cycle_key:
                if(not i in adatas.keys()):
                    raise ValueError(
                        f"Please set correct cycle."
                        )
        else:
            raise ValueError(
                f"Cycle key must be str or list."
                )

        for k in self._net.keys:
            data_config = copy.deepcopy(adatas[k].uns["SWITCH_config"])
            data_configs.append(data_config)
            adata = adatas[k].copy()
            adatas_ordered.append(adata)
        data = extract_data(adatas_ordered, data_configs)
        check_graph(
            graph, adatas.values(),
            cov="ignore", attr="error", loop="warn", sym="warn",
            verbose=False
        )
        graph_data = extract_graph(graph, self.vertices)

        data_size = max([adata.shape[0] for adata in adatas.values()])
        if(max_epochs is None):
            if(warmup):
                max_epochs = min(500, 200 + int(data_size / 1e4) * 400)
            else:
                max_epochs = min(500, 200 + int(data_size / 1e4) * 1000)

        self._net.logger.debug(f"Set `dsc_k` = {dsc_k} for training.")
        self._net.logger.debug(f"Set `max_epochs` = {max_epochs} for training.")

        self._trainer.train(data, graph_data, max_epochs=max_epochs, mini_batch=mini_batch, 
                            iteration=iteration, dsc_k=dsc_k, cycle_key=cycle_key, warmup=warmup, **kwargs)
        self._net.logger.info("Model training done.")

    def save(
            self, path: str, overwrite: bool=False
    )-> None:
        """
        Saves the model to the specified path.

        Parameters:
        ----------
        path : str
            The path where the model will be saved.

        overwrite : bool, optional (default=False)
            Whether to overwrite the existing model at the specified path if it exists.
        """

        os_path = Path(path)
        if not os_path.parent.exists():
            raise ValueError(f"Directory does not exist: '{os_path.parent}'")
        elif(os_path.exists()):
            if(overwrite):
                logger.info(f"File '{os_path}' already exists and was overwritten.")
                self._trainer.save(path=path)
            else:
                logger.warning(f"File '{os_path}' already exists, set `overwrite=True` to overwrite.")
        else:
            self._trainer.save(path=path)
            logger.info(f"Model saved to '{path}'")

    def load(
            self, path: str
    ) -> None:
        """
        Loads the model from the specified path.

        Parameters:
        ----------
        path : str
            The path from which the model will be loaded.
        """
        if not os.path.exists(path):
            raise ValueError(f"File does not exists: '{path}'.")
        else:
            self._trainer.load(path)
            logger.info(f"Model loaded from '{path}'")

    @torch.no_grad()
    def encode_graph(
            self,
            graph: nx.Graph,
            sample: bool = False
    ) -> np.ndarray:
        """
        Compute graph (feature) embedding

        Parameters
        ----------
        graph
            Input graph

        sample
            Whether to sample from the embedding distribution,
            by default ``False``, returns the mean of the embedding distribution.

        Returns
        -------
        np.ndarray
            graph_embedding
        """
        self._net.eval()
        eidx, ewt, esgn = extract_graph(graph, self.vertices)
        enorm = torch.as_tensor(
            normalize_edges(eidx, ewt),
            device=self._net.device
        )
        eidx = torch.as_tensor(eidx, device=self._net.device)
        ewt = torch.as_tensor(ewt, device=self._net.device)
        esgn = torch.as_tensor(esgn, device=self._net.device)
        v = self._net.g2v(eidx, enorm, esgn)
        if sample:
            return v.sample().detach().cpu().numpy()
        return v.mean.detach().cpu().numpy()

    @torch.no_grad()
    def encode_data(
            self,
            key: str,
            adata: AnnData,
            sample: bool = False, 
            return_library_size: bool = False,
            return_attention_weights: bool = False
    ) -> np.ndarray:
        r"""
        Compute data (cell) embedding

        Parameters
        ----------
        key
            Modality key.

        adata
            Input dataset.

        sample
            Whether to sample from the embedding distribution,
            by default ``False``, returns the mean of the embedding distribution.

        return_library_size
            Whether to return the library size.
        
        return_attention_weights: bool, optional (default=False)
            Whether to return attention weights of data.

        Returns
        -------
        np.ndarray
            data_embedding
        """
        self._net.eval()
        encoder = self._net.x2u[key]
        data = extract_data([adata],[adata.uns["SWITCH_config"]])
        x = data[0]
        edge_index = data[-2]
       
        result = []
        if(return_attention_weights):
            u, l, att = encoder(
            x=torch.as_tensor(x, device=self._net.device),
            edge_index=torch.as_tensor(edge_index, device=self._net.device),
            return_attention_weights=return_attention_weights)
        else:
            u, l = encoder(
                x=torch.as_tensor(x, device=self._net.device),
                edge_index=torch.as_tensor(edge_index, device=self._net.device),
            )
        if sample:
            result.append(u.sample().detach().cpu())
        else:
            result.append(u.mean.detach().cpu())

        if(return_library_size and return_attention_weights):
            return torch.cat(result).numpy(), l, att
        elif(return_library_size):
            return torch.cat(result).numpy(), l
        elif(return_attention_weights):
            return torch.cat(result).numpy(), att
        else:
            return torch.cat(result).numpy()

    @torch.no_grad()
    def decode_data(
            self,
            source_key: str,
            target_key: str,
            adata: AnnData,
            graph: nx.Graph,
            sample: bool = False,
            target_libsize: Optional[Union[float, np.ndarray]] = None,
            target_batch: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Decode data

        Parameters
        ----------
        source_key
            Source modality key

        target_key
            Target modality key

        adata
            Source modality data

        graph
            Guidance graph

        sample
            Whether to sample from the decoder distribution,

        target_libsize
            Target modality library size, by default 1.0

        target_batch
            Target modality batch, by default batch 0

        Returns
        -------
        np.ndarray
            Decoded data

        """
        l = target_libsize or 1.0
        if not isinstance(l, np.ndarray):
            l = np.asarray(l)
        l = l.squeeze()
        if l.ndim == 0:  # Scalar
            l = l[np.newaxis]
        elif l.ndim > 1:
            raise ValueError("`target_libsize` cannot be >1 dimensional")
        if l.size == 1:
            l = np.repeat(l, adata.shape[0])
        if l.size != adata.shape[0]:
            raise ValueError("`target_libsize` must have the same size as `adata`!")
        l = l.reshape((-1, 1))

        use_batch = self.modalities[target_key]["use_batch"]
        batches = self.modalities[target_key]["batches"]
        if use_batch and target_batch is not None:
            target_batch = np.asarray(target_batch)
            if target_batch.size != adata.shape[0]:
                raise ValueError("`target_batch` must have the same size as `adata`!")
            b = batches.get_indexer(target_batch)
        else:
            b = np.zeros(adata.shape[0], dtype=int)

        net = self.net
        device = net.device
        net.eval()

        u = self.encode_data(source_key, adata, sample=sample)
        v = self.encode_graph(graph, sample=sample)
        v = torch.as_tensor(v, device=device)
        v = v[getattr(net, f"{target_key}_idx")]

        decoder = net.u2x[target_key]

        result = []
        u = torch.as_tensor(u,device=self.net.device)
        b = torch.as_tensor(b, device=self.net.device)
        l = torch.as_tensor(l, device=self.net.device)
        result.append(decoder(u, v, b, l).mean.detach().cpu())
        return torch.cat(result).numpy()
    
    @torch.no_grad()
    def impute_data(
            self,
            source_key: str,
            target_key: str, 
            source_adata: AnnData,
            target_adata: AnnData,
            graph: nx.Graph, sample: bool=False,
            target_libsize: Optional[Union[float, np.ndarray]]=None,
            target_batch: Optional[np.ndarray] = None,
        ):
        """
        Impute data

        Parameters
        ----------
        source_key
            Source modality key

        target_key
            Target modality key

        source adata
            Source modality data

        target_adata
            Target modality data

        graph
            Guidance graph

        target_libsize
            Target modality library size, by default 1.0

        target_batch
            Target modality batch, by default batch 0

        Returns
        -------
        np.ndarray
            Imputed data

        """

        self._net.eval()
        decoder = self._net.u2x[target_key]
        l = target_libsize or 1.0
        use_batch = self.modalities[target_key]["use_batch"]
        batches = self.modalities[target_key]["batches"]
        if use_batch and target_batch is not None:
            target_batch = np.asarray(target_batch)
            if target_batch.size != target_adata.shape[0]:
                raise ValueError("`target_batch` must have the same size as `adata`!")
            b = batches.get_indexer(target_batch)
        else:
            b = np.zeros(source_adata.shape[0], dtype=int)
        
        b = torch.as_tensor(b, device=self._net.device)
        
        u, _ = self.encode_data(source_key, source_adata, sample=sample, return_library_size=True)        

        l = torch.as_tensor(l, device=self._net.device)

        v = self.encode_graph(graph, sample=sample)
        v = torch.as_tensor(v, device=self._net.device)
        v = v[getattr(self._net, f"{target_key}_idx")]
        u = torch.as_tensor(u,device=self._net.device)
        fake_data = decoder(u, v, b, l)[0].mean.detach().cpu()

        return fake_data
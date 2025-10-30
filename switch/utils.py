import os
import random
from typing import Any, Mapping, Optional, List
import networkx as nx
from anndata import AnnData
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

DATA_CONFIG = Mapping[str, Any]

default_dtype = getattr(np, str(torch.get_default_dtype()).replace("torch.", ""))

def extract_data(
        adatas: List[AnnData],
        data_configs: List[DATA_CONFIG]
) -> List[torch.Tensor]:
    """
    Extracts the data required for training from the provided AnnData objects.

    Parameters:
    ----------
    adatas : List[AnnData]
        A list of AnnData objects containing the data to be extracted.

    data_configs : List[DATA_CONFIG]
        A list of configurations specifying how to extract the data.

    Returns:
    -------
    List[torch.Tensor]
        A list of tensors containing the extracted data ready for training.
    """
        
    def _extract_x(adata: AnnData, data_config: DATA_CONFIG) -> torch.Tensor:
        """
        Extracts the data matrix from the given AnnData object based on the provided configuration.

        Parameters:
        ----------
        adata : AnnData
            The AnnData object from which the matrix is to be extracted.

        data_config : DATA_CONFIG
            The configuration specifying how to extract the data.

        Returns:
        -------
        torch.Tensor
            The extracted matrix as a tensor.
        """
        features = data_config["features"]
        use_layer = data_config["use_layer"]
        if not np.array_equal(adata.var_names, features):
            adata = adata[:, features]  # This will load all data to memory if backed
        if use_layer:
            if use_layer not in adata.layers:
                raise ValueError(
                    f"Configured data layer '{use_layer}' "
                    f"cannot be found in input data!"
                )
            x = adata.layers[use_layer]
        else:
            x = adata.X.copy()
        if sp.issparse(x):
            x = x.todense()
        if x.dtype.type is not default_dtype:
            x = x.astype(default_dtype)    
        if(data_config["prob_model"].upper()=="BER"):
            x[x>1] = 1
        return x

    def _extract_xnet(adata: AnnData, data_config: DATA_CONFIG) -> torch.Tensor:
        """
        Extracts the spatial adjacency graph from the given AnnData object based on the provided configuration.

        Parameters:
        ----------
        adata : AnnData
            The AnnData object from which the spatial adjacency graph is to be extracted.

        data_config : DATA_CONFIG
            The configuration specifying how to extract the spatial adjacency graph.

        Returns:
        -------
        torch.Tensor
            The extracted spatial adjacency graph as a tensor.
        """

        if( not data_config["use_spatial"]):
            n_spots = len(adata.obs_names)
            G = np.eye(n_spots)
            edgeList = np.nonzero(G)
            edge_index = np.array([edgeList[0], edgeList[1]], dtype=int)
            edge_type = np.array([0] * n_spots)
            return (edge_index, edge_type)

        if 'Spatial_Net' not in adata.uns.keys():
            raise ValueError("Spatial_Net is not existed! Run Cal_Spatial_Net first!")
        G_df = adata.uns['Spatial_Net'].copy()
        cells = np.array(adata.obs_names)
        cells_id = dict(zip(cells, range(cells.shape[0])))
        G_df['Cell1'] = G_df['Cell1'].map(cells_id)
        G_df['Cell2'] = G_df['Cell2'].map(cells_id)


        if(not "edge_type" in G_df.columns):
            G_df["edge_type"] = "0"

        self_loops = pd.DataFrame({'Cell1': np.arange(len(cells)), 'Cell2': np.arange(len(cells)), 'Distance':0,'edge_type':'0'})
        G_df = pd.concat([G_df, self_loops], ignore_index=True)
        G_df.drop_duplicates(inplace=True, subset=["Cell1","Cell2"])
        G_df.sort_values(by=['Cell1', 'Cell2'], inplace=True)

        # G = sp.coo_matrix((np.ones(G_df.shape[0]), (G_df['Cell1'], G_df['Cell2'])), shape=(adata.n_obs, adata.n_obs))
        # G = G + sp.eye(G.shape[0])
        # edgeList = np.nonzero(G)
        # edge_index = np.array([edgeList[0], edgeList[1]], dtype=int)

        edge_index = np.array([list(G_df['Cell1']), G_df["Cell2"]], dtype=int)
        edge_type = None
        # if(data_config["use_multi_net"]):
        #     edge_type = np.array(list(G_df["edge_type"]),dtype=int)
        # else:
        #     edge_type = None

        return (edge_index, edge_type)

    x = [
        _extract_x(adata, data_config)
        for adata, data_config in zip(adatas, data_configs)
    ]
    xnet, edge_type = zip(*[
        _extract_xnet(adata, data_config)
        for adata, data_config in zip(adatas, data_configs)
    ])
    xnet = list(xnet)
    edge_type = list(edge_type)
    return x + xnet + edge_type

def extract_graph(
        graph: nx.Graph,
        vertices: pd.Index
) -> List[np.array]:
    """
    Extracts graph data from the given feature graph.

    Parameters:
    ----------
    graph : nx.Graph
        The graph from which the graph data will be extracted.

    vertices : pd.Index
        The vertices (nodes) of the graph for which the graph data is to be extracted.

    Returns:
    -------
    List[np.array]
        A list of graph data arrays extracted from the graph.
    """
    graph = nx.MultiDiGraph(graph)  # Convert undirecitonal to bidirectional, while keeping multi-edges

    default_dtype = getattr(np, str(torch.get_default_dtype()).replace("torch.", ""))
    i, j, w, s = [], [], [], []
    for k, v in dict(graph.edges).items():
        i.append(k[0])
        j.append(k[1])
        w.append(v["weight"])
        s.append(v["sign"])
    eidx = np.stack([
        vertices.get_indexer(i),
        vertices.get_indexer(j)
    ]).astype(np.int64)
    if eidx.min() < 0:
        raise ValueError("Missing vertices!")
    ewt = np.asarray(w).astype(default_dtype)
    if ewt.min() <= 0 or ewt.max() > 1:
        raise ValueError("Invalid edge weight!")
    esgn = np.asarray(s).astype(default_dtype)
    if set(esgn).difference({-1, 1}):
        raise ValueError("Invalid edge sign!")
    
    return eidx, ewt, esgn

def vertex_degrees(
        eidx: np.ndarray,
        ewt: np.ndarray,
        vnum: Optional[int]=None,
        direction: str = "both"
) -> np.ndarray:
    r"""
    Compute vertex degrees

    Parameters
    ----------
    eidx
        Vertex indices of edges (:math:`2 \times n_{edges}`)
    ewt
        Weight of edges (:math:`n_{edges}`)
    vnum
        Total number of vertices (determined by max edge index if not specified)
    direction
        Direction of vertex degree, should be one of {"in", "out", "both"}

    Returns
    -------
    degrees
        Vertex degrees
    """
    vnum = vnum or eidx.max() + 1
    adj = sp.coo_matrix((ewt, (eidx[0], eidx[1])), shape=(vnum, vnum))
    if direction == "in":
        return adj.sum(axis=0).A1
    elif direction == "out":
        return adj.sum(axis=1).A1
    elif direction == "both":
        return adj.sum(axis=0).A1 + adj.sum(axis=1).A1 - adj.diagonal()
    raise ValueError("Unrecognized direction!")

def normalize_edges(
        eidx: np.ndarray, 
        ewt: np.ndarray,
        method: str = "keepvar"
) -> np.ndarray:
    r"""
    Normalize graph edge weights

    Parameters
    ----------
    eidx
        Vertex indices of edges (:math:`2 \times n_{edges}`)
    ewt
        Weight of edges (:math:`n_{edges}`)
    method
        Normalization method, should be one of {"in", "out", "sym", "keepvar"}

    Returns
    -------
    enorm
        Normalized weight of edges (:math:`n_{edges}`)
    """
    if method not in ("in", "out", "sym", "keepvar"):
        raise ValueError("Unrecognized method!")
    enorm = ewt
    if method in ("in", "keepvar", "sym"):
        in_degrees = vertex_degrees(eidx, ewt, direction="in")
        in_normalizer = np.power(
            in_degrees[eidx[1]],
            -1 if method == "in" else -0.5
        )
        in_normalizer[~np.isfinite(in_normalizer)] = 0  # In case there are unconnected vertices
        enorm = enorm * in_normalizer
    if method in ("out", "sym"):
        out_degrees = vertex_degrees(eidx, ewt, direction="out")
        out_normalizer = np.power(
            out_degrees[eidx[0]],
            -1 if method == "out" else -0.5
        )
        out_normalizer[~np.isfinite(out_normalizer)] = 0  # In case there are unconnected vertices
        enorm = enorm * out_normalizer
    return enorm

def seed_everything(seed):
    """
    Sets the random seed for reproducibility.

    Parameters:
    ----------
    seed : int
    The random seed to be set for all random number generators (e.g., numpy, torch, random).
    """
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

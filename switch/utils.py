import os
from collections import Counter
import random
from typing import Any, Mapping, Optional, List
import networkx as nx
from anndata import AnnData
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from sklearn.preprocessing import normalize


DATA_CONFIG = Mapping[str, Any]

default_dtype = getattr(np, str(torch.get_default_dtype()).replace("torch.", ""))

def extract_data(
        adatas: List[AnnData], data_configs: List[DATA_CONFIG]
    ) -> List[torch.Tensor]:
        
        def _extract_x(adata: AnnData, data_config: DATA_CONFIG) -> torch.Tensor:
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

        def _extract_xbch(adata: AnnData, data_config: DATA_CONFIG) -> torch.Tensor:
            use_batch = data_config["use_batch"]
            batches = data_config["batches"]
            if use_batch:
                if use_batch not in adata.obs:
                    raise ValueError(
                        f"Configured data batch '{use_batch}' "
                        f"cannot be found in input data!"
                    )
                return batches.get_indexer(adata.obs[use_batch])
            return np.zeros(adata.shape[0], dtype=int)

        def _extract_xnet(adata: AnnData, data_config: DATA_CONFIG) -> torch.Tensor:

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
        xbch = [
            _extract_xbch(adata, data_config)
            for adata, data_config in zip(adatas, data_configs)
        ]
        xnet, edge_type = zip(*[
            _extract_xnet(adata, data_config)
            for adata, data_config in zip(adatas, data_configs)
        ])
        xnet = list(xnet)
        edge_type = list(edge_type)
        return x + xbch + xnet + edge_type

def extract_graph(
        graph: nx.Graph, vertices: pd.Index
    ) -> List[np.array]:
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
        eidx: np.ndarray, ewt: np.ndarray,
        vnum: Optional[int] = None, direction: str = "both"
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
        eidx: np.ndarray, ewt: np.ndarray, method: str = "keepvar"
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

def eidx_to_adj(edge_index: np.array, num_nodes: int) -> np.array:

    """
    Convert an edge_index representation to an adjacency matrix.

    Parameters:
    edge_index (torch.Tensor): The edge_index tensor of shape [2, num_edges].
    num_nodes (int): The number of nodes in the graph.

    Returns:
    torch.Tensor: The adjacency matrix of shape [num_nodes, num_nodes].
    """
    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=np.int8)

    # Fill the adjacency matrix
    adj_matrix[edge_index[0], edge_index[1]] = 1

    adj_matrix = adj_matrix+ sp.eye(num_nodes, num_nodes)

    return adj_matrix

def tfidf(X):
    r"""
    TF-IDF normalization (following the Seurat v3 approach)

    Parameters
    ----------
    X
        Input matrix

    Returns
    -------
    X_tfidf
        TF-IDF normalized matrix
    """
    idf = X.shape[0] / X.sum(axis=0)
    if sp.issparse(X):
        tf = X.multiply(1 / X.sum(axis=1))
        return tf.multiply(idf)
    else:
        tf = X / X.sum(axis=1, keepdims=True)
        return tf * idf

def lsi(
        adata: AnnData, n_components: int = 20,
        use_highly_variable: Optional[bool] = None, **kwargs
) -> None:
    r"""
    LSI analysis (following the Seurat v3 approach)

    Parameters
    ----------
    adata
        Input dataset
    n_components
        Number of dimensions to use
    use_highly_variable
        Whether to use highly variable features only, stored in
        ``adata.var['highly_variable']``. By default uses them if they
        have been determined beforehand.
    **kwargs
        Additional keyword arguments are passed to
        :func:`sklearn.utils.extmath.randomized_svd`
    """
    if "random_state" not in kwargs:
        kwargs["random_state"] = 0  # Keep deterministic as the default behavior
    if use_highly_variable is None:
        use_highly_variable = "highly_variable" in adata.var
    adata_use = adata[:, adata.var["highly_variable"]] if use_highly_variable else adata
    X = tfidf(adata_use.X)
    X_norm = normalize(X, norm="l1")
    X_norm = np.log1p(X_norm * 1e4)
    X_lsi = sklearn.utils.extmath.randomized_svd(X_norm, n_components, **kwargs)[0]
    X_lsi -= X_lsi.mean(axis=1, keepdims=True)
    X_lsi /= X_lsi.std(axis=1, ddof=1, keepdims=True)
    adata.obsm["X_lsi"] = X_lsi

def seed_everything(seed):
   torch.manual_seed(seed)
   os.environ['PYTHONHASHSEED'] = str(seed)
   torch.cuda.manual_seed(seed)
   torch.cuda.manual_seed_all(seed)
   np.random.seed(seed)
   random.seed(seed)
#    torch.backends.cudnn.benchmark = False
#    torch.backends.cudnn.deterministic = True
#    torch.backends.cudnn.enabled = True

def refine_labels(adata, label_key, copy=False):
    assert ("Spatial_Net" in adata.uns)
    
    labels = list(adata.obs[label_key])
    cells = np.array(adata.obs_names)
    cells_id_tran = dict(zip(cells, range(cells.shape[0])))
    G_df = adata.uns['Spatial_Net'].copy()
    G_df['Cell1'] = G_df['Cell1'].map(cells_id_tran)
    G_df['Cell2'] = G_df['Cell2'].map(cells_id_tran)
    edge_index = [list(G_df["Cell1"]),list(G_df["Cell2"])]
    neighbors = [[] for _ in range(len(labels))]
    for start, end in zip(edge_index[0], edge_index[1]):
        neighbors[start].append(end)

    new_labels = labels[:]

    for i in range(len(labels)):
        if neighbors[i]:  # 如果该节点有邻居
            neighbor_labels = [labels[n] for n in neighbors[i]]
            label_count = Counter(neighbor_labels)
            most_common_label, most_common_count = label_count.most_common(1)[0]
            
            # 判断是否需要更改标签
            if labels[i] != most_common_label and most_common_count > len(neighbor_labels) / 2:
                new_labels[i] = most_common_label
    
    adata.obs["refined_labels"] = new_labels
    
    return adata.copy() if copy else None

def mclust_R(adata, n_cluster, use_rep, modelNames='EEE', random_seed=0, key_added="mclust"):
    """\
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """
    
    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']
    
    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[use_rep]), n_cluster, modelNames)
    mclust_res = np.array(res[-2])

    adata.obs[key_added] = mclust_res
    adata.obs[key_added] = adata.obs[key_added].astype('int')
    adata.obs[key_added] = adata.obs[key_added].astype('category')
    return adata

from typing import List, Optional
from anndata import AnnData
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from scipy.stats import rankdata
import scanpy as sc
from tqdm import tqdm
from ..preprocess._preprocess import rna_anchored_guidance_graph
import scipy
import numpy as np

def foscttm(
        x: np.ndarray, y: np.ndarray, **kwargs
):
    if x.shape != y.shape:
        raise ValueError("Shapes do not match!")
    d = scipy.spatial.distance_matrix(x, y, **kwargs)
    n1, _ = d.shape
    true_matching = np.arange(n1)
    mask = (d.T < d[np.arange(n1), true_matching]).T
    return np.mean(np.mean(mask, axis=1))

def integration_score(
        adatas: List[AnnData],
        use_rep: str="SWITCH",
        distance: int=2000,
        use_layer: Optional[str]=None,
        resolution: int=30,
        random_seed: int=0,
        corr: str="pcc",
        metric: str="cosine",
        subsample_size: float=None,
        scale: bool=True,
        corrupt_rate: float=None,
        eval_on_paired_data: bool=False,
        verbose: bool=False,
        n_perm: int=0,
        **kwargs
) -> float:
    """
    Computes the integration quality score for the given dataset.

    Parameters:
    ----------
    adatas : List[AnnData]
        The list of AnnData objects to be used for computing the integration score.

    use_rep : str, optional (default="SWITCH")
        The representation from adata.obsm to aggregate cells into meta-cells.

    distance : int, optional (default=5e5)
        Distance threshold for high-confidence gene-peak pairs.

    use_layer : Optional[str], optional
        The layer from adata.layer to compute the correlation coefficient.

    resolution : int, optional
        The resolution used to generate meta cell.

    random_seed : int, optional (default=0)
        The random seed for reproducibility.

    corr : str, optional (default="pcc")
        The method to compute correlation. Can be "pcc" (Pearson correlation) or "spr" (Spearman correlation).

    metric : str, optional (default="cosine")
        The distance metric to use for computation.

    subsample_size : float, optional
        The proportion of features to subsample.

    scale : bool, optional (default=True)
        Whether to scale the enrichment scores.

    corrupt_rate : float, optional
        *** DO NOT USE ***
        The proportion of the pairing relationships to be corrupted, only used for evaluate.

    eval_on_paired_data : bool, optional (default=False)
        *** DO NOT USE ***
        Whether to evaluate using true paired data, only used for evaluate.
    
    verbose : bool, optional (default=False)
        If True, prints progress messages.
    
    n_perm : int, optional (default=0)
        Number of permutations for normalization.

    **kwargs
        Additional arguments to be passed to the function.

    Returns:
    -------
    float
        The computed integration quality score.
    """
    
    np.random.seed(random_seed)
    assert len(adatas)==2
    assert corr in ["pcc", "spr"]
    for ad in adatas:
        assert use_rep in ad.obsm

    # n_meta_cell = n_meta_cell or int((adatas[0].shape[0] + adatas[1].shape[0])/30)
    
    def _fast_pearson(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
        """
        Computes the Pearson correlation between columns of two DataFrames.

        Parameters:
        ----------
        df1 : pd.DataFrame
            The first DataFrame.

        df2 : pd.DataFrame
            The second DataFrame.

        Returns:
        -------
        pd.DataFrame
            A DataFrame containing the Pearson correlation between corresponding columns of df1 and df2.
        """
        assert df1.shape[0] == df2.shape[0]

        X = df1.to_numpy()
        Y = df2.to_numpy()

        X_std = X.std(axis=0)
        Y_std = Y.std(axis=0)

        X_std[X_std == 0] = 1
        Y_std[Y_std == 0] = 1

        X_center = (X - X.mean(axis=0)) / X_std
        Y_center = (Y - Y.mean(axis=0)) / Y_std

        corr = np.dot(X_center.T, Y_center) / X.shape[0]

        return pd.DataFrame(corr, index=df1.columns, columns=df2.columns)
    
    def _fast_spearman(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
        """
        Computes the Spearman correlation between columns of two DataFrames.

        Parameters:
        ----------
        df1 : pd.DataFrame
            The first DataFrame.

        df2 : pd.DataFrame
            The second DataFrame.

        Returns:
        -------
        pd.DataFrame
            A DataFrame containing the Spearman correlation between corresponding columns of df1 and df2.
        """
        assert df1.shape[0] == df2.shape[0]
        X = df1.to_numpy()
        Y = df2.to_numpy()
        
        X_ranked = np.apply_along_axis(rankdata, axis=0, arr=X)
        Y_ranked = np.apply_along_axis(rankdata, axis=0, arr=Y)

        X_ranked = (X_ranked - X_ranked.mean(axis=0)) / X_ranked.std(axis=0)
        Y_ranked = (Y_ranked - Y_ranked.mean(axis=0)) / Y_ranked.std(axis=0)

        corr_matrix = np.dot(X_ranked.T, Y_ranked) / X.shape[0]

        return pd.DataFrame(corr_matrix, index=df1.columns, columns=df2.columns)

    def _aggregate(expr_matrix: pd.DataFrame, cell_labels: pd.Series, func: str="mean"
    ) -> pd.DataFrame:
        """
        Aggregates the expression matrix into meta-cell matrix based on cell labels.

        Parameters:
        ----------
        expr_matrix : pd.DataFrame
            The expression matrix to be aggregated.

        cell_labels : pd.Series
            The labels used to group cells into meta-cells.

        func : str, optional (default="mean")
            The aggregation function. Can be "mean" or "sum".

        Returns:
        -------
        pd.DataFrame
            The aggregated meta-cell expression matrix.
        """

        unique_labels = cell_labels.unique()
        gene_names = expr_matrix.columns
        label_map = {label: i for i, label in enumerate(unique_labels)}

        label_indices = cell_labels.map(label_map).to_numpy()
        data = expr_matrix.to_numpy()

        meta_expr = np.zeros((len(unique_labels), data.shape[1]))
        meta_counts = np.zeros(len(unique_labels))

        for i in range(len(data)):
            meta_expr[label_indices[i]] += data[i]
            meta_counts[label_indices[i]] += 1

        if(func!="sum"):
            meta_expr /= meta_counts[:, np.newaxis]

        meta_df = pd.DataFrame(meta_expr, index=unique_labels, columns=gene_names)
        return meta_df

    def _enrichment_score(df_matrix: pd.DataFrame, true_pairs: set, n_perm: int = 100) -> float:
        """
        Computes the enrichment score based on the given data matrix and true pairs.

        Parameters:
        ----------
        df_matrix : pd.DataFrame
            The data matrix used for computing the enrichment score.

        true_pairs : set
            A set of true pairs used to compute the enrichment score.
        
        n_perm : int, optional (default=100)
            Number of permutations for normalization.

        Returns:
        -------
        float
            The computed enrichment score.
        """
        gene_names = np.array(df_matrix.index)
        peak_names = np.array(df_matrix.columns)
        corr_values = df_matrix.to_numpy().ravel()
        
        gene_ids = np.repeat(gene_names, len(peak_names))
        peak_ids = np.tile(peak_names, len(gene_names))

        pair_labels = pd.Series(list(zip(gene_ids, peak_ids)))
        true_mask = pair_labels.isin(true_pairs).to_numpy(dtype=int)

        order = np.argsort(-corr_values)
        hits_sorted = true_mask[order]

        N_hit = hits_sorted.sum()
        N_total = len(hits_sorted)
        N_miss = N_total - N_hit
        if N_hit == 0 or N_hit == N_total:
            raise ValueError("No true pairs found in input.")
        else:
            increment_hit = hits_sorted / N_hit
            increment_miss = (1 - hits_sorted) / N_miss
            running_sum = np.cumsum(increment_hit - increment_miss)
            enrichment_score = np.max(np.abs(running_sum))
        
        if(n_perm > 0):
            es_perm = []
            for _ in tqdm(range(n_perm), desc='Permutation:'):
                np.random.shuffle(hits_sorted)
                inc_hit_perm = hits_sorted / N_hit
                inc_miss_perm = (1 - hits_sorted) / N_miss
                rs_perm = np.cumsum(inc_hit_perm - inc_miss_perm)
                es_perm.append(np.max(np.abs(rs_perm)))
            
            es_perm = np.array(es_perm)
            es_mean = np.mean(np.abs(es_perm)) + 1e-8 
            enrichment_score = enrichment_score / (es_mean * 35) # prevent excessive es

        return enrichment_score
    
    def _shuffle(lst: List, proportion: float) -> None:
        """
        Randomly shuffles a portion of the list.

        Parameters:
        ----------
        lst : List
            The list to be shuffled.

        proportion : float
            The proportion of the list to shuffle, between 0 and 1.
        """
        if not (0 <= proportion <= 1):
            raise ValueError("")

        n = len(lst)
        num_to_shuffle = int(n * proportion)

        indices = list(range(n))
        shuffle_indices = random.sample(indices, num_to_shuffle)

        elements_to_shuffle = [lst[i] for i in shuffle_indices]
        random.shuffle(elements_to_shuffle)

        new_lst = lst.copy()
        for idx, new_val in zip(shuffle_indices, elements_to_shuffle):
            new_lst[idx] = new_val

        return new_lst

    corr_func = _fast_pearson if corr!="spr" else _fast_spearman

    # embed = np.concatenate([adatas[0].obsm[use_rep], adatas[1].obsm[use_rep]], axis=0)
    # if(metric=="cosine"):
    #     embed = normalize(embed, norm="l2")
    
    mat1 = adatas[0].X.copy() if use_layer is None else adatas[0].layers[use_layer].copy()
    mat2 = adatas[1].X.copy() if use_layer is None else adatas[1].layers[use_layer].copy()

    mat1 = pd.DataFrame(mat1.todense()) if sp.issparse(mat1) else pd.DataFrame(mat1)
    mat2 = pd.DataFrame(mat2.todense()) if sp.issparse(mat2) else pd.DataFrame(mat2)

    mat1.columns = adatas[0].var_names
    mat2.columns = adatas[1].var_names

    if(subsample_size is not None):
        assert subsample_size>=0 and subsample_size <=1.0
        mat1_idx = np.random.choice(mat1.shape[1], int(mat1.shape[1]*subsample_size), replace=False)
        mat2_idx = np.random.choice(mat2.shape[1], int(mat2.shape[1]*subsample_size), replace=False)
        mat1 = mat1.iloc[:, mat1_idx]
        mat2 = mat2.iloc[:, mat2_idx]

    adatas[0] = adatas[0][:,mat1.columns]
    adatas[1] = adatas[1][:,mat2.columns]

    if(adatas[0].var_names[0][0:3]=="chr"):
        graph = rna_anchored_guidance_graph(adatas[1], adatas[0], promoter_len=distance, propagate_highly_variable=False)
    else:
        graph = rna_anchored_guidance_graph(adatas[0], adatas[1], promoter_len=distance, propagate_highly_variable=False)
    
    true_pairs = [(i[0], i[1]) for i in list(graph.edges)]
    true_pairs = set(filter(lambda x: x[0] in mat1.columns and x[1] in mat2.columns, true_pairs))
    true_pair_rate = (mat1.shape[1] * mat2.shape[1]) / len(true_pairs)
    scale_factor = true_pair_rate / 5e4
    if(verbose):
        print(f"Scale factor: {scale_factor}")
    
    concat = sc.concat([adatas[0], adatas[1]])
    sc.pp.neighbors(concat, use_rep=use_rep, n_neighbors=15, metric=metric)
    sc.tl.leiden(concat, resolution=resolution, key_added="temp_label")
    n_meta_cell = len(set(concat.obs["temp_label"]))
    label = list(concat.obs["temp_label"])

    if(verbose):
        print(f"Num of meta cell: {n_meta_cell}")

    if(eval_on_paired_data):
        print(f"- WARNING - `eval_on_paired_data` is only used for evaluate.")
        # n_meta_cell = 200
        # km = KMeans(n_clusters=n_meta_cell, random_state=0, n_init="auto").fit(np.array(adatas[0].obsm[use_rep]))
        # label = list(km.labels_)
        meta1 = _aggregate(mat1, pd.Series(label[0:adatas[0].shape[0]]))
        meta2 = _aggregate(mat2, pd.Series(label[0:adatas[0].shape[0]]))
        delta = 1
    else:
        # km = KMeans(n_clusters=n_meta_cell, random_state=0, n_init="auto").fit(np.array(embed))
        # label = list(km.labels_)
        # meta1 = _aggregate(mat1, pd.Series(label[0:adatas[0].shape[0]]))
        # meta2 = _aggregate(mat2, pd.Series(label[adatas[0].shape[0]:]))

        meta1 = _aggregate(mat1, pd.Series(label[0:adatas[0].shape[0]]))
        meta2 = _aggregate(mat2, pd.Series(label[adatas[0].shape[0]:]))

        comm_label = set(meta1.index) & set(meta2.index)
        # print(len(comm_label))
        if(len(comm_label) < 2):
            return -1
        meta1 = meta1.reindex(comm_label)
        meta2 = meta2.reindex(comm_label)
        delta = len(comm_label) / n_meta_cell

    if(verbose):
        print(f"Delta: {delta}")

    if(corrupt_rate is not None and corrupt_rate > 0):
        print(f"- WARNING - `corrupt_rate` is only used for evaluate.")
        assert corrupt_rate >= 0 and corrupt_rate <= 1
        shuffled_idx = _shuffle(list(meta1.index), corrupt_rate)
        meta2 = meta2.loc[shuffled_idx,] 

    corr_df = corr_func(meta1, meta2)
    es = _enrichment_score(corr_df, true_pairs, n_perm=n_perm, **kwargs)
    es = es / scale_factor if scale else es

    return es * delta
import os
import collections
from itertools import chain
from typing import Any, Callable, Iterable, Mapping, Optional, Set, List
import networkx as nx
from anndata import AnnData
import pybedtools
from pybedtools import BedTool
from pybedtools.cbedtools import Interval
import numpy as np
import pandas as pd
import scipy.sparse as sp
import sklearn.neighbors
import re
from tqdm.auto import tqdm

def Cal_Spatial_Net(adata, use_rep="spatial", rad_cutoff=None, k_cutoff=None, 
                    model='Radius',  verbose=True, copy=False):
    """
    Construct the spatial neighbor networks.

    Parameters
    ----------
    adata
        AnnData object of scanpy package.
    use_rep
        Use the indicated representation in .obsm to construct the neighbor networks.
    rad_cutoff
        radius cutoff when model='Radius'
    k_cutoff
        The number of nearest neighbors when model='KNN'
    model
        The network construction model. When model=='Radius', the spot is connected to spots whose 
        distance is less than rad_cutoff. When model=='KNN', the spot is connected to its first k_cutoff nearest neighbors.
    verbose
        Whether to output log information.
    copy
        If an :class:`~anndata.AnnData` is passed, determines whether a copy
        is returned.
    
    Returns
    -------
    The spatial networks are saved in adata.uns['Spatial_Net']
    """

    assert(model in ['Radius', 'KNN'])
    if verbose:
        print(f"- INFO - Calculating '{use_rep}' graph.")
    coor = pd.DataFrame(adata.obsm[use_rep])
    coor.index = adata.obs.index
    # coor.columns = ['imagerow', 'imagecol']

    if model == 'Radius':
        nbrs = sklearn.neighbors.NearestNeighbors(radius=rad_cutoff).fit(coor)
        distances, indices = nbrs.radius_neighbors(coor, return_distance=True)
        KNN_list = []
        for it in range(indices.shape[0]):
            KNN_list.append(pd.DataFrame(zip([it]*indices[it].shape[0], indices[it], distances[it])))
    
    if model == 'KNN':
        nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=k_cutoff+1).fit(coor)
        distances, indices = nbrs.kneighbors(coor)
        KNN_list = []
        for it in range(indices.shape[0]):
            KNN_list.append(pd.DataFrame(zip([it]*indices.shape[1],indices[it,:], distances[it,:])))

    KNN_df = pd.concat(KNN_list)
    KNN_df.columns = ['Cell1', 'Cell2', 'Distance']

    Spatial_Net = KNN_df.copy()
    Spatial_Net = Spatial_Net.loc[Spatial_Net['Distance']>0,]
    id_cell_trans = dict(zip(range(coor.shape[0]), np.array(coor.index), ))
    Spatial_Net['Cell1'] = Spatial_Net['Cell1'].map(id_cell_trans)
    Spatial_Net['Cell2'] = Spatial_Net['Cell2'].map(id_cell_trans)
    if verbose:
        print(f'The graph contains {Spatial_Net.shape[0]} edges, {adata.n_obs} spots.')
        print(f'{Spatial_Net.shape[0]/adata.n_obs:.4f} neighbors per spot on average.')
    
    adata.uns['Spatial_Net'] = Spatial_Net
    return Spatial_Net if copy else None

# Modified from https://github.com/gao-lab/GLUE
def config_data(
        adata: AnnData, prob_model: str,
        use_highly_variable: bool = True,
        use_layer: Optional[str] = None,
        use_batch: Optional[str] = None,
        use_spatial: bool = True,
        use_obs_names: bool = True,
        key_added: str = "SWITCH_config"
    ):
    '''
    Configure dataset for model training

    Parameters
    ----------
    adata
        Dataset to be configured

    prob_model
        Probabilistic generative model used by the decoder, must be one of {"Normal", "ZIN", "ZILN", "NB", "ZINB"}.

    use_highly_variable
        Whether to use highly variable features

    use_layer
        Data layer to use (key in adata.layers)

    use_batch
        Data batch to use (key in adata.obs)

    use_obs_names
        Whether to use obs_names to mark paired cells across different datasets
        ** CAUTION **, Not implemented.
    '''
    data_config = {}
    data_config["prob_model"] = prob_model
    if use_highly_variable:
        if "highly_variable" not in adata.var:
            raise ValueError("Please mark highly variable features first!")
        data_config["use_highly_variable"] = True
        data_config["features"] = adata.var.query("highly_variable").index.to_numpy().tolist()
    else:
        data_config["use_highly_variable"] = False
        data_config["features"] = adata.var_names.to_numpy().tolist()
    if use_layer:
        if use_layer not in adata.layers:
            raise ValueError("Invalid `use_layer`!")
        data_config["use_layer"] = use_layer
    else:
        data_config["use_layer"] = None
    if use_batch:
        if use_batch not in adata.obs:
            raise ValueError("Invalid `use_batch`!")
        data_config["use_batch"] = use_batch
        data_config["batches"] = pd.Index(
            adata.obs[use_batch]
        ).dropna().drop_duplicates().sort_values().to_numpy() 
    else:
        data_config["use_batch"] = None
        data_config["batches"] = None
    data_config["use_obs_names"] = use_obs_names
    if use_spatial:
        if "spatial" not in adata.obsm:
            raise ValueError("No spatial coordinates exist!")
        data_config["use_spatial"] = use_spatial
    else:
        data_config["use_spatial"] = False
    adata.uns[key_added] = data_config

# Cite from https://github.com/gao-lab/GLUE
def get_gene_annotation(
        adata: AnnData, gtf: os.PathLike, 
        gtf_by: str, var_by: str = None,
        drop_na: bool = True,
) -> None:
    r"""
    Get genomic annotation of genes by joining with a GTF file.

    Parameters
    ----------
    adata
        Input dataset
    gtf
        Path to the GTF file
    gtf_by
        Specify a field in the GTF attributes used to merge with ``adata.var``,
        e.g. "gene_id", "gene_name".
    var_by
        Specify a column in ``adata.var`` used to merge with GTF attributes,
        otherwise ``adata.var_names`` is used by default.
    drop_na
        Whether to remove genes that have not been successfully annotated.

    Note
    ----
    The genomic locations are converted to 0-based as specified
    in bed format rather than 1-based as specified in GTF format.
    """
    GTF_COLUMNS = pd.Index([
    "seqname", "source", "feature", "start", "end",
    "score", "strand", "frame", "attribute"
    ])

    def read_gtf(fname: os.PathLike):
        r"""
        Read GTF file

        Parameters
        ----------
        fname
            GTF file

        Returns
        -------
        gtf
            Loaded :class:`Gtf` object
        """
        loaded = pd.read_csv(fname, sep="\t", header=None, comment="#")
        loaded.columns = GTF_COLUMNS[:loaded.shape[1]]
        return loaded

    def split_attribute(df: pd.DataFrame):
        r"""
        Extract all attributes from the "attribute" column
        and append them to existing columns

        Returns
        -------
        splitted
            Gtf with splitted attribute columns appended
        """
        pattern = re.compile(r'([^\s]+) "([^"]+)";')
        splitted = pd.DataFrame.from_records(np.vectorize(lambda x: {
            key: val for key, val in pattern.findall(x)
        })(df["attribute"]), index=df.index)

        return df.assign(**splitted)

    def GTF2BED(df: pd.DataFrame,name: Optional[str] = None):
        r"""
        Convert GTF files to BED format
        """
        bed_df = pd.DataFrame(df, copy=True).loc[
            :, ("seqname", "start", "end", "score", "strand")
        ]
        bed_df.insert(3, "name", np.repeat(
            ".", len(bed_df)
        ) if name is None else df[name])
        bed_df["start"] -= 1  # Convert to zero-based
        bed_df.columns = (
            "chrom", "chromStart", "chromEnd", "name", "score", "strand"
        )
        return bed_df
    
    var_by = adata.var_names if var_by is None else adata.var[var_by]
    gtf = split_attribute(read_gtf(gtf).query("feature == 'gene'"))
    gtf = gtf.sort_values("seqname").drop_duplicates(
        subset=[gtf_by], keep="last"
    )  # Typically, scaffolds come first, chromosomes come last
    merge_df = pd.concat([
        pd.DataFrame(GTF2BED(gtf,name=gtf_by)),
        pd.DataFrame(gtf).drop(columns=GTF_COLUMNS)  # Only use the splitted attributes
    ], axis=1).set_index(gtf_by).reindex(var_by).set_index(adata.var.index)
    adata.var = adata.var.assign(**merge_df)
    if(drop_na):
        filtered_rows = adata.var[~adata.var[['chrom', 'chromStart', 'chromEnd']].isna().any(axis=1)].index
        # print(filtered_rows)
        if(adata.shape[1] - len(filtered_rows) > 0):
            print(f"- INFO - {adata.shape[1] - len(filtered_rows)} genes were not annotated and dropped.")
        adata._inplace_subset_var(filtered_rows)

# Cite from https://github.com/gao-lab/GLUE        
def rna_anchored_guidance_graph(
    rna: AnnData, *others: AnnData,
    gene_region: str = "combined", promoter_len: int = 2000,
    extend_range: int = 0, extend_fn: Callable[[int], float] = None,
    signs: Optional[List[int]] = None, propagate_highly_variable: bool = True,
) -> nx.MultiDiGraph:
    r"""
    Build guidance graph anchored on RNA genes

    Parameters
    ----------
    rna
        Anchor RNA dataset
    *others
        Other datasets
    gene_region
        Defines the genomic region of genes, must be one of
        ``{"gene_body", "promoter", "combined"}``.
    promoter_len
        Defines the length of gene promoters (bp upstream of TSS)
    extend_range
        Maximal extend distance beyond gene regions
    extend_fn
        Distance-decreasing weight function for the extended regions
        (by default :func:`dist_power_decay`)
    signs
        Sign of edges between RNA genes and features in each ``*others``
        dataset, must have the same length as ``*others``. Signs must be
        one of ``{-1, 1}``. By default, all edges have positive signs of ``1``.
    propagate_highly_variable
        Whether to propagate highly variable genes to other datasets,
        datasets in ``*others`` would be modified in place.

    Returns
    -------
    graph
        Prior regulatory graph

    Note
    ----
    In this function, features in the same dataset can only connect to
    anchor genes via the same edge sign. For more flexibility, please
    construct the guidance graph manually.
    """

    def dist_power_decay(x: int) -> float:
        r"""
        Distance-based power decay weight, computed as
        :math:`w = {\left( \frac {d + 1000} {1000} \right)} ^ {-0.75}`

        Parameters
        ----------
        x
            Distance (in bp)

        Returns
        -------
        weight
            Decaying weight
        """
        return ((x + 1000) / 1000) ** (-0.75)

    def rectify_df(df: pd.DataFrame) -> pd.DataFrame:

        BED_COLUMNS = pd.Index([
        "chrom", "chromStart", "chromEnd", "name", "score",
        "strand", "thickStart", "thickEnd", "itemRgb",
        "blockCount", "blockSizes", "blockStarts"
        ])
        for item in BED_COLUMNS:
            if item in df:
                if item in ("chromStart", "chromEnd"):
                    df[item] = df[item].astype(int)
                else:
                    df[item] = df[item].astype(str)
            elif item not in ("chrom", "chromStart", "chromEnd"):
                df[item] = "."
            else:
                raise ValueError(f"Required column {item} is missing!")
        df = df.loc[:, BED_COLUMNS]
        if len(df.columns) != len(BED_COLUMNS) or np.any(df.columns != BED_COLUMNS):
            raise ValueError("Invalid BED format!")
        return df
    
    def strand_specific_start_site(df: pd.DataFrame):
        r"""
        Convert to strand-specific start sites of genomic features

        Returns
        -------
        start_site_bed
            A new :class:`Bed` object, containing strand-specific start sites
            of the current :class:`Bed` object
        """
        if set(df["strand"]) != set(["+", "-"]):
            raise ValueError("Not all features are strand specific!")
        df = pd.DataFrame(df, copy=True)
        pos_strand = df.query("strand == '+'").index
        neg_strand = df.query("strand == '-'").index
        df.loc[pos_strand, "chromEnd"] = df.loc[pos_strand, "chromStart"] + 1
        df.loc[neg_strand, "chromStart"] = df.loc[neg_strand, "chromEnd"] - 1
        return df

    def expand(
            df: pd.DataFrame, upstream: int, downstream: int,
            chr_len: Optional[Mapping[str, int]] = None
    ):
        r"""
        Expand genomic features towards upstream and downstream

        Parameters
        ----------
        upstream
            Number of bps to expand in the upstream direction
        downstream
            Number of bps to expand in the downstream direction
        chr_len
            Length of each chromosome

        Returns
        -------
        expanded_bed
            A new :class:`Bed` object, containing expanded features
            of the current :class:`Bed` object

        Note
        ----
        Starting position < 0 after expansion is always trimmed.
        Ending position exceeding chromosome length is trimed only if
        ``chr_len`` is specified.
        """
        if upstream == downstream == 0:
            return df
        df = pd.DataFrame(df, copy=True)
        if upstream == downstream:  # symmetric
            df["chromStart"] -= upstream
            df["chromEnd"] += downstream
        else:  # asymmetric
            if set(df["strand"]) != set(["+", "-"]):
                raise ValueError("Not all features are strand specific!")
            pos_strand = df.query("strand == '+'").index
            neg_strand = df.query("strand == '-'").index
            if upstream:
                df.loc[pos_strand, "chromStart"] -= upstream
                df.loc[neg_strand, "chromEnd"] += upstream
            if downstream:
                df.loc[pos_strand, "chromEnd"] += downstream
                df.loc[neg_strand, "chromStart"] -= downstream
        df["chromStart"] = np.maximum(df["chromStart"], 0)
        if chr_len:
            chr_len = df["chrom"].map(chr_len)
            df["chromEnd"] = np.minimum(df["chromEnd"], chr_len)
        return df

    def df2bedtool(df: pd.DataFrame):
        return BedTool(Interval(
            row["chrom"], row["chromStart"], row["chromEnd"],
            name=row["name"], score=row["score"], strand=row["strand"]
        ) for _, row in df.iterrows())

    def interval_dist(x: Interval, y: Interval) -> int:
        r"""
        Compute distance and relative position between two bed intervals

        Parameters
        ----------
        x
            First interval
        y
            Second interval

        Returns
        -------
        dist
            Signed distance between ``x`` and ``y``
        """
        if x.chrom != y.chrom:
            return np.inf * (-1 if x.chrom < y.chrom else 1)
        if x.start < y.stop and y.start < x.stop:
            return 0
        if x.stop <= y.start:
            return x.stop - y.start - 1
        if y.stop <= x.start:
            return x.start - y.stop + 1
    
    def window_graph(
        left: pd.DataFrame, right: pd.DataFrame, window_size: int,
        left_sorted: bool = False, right_sorted: bool = False,
        attr_fn: Optional[Callable[[Interval, Interval, float], Mapping[str, Any]]] = None
    ) -> nx.MultiDiGraph:
        r"""
        Construct a window graph between two sets of genomic features, where
        features pairs within a window size are connected.

        Parameters
        ----------
        left
            First feature set, either a :class:`Bed` object or path to a bed file
        right
            Second feature set, either a :class:`Bed` object or path to a bed file
        window_size
            Window size (in bp)
        left_sorted
            Whether ``left`` is already sorted
        right_sorted
            Whether ``right`` is already sorted
        attr_fn
            Function to compute edge attributes for connected features,
            should accept the following three positional arguments:

            - l: left interval
            - r: right interval
            - d: signed distance between the intervals

            By default no edge attribute is created.

        Returns
        -------
        graph
            Window graph
        """
        pbar_total = len(left)
        left = df2bedtool(left)
        if not left_sorted:
            left = left.sort(stream=True)
        left = iter(left)  # Resumable iterator
        right = df2bedtool(right)
        if not right_sorted:
            right = right.sort(stream=True)
        right = iter(right)  # Resumable iterator

        attr_fn = attr_fn or (lambda l, r, d: {})
        if pbar_total is not None:
            left = tqdm(left, total=pbar_total, desc="window_graph")
        graph = nx.MultiDiGraph()
        window = collections.OrderedDict()  # Used as ordered set
        for l in left:
            for r in list(window.keys()):  # Allow remove during iteration
                d = interval_dist(l, r)
                if -window_size <= d <= window_size:
                    graph.add_edge(l.name, r.name, **attr_fn(l, r, d))
                elif d > window_size:
                    del window[r]
                else:  # dist < -window_size
                    break  # No need to expand window
            else:
                for r in right:  # Resume from last break
                    d = interval_dist(l, r)
                    if -window_size <= d <= window_size:
                        graph.add_edge(l.name, r.name, **attr_fn(l, r, d))
                    elif d > window_size:
                        continue
                    window[r] = None  # Placeholder
                    if d < -window_size:
                        break
        pybedtools.cleanup()
        return graph
    
    def compose_multigraph(*graphs: nx.Graph) -> nx.MultiGraph:
        r"""
        Compose multi-graph from multiple graphs with no edge collision

        Parameters
        ----------
        graphs
            An arbitrary number of graphs to be composed from

        Returns
        -------
        composed
            Composed multi-graph

        Note
        ----
        The resulting multi-graph would be directed if any of the input graphs
        is directed.
        """
        if any(nx.is_directed(graph) for graph in graphs):
            graphs = [graph.to_directed() for graph in graphs]
            composed = nx.MultiDiGraph()
        else:
            composed = nx.MultiGraph()
        composed.add_edges_from(
            (e[0], e[1], graph.edges[e])
            for graph in graphs for e in graph.edges
        )
        return composed
    
    def reachable_vertices(graph: nx.Graph, source: Iterable[Any]) -> Set[Any]:
        r"""
        Identify vertices reachable from source vertices
        (including source vertices themselves)

        Parameters
        ----------
        graph
            Input graph
        source
            Source vertices

        Returns
        -------
        reachable_vertices
            Reachable vertices
        """
        source = set(source)
        return set(chain.from_iterable(
            nx.descendants(graph, item) for item in source
            if graph.has_node(item)
        )).union(source)

    signs = signs or [1] * len(others)
    extend_fn = extend_fn or dist_power_decay
    if len(others) != len(signs):
        raise RuntimeError("Length of ``others`` and ``signs`` must match!")
    if set(signs).difference({-1, 1}):
        raise RuntimeError("``signs`` can only contain {-1, 1}!")

    rna_bed = rectify_df(rna.var.assign(name=rna.var_names))
    other_beds = [rectify_df(other.var.assign(name=other.var_names)) for other in others]

    if gene_region == "promoter":
        rna_bed = expand(strand_specific_start_site(rna_bed), promoter_len, 0)
    elif gene_region == "combined":
        rna_bed = expand(rna_bed, promoter_len, 0)
    elif gene_region != "gene_body":
        raise ValueError("Unrecognized `gene_range`!")
    graphs = [window_graph(
        rna_bed, other_bed, window_size=extend_range,
        attr_fn=lambda l, r, d, s=sign: {
            "dist": abs(d), "weight": extend_fn(abs(d)), "sign": s
        }
    ) for other_bed, sign in zip(other_beds, signs)]
    graph = compose_multigraph(*graphs)

    if propagate_highly_variable:
        hvg_reachable = reachable_vertices(graph, rna.var.query("highly_variable").index)
        for other in others:
            other.var["highly_variable"] = [
                item in hvg_reachable for item in other.var_names
            ]

    rgraph = graph.reverse()
    nx.set_edge_attributes(graph, "fwd", name="type")
    nx.set_edge_attributes(rgraph, "rev", name="type")
    graph = compose_multigraph(graph, rgraph)
    all_features = set(chain.from_iterable(
        map(lambda x: x.var_names, [rna, *others])
    ))
    for item in all_features:
        graph.add_edge(item, item, weight=1.0, sign=1, type="loop")
    return graph

# Cite from https://github.com/gao-lab/GLUE
def check_graph(
        graph: nx.Graph, adatas: Iterable[AnnData], verbose = True,
        cov: str = "error", attr: str = "error",
        loop: str = "error", sym: str = "error"
) -> None:
    
    passed = True
    if(verbose and cov != "ignore"):
        print(f"- INFO - Checking variable coverage...")
    if not all(
        all(graph.has_node(var_name) for var_name in adata.var_names)
        for adata in adatas
    ):
        passed = False
        msg = "Some variables are not covered by the graph!"
        if cov == "error":
            raise ValueError(msg)
        elif cov == "warn":
            print(msg)
        elif cov != "ignore":
            raise ValueError(f"Invalid `cov`: {cov}")
    
    if(verbose and attr != "ignore"):
        print(f"- INFO - Checking edge attributes...")
    if not all(
        "weight" in edge_attr and "sign" in edge_attr
        for edge_attr in dict(graph.edges).values()
    ):
        passed = False
        msg = "Missing weight or sign as edge attribute!"
        if attr == "error":
            raise ValueError(msg)
        elif attr == "warn":
            print(msg)
        elif cov != "ignore":
            raise ValueError(f"Invalid `attr`: {attr}")
    
    if(verbose and loop != "ignore"):
        print(f"- INFO - Checking self-loops...")
    if not all(
        graph.has_edge(node, node) for node in graph.nodes
    ):
        passed = False
        msg = "Missing self-loop!"
        if loop == "error":
            raise ValueError(msg)
        elif loop == "warn":
            print(msg)
        elif loop != "ignore":
            raise ValueError(f"Invalid `loop`: {loop}")

    if(verbose and sym != "ignore"):
        print(f"- INFO - Checking graph symmetry...")
    if not all(
        graph.has_edge(e[1], e[0]) for e in graph.edges
    ):
        passed = False
        msg = "Graph is not symmetric!"
        if sym == "error":
            raise ValueError(msg)
        elif sym == "warn":
            print(msg)
        elif sym != "ignore":
            raise ValueError(f"Invalid `sym`: {sym}")
    if passed and verbose:
        print(f"- INFO - All checks passed!")
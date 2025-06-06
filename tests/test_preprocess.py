import sys
sys.path.append("/mnt/datadisk/lizhongzhan/SpaMultiOmics/main/")
from switch import preprocess

import pandas as pd
import networkx as nx

def test_Cal_Spatial_Net(rna):
    result = preprocess.Cal_Spatial_Net(rna, use_rep="spatial", k_cutoff=3, model="KNN", verbose=False, copy=True)
    assert isinstance(result, pd.DataFrame)
    assert set(["Cell1", "Cell2", "Distance"]).issubset(result.columns)
    assert (result["Cell1"] != result["Cell2"]).all()

    result = preprocess.Cal_Spatial_Net(rna, use_rep="spatial", rad_cutoff=1.0, model="Radius", verbose=False, copy=True)
    assert isinstance(result, pd.DataFrame)
    assert set(["Cell1", "Cell2", "Distance"]).issubset(result.columns)

def test_config_data(rna_pp):

    preprocess.config_data(rna_pp, prob_model="NB", use_highly_variable=True)
    assert "SWITCH_config" in rna_pp.uns
    config = rna_pp.uns["SWITCH_config"]
    assert config["prob_model"] == "NB"
    assert config["use_highly_variable"] is True
    assert isinstance(config["features"], list)

    preprocess.config_data(rna_pp, prob_model="NB", use_highly_variable=False)
    assert "SWITCH_config" in rna_pp.uns
    config = rna_pp.uns["SWITCH_config"]
    assert config["use_highly_variable"] is False
    assert len(config["features"]) == rna_pp.shape[1]


def test_get_gene_annotation(rna, gtf_file):
    preprocess.get_gene_annotation(rna, gtf=gtf_file, gtf_by="gene_id", drop_na=True)
    assert all(x in rna.var.columns for x in ["chrom", "chromStart", "chromEnd"])

def test_rna_anchored_guidance_graph(rna_pp, atac_pp):
    graph = preprocess.rna_anchored_guidance_graph(rna_pp, atac_pp, promoter_len=2, extend_range=15)
    assert isinstance(graph, nx.MultiDiGraph)

    for n in graph.nodes:
        assert graph.has_edge(n, n)
    any_edge = list(graph.edges(data=True))[0]
    assert "weight" in any_edge[2] and "sign" in any_edge[2]

def test_check_graph(guidance, rna, atac):
    preprocess.check_graph(guidance, [rna, atac])



r"""
Test configuration
"""

import anndata
import networkx as nx
import numpy as np
import pandas as pd
import pytest
import scanpy as sc
import scipy.sparse
import torch

import sys
sys.path.append("/mnt/datadisk/lizhongzhan/SpaMultiOmics/main/")
from switch import SWITCH, preprocess


def pytest_addoption(parser):
    pass


def pytest_generate_tests():
    np.random.seed(0)
    torch.manual_seed(0)


def pytest_configure(config):
    pass

@pytest.fixture
def gtf_file(tmp_path):
    file = tmp_path / "test.gtf"
    with file.open("w") as f:
        f.write(
            """
chr1\tHAVANA\tgene\t11\t20\t.\t+\t.\tgene_id "A"; source "havana";
chr2\tHAVANA\tgene\t1\t10\t.\t-\t.\tgene_id "B"; source "havana";
chr2\tHAVANA\tgene\t11\t20\t.\t+\t.\tgene_id "C"; source "havana";
        """.strip(
                " \n"
            )
        )
    yield file
    file.unlink()


@pytest.fixture
def bed_file(tmp_path):
    file = tmp_path / "test.bed"
    with file.open("w") as f:
        f.write(
            """
chr1\t0\t10\ta\t.\t+
chr1\t20\t30\tb\t.\t+
chr1\t30\t40\tc\t.\t-
chr2\t0\t10\td\t.\t+
chr2\t10\t20\te\t.\t-
chr3\t0\t10\tf\t.\t+
        """.strip(
                " \n"
            )
        )
    yield file
    file.unlink()

@pytest.fixture
def mat():
    return np.random.randint(0, 50, size=(144, 3))


@pytest.fixture
def spmat():
    return scipy.sparse.csr_matrix(np.random.randint(0, 20, size=(100, 6)))


@pytest.fixture
def rna(mat):
    X = mat
    obs = pd.DataFrame(
        {},
        index=[f"RNA-{i}" for i in range(144)],
    )
    var = pd.DataFrame(index=["A", "B", "C"])
    arange = np.expand_dims(np.arange(mat.shape[0]), 1).repeat(mat.shape[1], axis=1)
    obs.index.name, var.index.name = "cells", "genes",
    obsm = {"spatial": np.array([[i, j]  for j in range(12) for i in range(12)])}
    return anndata.AnnData(
        X=X, obs=obs, var=var, layers={"arange": arange},
        dtype=np.int32, obsm =obsm
    )


@pytest.fixture
def atac(spmat, bed_file):
    X = spmat
    obs = pd.DataFrame(
        {},
        index=[f"ATAC-{i}" for i in range(100)],
    )
    var = pd.read_csv(
        bed_file,
        sep="\t",
        header=None,
        comment="#",
        names=["chrom", "chromStart", "chromEnd", "name", "score", "strand"],
    ).set_index("name", drop=False)
    arange = np.expand_dims(np.arange(spmat.shape[0]), 1).repeat(spmat.shape[1], axis=1)
    obs.index.name, var.index.name = "cells", "peaks"
    obsm = {"spatial": np.array([[i, j]  for j in range(10) for i in range(10)])}
    return anndata.AnnData(
        X=X, obs=obs, var=var, layers={"arange": arange},
        dtype=np.int32, obsm=obsm,
    )


@pytest.fixture
def rna_pp(rna, gtf_file):
    preprocess.get_gene_annotation(
        rna, gtf=gtf_file, gtf_by="gene_id", drop_na=True
    )
    rna.var["highly_variable"] = [True, False, True]
    rna.layers["counts"] = rna.X.copy()
    sc.pp.normalize_total(rna)
    sc.pp.log1p(rna)
    return rna


@pytest.fixture
def atac_pp(atac):
    return atac


@pytest.fixture
def guidance(rna_pp, atac_pp):
    return preprocess.rna_anchored_guidance_graph(
        rna_pp,
        atac_pp,
        promoter_len=2,
        extend_range=15,
        propagate_highly_variable=False,
    )


@pytest.fixture
def eidx():
    return np.array([[0, 1, 2, 1], [0, 1, 2, 2]])


@pytest.fixture
def ewt():
    return np.array([1.0, 0.4, 0.7, 0.1])


@pytest.fixture
def esgn():
    return np.array([1.0, 1.0, 1.0, 1.0])
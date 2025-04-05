# %%
import warnings
import anndata as ad
import networkx as nx
import scanpy as sc
import random
import numpy as np
import torch
import scglue
from itertools import chain

import sys
data_dir = str(sys.argv[1])
save_dir = str(sys.argv[2])
wk_dir = ""
n_clus = int(sys.argv[3])
peak_data_name = str(sys.argv[4])

import os
os.chdir(wk_dir)

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
seed_everything(111)

rna = ad.read_h5ad(data_dir+"/rna-pp.h5ad")
atac = ad.read_h5ad(data_dir+"/"+peak_data_name)

rna.layers["counts"] = rna.X.copy()
sc.pp.highly_variable_genes(rna, n_top_genes=2000, flavor="seurat_v3",layer="counts")

split = atac.var_names.str.split(r"[:-]")
atac.var["chrom"] = split.map(lambda x: x[0])
atac.var["chromStart"] = split.map(lambda x: x[1]).astype(int)
atac.var["chromEnd"] = split.map(lambda x: x[2]).astype(int)
# print(i.var)

scglue.data.get_gene_annotation(
    rna, gtf="../Mouse_embryo/gencode.vM25.chr_patch_hapl_scaff.annotation.gtf.gz",
    gtf_by="gene_name"
)
guidance = scglue.genomics.rna_anchored_guidance_graph(rna, atac)

guidance_hvf = guidance.subgraph(chain(
    rna.var.query("highly_variable").index,
    atac.var.query("highly_variable").index
)).copy()

rna.X = rna.layers["counts"]
sc.pp.normalize_total(rna)
sc.pp.log1p(rna)
sc.pp.scale(rna)
sc.tl.pca(rna, n_comps=100, svd_solver="auto")

scglue.data.lsi(atac, n_components=100, n_iter=15)

scglue.models.configure_dataset(
    rna, "NB", use_highly_variable=True,
    use_layer="counts",
    use_rep="X_pca",
)
scglue.models.configure_dataset(
    atac, "NB", use_highly_variable=True,
    use_rep="X_lsi",
)

glue = scglue.models.fit_SCGLUE(
    {"rna": rna, "atac": atac}, guidance_hvf,
    model=scglue.models.SCGLUEModel,
    fit_kws={"directory": "glue"},
)

rna.obsm["X_glue"] = glue.encode_data("rna", rna)
atac.obsm["X_glue"] = glue.encode_data("atac", atac)

combined = ad.concat([rna, atac], label="omics")

import pandas as pd
def res_search_fixed_clus(adata, fixed_clus_count, increment=0.05):
    closest_count = np.inf  
    closest_res = None  
    
    for res in sorted(list(np.arange(0.1, 2, increment)), reverse=True):
        sc.tl.leiden(adata, random_state=0, resolution=res)
        count_unique_leiden = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
        current_diff = abs(count_unique_leiden - fixed_clus_count)
        if current_diff < closest_count:
            closest_count = current_diff
            closest_res = res
        if count_unique_leiden == fixed_clus_count:
            break

    return closest_res
sc.pp.neighbors(combined, use_rep="X_glue")
res =  res_search_fixed_clus(combined, fixed_clus_count=n_clus)

sc.tl.leiden(adata=combined, resolution=res)
sc.tl.umap(combined)

cluster = combined.obs['leiden']
cluster.to_csv(save_dir+"/GLUE_cluster.csv")

umap = pd.DataFrame(combined.obsm["X_umap"])
umap.to_csv(save_dir+"/GLUE_umap.csv")

embed = pd.DataFrame(combined.obsm["X_glue"])
embed.to_csv(save_dir+"/GLUE_embed.csv")

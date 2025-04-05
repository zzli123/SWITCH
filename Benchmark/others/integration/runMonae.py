# %%
import anndata
import networkx as nx
import scanpy as sc
import anndata
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import sys
sys.path.append('/mnt/datadisk/lizhongzhan/SpaMultiOmics/main/')
import scglue
from switch import preprocess
from monae.src.config import configure_dataset
from monae.src.train import covel_train
import dill

# %%
import sys
data_dir = str(sys.argv[1])
save_dir = str(sys.argv[2])
n_clus = int(sys.argv[3])
peak_data_name = str(sys.argv[4])


wk_dir = "/mnt/datadisk/lizhongzhan/SpaMultiOmics/SCRIPT/Figure1/Monae/"
import os
os.chdir(wk_dir)

rna = anndata.read_h5ad(data_dir+"/rna-pp.h5ad")
atac = anndata.read_h5ad(data_dir+"/"+peak_data_name)

# preprocess.get_gene_annotation(rna, 
#                     gtf=data_dir+"/gencode.v47.annotation.gtf.gz",
#                     gtf_by="gene_name",
#                     drop_na=True
# )
from itertools import chain

# split = atac.var_names.str.split(r"[:-]")
# atac.var["chrom"] = split.map(lambda x: x[0])
# atac.var["chromStart"] = split.map(lambda x: x[1]).astype(int)
# atac.var["chromEnd"] = split.map(lambda x: x[2]).astype(int)
# print(i.var)
guidance = nx.read_graphml(data_dir+"/guidance.graphml.gz")

# guidance_hvf = nx.read_graphml("guidance_hvf.graphml.gz")

# guidance = scglue.genomics.rna_anchored_guidance_graph(rna, atac)
graph = guidance.subgraph(chain(
    rna.var.query("highly_variable").index,
    atac.var.query("highly_variable").index
)).copy()

rna.obsm['X_pre'] = rna.X
atac.obsm['X_pre'] = atac.X.todense()

adatas=[rna, atac]
modal_names=["RNA", "ATAC"]
prob=['NB','NB']
rep = ['X_pre', 'X_pre']

vertices = sorted(graph.nodes)
for idx, adata in enumerate(adatas):
    configure_dataset(adata, prob[idx], 
                      use_highly_variable=True,
                      use_rep=rep[idx],
                      )

data = dict(zip(modal_names, adatas))

# %%
covel = covel_train(
        adatas, 
        graph,
        fit_kws={"directory": "monae_out"},
        config = [modal_names, prob, rep],
        result_path = "monae_out"
)

# %%
for modal_name in modal_names:
    data[modal_name].obs['domain'] = modal_name

# %%
for modal_name in modal_names:
    data[modal_name].obsm['embedding'] = covel.encode_data(modal_name, data[modal_name])

adata = sc.concat([data["RNA"], data["ATAC"]])

# print("UMAP vis integration data, in combined.h5ad X_umap")
sc.pp.neighbors(adata, use_rep="embedding", metric="cosine")
sc.tl.umap(adata)

# %%
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
res =  res_search_fixed_clus(adata, fixed_clus_count=n_clus)

# imputation_X = covel.decode_data("ATAC", "RNA", data["ATAC"], graph)
# imputation = sc.AnnData(imputation_X)
# imputation.obsm["spatial"] = atac.obsm["spatial"]
# hvg_rna = rna[:,rna.var["highly_variable"]]
# imputation.var_names = list(hvg_rna.var_names)
# imputation.write_h5ad(data_dir+"/monae_imp_rna.h5ad")
# print(imputation)

# %%
# imputation_X = covel.decode_data("RNA", "ATAC", data["RNA"], graph)
# imputation = sc.AnnData(imputation_X)
# imputation.obsm["spatial"] = rna.obsm["spatial"]
# hvg_atac = atac[:,atac.var["highly_variable"]]
# imputation.var_names = list(hvg_atac.var_names)
# imputation.write_h5ad(data_dir+"/monae_imp_atac.h5ad")
# print(imputation)


cluster = adata.obs['leiden']
cluster.to_csv(save_dir+"/Monae_cluster.csv")

umap = pd.DataFrame(adata.obsm["X_umap"])
umap.to_csv(save_dir+"/Monae_umap.csv")

embed = pd.DataFrame(adata.obsm["embedding"])
embed.to_csv(save_dir+"/Monae_embed.csv")
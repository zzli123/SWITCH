# %%
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from scipy.spatial.distance import cdist
import muon as mu
import numpy as np
import anndata as ad
import scanpy as sc

import scconfluence

# %%
import os
import sys
data_dir = str(sys.argv[1])
save_dir = str(sys.argv[2])
wk_dir = ""
n_clus = int(sys.argv[3])
peak_data_name = str(sys.argv[4])


# %%
rna = sc.read_h5ad(data_dir+"/rna-pp.h5ad")
atac = sc.read_h5ad(data_dir+"/"+peak_data_name)

# %%
# sc.pp.filter_genes(atac, min_cells=atac.shape[0]*0.01)
atac = atac[:, atac.var["highly_variable"]]

# %%
sc.pp.filter_genes(rna, min_cells=20)
# we're actually filtering peaks here
sc.pp.filter_genes(atac, min_cells=20)

# %%
gs = sc.read_h5ad(data_dir+"/simba_gs.h5ad")
cm_genes = list(set(rna.var_names) & set(gs.var_names))
cm_features_rna = rna[:, cm_genes].copy()
cm_features_atac = gs[:, cm_genes].copy()

# %%
sc.pp.normalize_total(cm_features_rna, target_sum=10000.)
sc.pp.log1p(cm_features_rna)

# Log-normalize scATAC gene activities
sc.pp.normalize_total(cm_features_atac, target_sum=10000.)
sc.pp.log1p(cm_features_atac)

# Select highly variable genes for both modalities using the more reliable scRNA counts
cm_hvg_genes = sc.pp.highly_variable_genes(cm_features_rna, n_top_genes=3000, subset=False, inplace=False)
cm_features_rna = cm_features_rna[:, cm_hvg_genes["highly_variable"]].copy()
cm_features_atac = cm_features_atac[:, cm_hvg_genes["highly_variable"]].copy()

sc.pp.scale(cm_features_rna)
sc.pp.scale(cm_features_atac)

# %%
mdata = mu.MuData({'rna': rna, 'atac': atac})
mdata.uns["cross_rna+atac"] = cdist(cm_features_rna.X, cm_features_atac.X, metric="correlation")
mdata.uns["cross_keys"] = ["cross_rna+atac"]

# %%
mdata["rna"].layers["counts"] = mdata["rna"].X.copy()

# Log-normalize the scRNA counts
sc.pp.normalize_total(mdata["rna"], target_sum=10000.)
sc.pp.log1p(mdata["rna"])

# Since we use both raw and normalized gene counts in our autoencoder it makes sense to select highly variable genes based on both criteria
# raw_hvg = sc.pp.highly_variable_genes(mdata["rna"], layer="counts", n_top_genes=3000, subset=False, inplace=False,
#                                       )["highly_variable"].values
raw_hvg = rna.var["highly_variable"].values
norm_hvg = sc.pp.highly_variable_genes(mdata["rna"], n_top_genes=3000, subset=False,
                                       inplace=False)["highly_variable"].values
mdata.mod["rna"] = mdata["rna"][:, np.logical_or(raw_hvg, norm_hvg)].copy()

# Perform PCA on the selected genes
sc.tl.pca(mdata["rna"], n_comps=100, zero_center=None)

# %%
mu.atac.pp.tfidf(mdata["atac"], log_tf=True, log_idf=True)
sc.tl.pca(mdata["atac"], n_comps=100, zero_center=None)

# %%
import torch
torch.manual_seed(1792)
autoencoders = {"rna": scconfluence.unimodal.AutoEncoder(mdata["rna"],
                                                         modality="rna",
                                                         rep_in="X_pca",
                                                         rep_out="counts",
                                                         batch_key=None,
                                                         n_hidden=64,
                                                         n_latent=16,
                                                         type_loss="zinb"),
                "atac": scconfluence.unimodal.AutoEncoder(mdata["atac"],
                                                          modality="atac",
                                                          rep_in="X_pca",
                                                          rep_out=None,
                                                          batch_key=None,
                                                          n_hidden=64,
                                                          n_latent=16,
                                                          type_loss="l2",
                                                          reconstruction_weight=5.)}

# %%
model = scconfluence.model.ScConfluence(mdata=mdata, unimodal_aes=autoencoders,
                                        mass=0.5, reach=0.3, iot_loss_weight=0.01, sinkhorn_loss_weight=0.1)
model.fit(save_path="demo_rna_atac", use_cuda=True, max_epochs=1000)

# %%
rna.obs["omic"] = "rna"
atac.obs["omic"] = "atac"
adata_all = ad.concat([rna, atac])

# %%
adata_all.obsm["latent"] = model.get_latent(use_cuda=True)

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

# %%
sc.pp.neighbors(adata_all, use_rep="latent")
res =  res_search_fixed_clus(adata_all, fixed_clus_count=n_clus)

# %%
sc.tl.umap(adata_all)
sc.tl.leiden(adata_all, resolution=res)
# sc.pl.umap(adata_all, color=["omic","leiden"], size=50, alpha=0.7)

# %%
# t_rna, t_atac = adata_all[adata_all.obs["omic"]=="rna"], adata_all[adata_all.obs["omic"]=="atac"]
# sc.pl.spatial(t_rna, color="leiden", spot_size=1)
# sc.pl.spatial(t_atac, color="leiden", spot_size=1)

# %%
cluster = adata_all.obs['leiden']
cluster.to_csv(save_dir+"/scConfluence_cluster.csv")
umap = pd.DataFrame(adata_all.obsm["X_umap"])
umap.to_csv(save_dir+"/scConfluence_umap.csv")
embed = pd.DataFrame(adata_all.obsm["latent"])
embed.to_csv(save_dir+"/scConfluence_embed.csv")



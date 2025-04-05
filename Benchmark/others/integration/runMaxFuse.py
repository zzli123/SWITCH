# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append("/mnt/datadisk/lizhongzhan/SpaMultiOmics/main/MaxFuse_devo/09302022V/")
import match

import anndata as ad
import scanpy as sc

# %%
import sys
data_dir = str(sys.argv[1])
save_dir = str(sys.argv[2])
n_clus = int(sys.argv[3])
peak_data_name = str(sys.argv[4])

# %%
rna = sc.read_h5ad(data_dir+"/rna-pp.h5ad")
atac = sc.read_h5ad(data_dir+"/"+peak_data_name)

# %%
# sc.pp.filter_genes(atac, min_cells=atac.shape[0]*0.01)
atac = atac[:, atac.var["highly_variable"]]

# %%
gs = sc.read_h5ad(data_dir+"/simba_gs.h5ad")

# %%
shared_genes = np.intersect1d(rna.var_names, gs.var_names)
len(shared_genes)

# %%
rna_shared = np.mat(rna[:, shared_genes].X)
activity_shared = np.mat(gs[:, shared_genes].X.todense())

# mask = ((rna_shared.std(axis=0) > 0.01) & (activity_shared.std(axis=0) > 0.01)).A1 # filter out static ones
# rna_shared = rna_shared[:, mask]
# activity_shared = activity_shared[:, mask]

# %%
rna_shared = ad.AnnData(np.array(rna_shared))
sc.pp.normalize_total(rna_shared) # input data is already normalized
sc.pp.log1p(rna_shared)
sc.pp.highly_variable_genes(rna_shared, flavor="seurat_v3", n_top_genes=3000)
sc.pp.scale(rna_shared)
#rna_shared = rna_shared.X

## atac shared
activity_shared = ad.AnnData(np.array(activity_shared))
sc.pp.normalize_total(activity_shared)
sc.pp.log1p(activity_shared)
sc.pp.scale(activity_shared)
#activity_shared = activity_shared.X

# %%
vgenes = rna_shared.var.highly_variable
# shared features
rnaC_shared = rna_shared[:,vgenes].X
atac_shared = activity_shared[:,vgenes].X
# all features
rnaC_active = rna_shared[:,vgenes].X
atac_active = atac.X.todense()

# %%
spm = match.MaxFuse(
    shared_arr1=np.array(rnaC_shared),
    shared_arr2=np.array(atac_shared),
    active_arr1=np.array(rnaC_active),
    active_arr2=np.array(atac_active),
    method='centroid_shrinkage',
    labels1=None, # if None, then use scanpy clustering pipeline
    labels2=None
)

# %%
spm.split_into_batches(
    max_outward_size=5000,
    matching_ratio=5,
    metacell_size=2,
    method='binning',
    verbose=True,
    seed=42
)

# %%
spm.construct_graphs(
    n_neighbors1=15,
    n_neighbors2=15,
    svd_components1=30,
    svd_components2=15,
    resolution1=2,
    resolution2=2,
    randomized_svd=False, 
    svd_runs=1,
    resolution_tol=0.1,
    leiden_runs=1,
    leiden_seed=None,
    verbose=True
)

# %%
spm.find_initial_pivots(
    wt1=0.7, wt2=0.7,
    svd_components1=20, svd_components2=20,
    randomized_svd=False, svd_runs=1,
    verbose=True
)

# %%
spm.refine_pivots(
    wt1=0.7, wt2=0.7,
    svd_components1=100, svd_components2=None, # c500
    cca_components=20,
    filter_prop=0.,
    n_iters=8,
    randomized_svd=False, 
    svd_runs=1,
    verbose=True
)

# %%
spm.filter_bad_matches(target='pivot', filter_prop=0.4, verbose=True)

# %%
spm.propagate(
    wt1=0.7,
    wt2=0.7,
    svd_components1=30, 
    svd_components2=None, 
    randomized_svd=False, 
    svd_runs=1, 
    verbose=True
)

spm.filter_bad_matches(
    target='propagated',
    filter_prop=0.,
    verbose=True
)

dim_use = 15 # dimensions of the CCA embedding to be used for UMAP etc
rna_cca, atac_cca = spm.get_embedding(
    active_arr1 = spm.active_arr1,
    active_arr2 = spm.active_arr2,
    refit=False,
    matching=None,
    order=None,
    cca_components=20,
    cca_max_iter=None
)

rna.obsm["embed"], atac.obsm["embed"] = rna_cca[:,:dim_use], atac_cca[:,:dim_use]
rna.obs["omic"] = "rna"
atac.obs["omic"] = "atac"
adata_all = sc.concat([rna, atac])

sc.pp.neighbors(adata_all, n_neighbors=15, use_rep="embed")
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
res =  res_search_fixed_clus(adata_all, fixed_clus_count=n_clus)

sc.tl.umap(adata_all)
sc.tl.leiden(adata_all, resolution=res)

cluster = adata_all.obs['leiden']
cluster.to_csv(save_dir+"/MaxFuse_cluster.csv")

umap = pd.DataFrame(adata_all.obsm["X_umap"])
umap.to_csv(save_dir+"/MaxFuse_umap.csv")

embed = pd.DataFrame(adata_all.obsm["embed"])
embed.to_csv(save_dir+"/MaxFuse_embed.csv")


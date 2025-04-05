# %%
import scalex
from scalex import SCALEX
from scalex.plot import embedding
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns

# %%
import sys
data_dir = str(sys.argv[1])
save_dir = str(sys.argv[2])
n_clus = int(sys.argv[3])
peak_data_name = str(sys.argv[4])

wk_dir = "/mnt/datadisk/lizhongzhan/SpaMultiOmics/SCRIPT/Figure1/SCALEX/"

# %%
rna = sc.read_h5ad(data_dir+"/rna-pp.h5ad")
gs = sc.read_h5ad(data_dir+"/simba_gs.h5ad")

# %%
rna.obs["batch"] = "rna"
gs.obs["batch"] = "atac"
adata = sc.concat([rna, gs])

# %%
# sc.pp.filter_cells(adata, min_genes=0)
# sc.pp.filter_genes(adata, min_cells=0)

# sc.pp.normalize_total(adata, inplace=True)
# sc.pp.log1p(adata)
# sc.pp.highly_variable_genes(adata, n_top_genes=2000)
# adata = adata[:, adata.var.highly_variable]

# sc.pp.scale(adata, max_value=10)
# sc.tl.pca(adata)
# sc.pp.neighbors(adata, n_pcs=30, n_neighbors=30)
# sc.tl.umap(adata, min_dist=0.1)

# %%
adata.write(wk_dir+"/temp_adata.h5ad")

# %%
adata = SCALEX(data_list = [wk_dir+'temp_adata.h5ad'],
              min_features=0,
              min_cells=0,
              outdir=wk_dir,
              show=False,
              gpu=1)

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
sc.pp.neighbors(adata, use_rep="latent")
res =  res_search_fixed_clus(adata, fixed_clus_count=n_clus)

# %%
sc.tl.umap(adata)
sc.tl.leiden(adata, resolution=res)

# %%
# sc.pl.umap(adata, color=["batch","leiden"], size=50, alpha=0.7)

# %%
# t_rna, t_atac = adata[adata.obs["batch"]=="rna"], adata[adata.obs["batch"]=="atac"]
# t_rna.obsm["spatial"] , t_atac.obsm["spatial"] =  rna.obsm["spatial"], rna.obsm["spatial"]
# sc.pl.spatial(t_rna, color="leiden", spot_size=1)
# sc.pl.spatial(t_atac, color="leiden", spot_size=1)


cluster = adata.obs['leiden']
cluster.to_csv(save_dir+"/SCALEX_cluster.csv")

umap = pd.DataFrame(adata.obsm["X_umap"])
umap.to_csv(save_dir+"/SCALEX_umap.csv")

embed = pd.DataFrame(adata.obsm["latent"])
embed.to_csv(save_dir+"/SCALEX_embed.csv")


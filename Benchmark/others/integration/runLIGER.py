# %%
import pyliger
import scanpy as sc
import pandas as pd



# %%
import sys
data_dir = str(sys.argv[1])
save_dir = str(sys.argv[2])
n_clus = int(sys.argv[3])
peak_data_name = str(sys.argv[4])

# %%
rna = sc.read_h5ad(data_dir+"/rna-pp.h5ad")

# %%
# genescore = pd.read_csv("gs.csv", index_col=0)
# genescore = genescore.transpose()
# # genescore.index = ["H3keme3"+i.split(".")[1]+"-1" for i in list(genescore.index)]
# genescore = sc.AnnData(X=genescore)

# %%
genescore = sc.read_h5ad(data_dir+"/simba_gs.h5ad")

# %%
# cells = [i for i in list(rna.obs_names) if i in list(genescore.obs_names) ]
# genescore = genescore[cells,]
# genescore

# %%
rna.obs_names = ["RNA#"+i for i in list(rna.obs_names)]
genescore.obs_names = ["GS#"+i for i in list(genescore.obs_names)]

# %%
rna.obs.index.name = "cell"
rna.var.index.name = "gene"
genescore.obs.index.name = "cell"
genescore.var.index.name = "gene"
rna.uns["sample_name"] = "RNA"
genescore.uns["sample_name"] = "H3k4me3"

# %%
rna.var_names_make_unique()
genescore.var_names_make_unique()

# %%
adata_list = [rna, genescore]
ifnb_liger = pyliger.create_liger(adata_list)

# %%
pyliger.normalize(ifnb_liger)
pyliger.select_genes(ifnb_liger)
pyliger.scale_not_center(ifnb_liger)

# %%
pyliger.optimize_ALS(ifnb_liger, k = 20)

# %%
pyliger.quantile_norm(ifnb_liger)

# %%
pyliger.run_umap(ifnb_liger, distance = 'cosine', n_neighbors = 30, min_dist = 0.3)

# %%
import numpy as np
embed = pd.DataFrame(np.vstack([adata.obsm['H_norm'] for adata in ifnb_liger.adata_list]))

# %%
embed.index = sum([list(adata.obs_names) for adata in ifnb_liger.adata_list],[])

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

adata = sc.concat([rna, genescore])
adata.obsm["embed"] = np.array(embed)
sc.pp.neighbors(adata, use_rep="embed")
res =  res_search_fixed_clus(adata, fixed_clus_count=n_clus)

sc.tl.umap(adata)
sc.tl.leiden(adata, resolution=res)

# %%

cluster = adata.obs['leiden']
cluster.to_csv(save_dir+"/LIGER_cluster.csv")

umap = pd.DataFrame(adata.obsm["X_umap"])
umap.to_csv(save_dir+"/LIGER_umap.csv")

embed = pd.DataFrame(adata.obsm["embed"])
embed.to_csv(save_dir+"/LIGER_embed.csv")



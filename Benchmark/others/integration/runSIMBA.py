# %%
import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append("/mnt/datadisk/lizhongzhan/SpaMultiOmics/main/simba/")

# %%
import os
os.environ["OMP_NUM_THREADS"] = "1"

import simba as si

import sys
if __name__=='__main__':
    data_dir = str(sys.argv[1])
    save_dir = str(sys.argv[2])
    n_clus = int(sys.argv[3])
    peak_data_name = str(sys.argv[4])

    # %%
    workdir = '/mnt/datadisk/lizhongzhan/SpaMultiOmics/SCRIPT/Figure1/SIMBA/'
    si.settings.set_workdir(workdir)
    si.settings.pbg_params["workers"] = 10


    # %%
    import scanpy as sc
    rna = sc.read_h5ad(data_dir+"/rna-pp.h5ad")
    atac = sc.read_h5ad(data_dir+"/"+peak_data_name)

    # %%
    # rna.obs.index = rna.obs.index + '_rna'

    # %%
    # sc.pp.filter_genes(atac, min_cells=atac.shape[0]*0.01)
    atac = atac[:, atac.var["highly_variable"]]

    # %%
    si.pp.filter_peaks(atac,min_n_cells=3)

    # %%
    si.pp.pca(atac, n_components=50)

    # %%
    si.pp.select_pcs(atac,n_pcs=40)

    # %%
    si.pp.select_pcs_features(atac)

    # %%
    si.pp.filter_genes(rna, min_n_cells=3)

    # %%
    rna.X = rna.X.astype(float)
    si.pp.normalize(rna,method='lib_size')

    # %%
    si.pp.log_transform(rna)

    # %%
    si.pp.select_variable_genes(rna, n_top_genes=2000)

    # %%
    atac.var["chr"] = list(atac.var["chrom"])
    atac.var["start"] = list(atac.var["chromStart"])
    atac.var["end"] = list(atac.var["chromEnd"])

    # %%
    if(os.path.exists(data_dir+"/simba_gs.h5ad")):
        G_atac = sc.read_h5ad(data_dir+"/simba_gs.h5ad")
    else:
        G_atac = si.tl.gene_scores(atac,genome='mm10',use_gene_weigt=True, use_top_pcs=True)
        # G_atac.var_names = [i.upper() for i in list(G_atac.var_names)]
        G_atac.write(data_dir+"/simba_gs.h5ad")

    # %%
    si.pp.filter_genes(G_atac,min_n_cells=3)
    # print(G_atac.var_names[5000:5200])
    # print(rna.var_names)
    
    si.pp.cal_qc_rna(G_atac)
    si.pp.normalize(G_atac,method='lib_size')
    si.pp.log_transform(G_atac)

    # %%
    G_atac.obs.index = G_atac.obs.index + '_atac'
    atac.obs.index = atac.obs.index + '_atac'
    rna.obs.index = rna.obs.index + '_rna'

    # %%
    adata_CrnaCatac = si.tl.infer_edges(rna, G_atac, n_components=15, k=15)
    adata_CrnaCatac

    # %%
    si.tl.trim_edges(adata_CrnaCatac, cutoff=0.6)

    # %%
    # modify parameters

    # %%
    si.tl.gen_graph(list_CP=[atac],
                    list_CG=[rna],
                    list_CC=[adata_CrnaCatac],
                    copy=False,
                    use_highly_variable=True,
                    use_top_pcs=True,
                    dirname='graph0')

    # %%
    si.tl.pbg_train(auto_wd=False, save_wd=True,)

    # %%
    dict_adata = si.read_embedding()

    # %%
    adata_C = dict_adata['C']  # embeddings for ATACseq cells
    adata_C2 = dict_adata['C2']  # embeddings for RNAseq cells
    adata_G = dict_adata['G']  # embeddings for genes
    adata_P = dict_adata['P']  # embeddings for peaks

    # %%
    atac.obsm["embed"] = adata_C[atac.obs_names].X
    rna.obsm["embed"] = adata_C2[rna.obs_names].X
    atac.obs["omic"] = "atac"
    rna.obs["omic"] = "rna"
    adata_all = sc.concat([rna, atac])

    # %%
    sc.pp.neighbors(adata_all, use_rep="embed")
    sc.tl.umap(adata_all,)

    # %%
    import pandas as pd
    import numpy as np
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

    # %%
    sc.tl.leiden(adata_all, resolution=res)

    # %%
    # sc.pl.umap(adata_all, color=["omic","leiden"])

    # %%
    # t_rna, t_atac  = adata_all[adata_all.obs["omic"]=="rna",], adata_all[adata_all.obs["omic"]=="atac",]
    # sc.pl.spatial(t_rna, color="leiden",spot_size=1)
    # sc.pl.spatial(t_atac, color="leiden",spot_size=1)

    cluster = adata_all.obs['leiden']
    cluster.to_csv(save_dir+"/SIMBA_cluster.csv")

    umap = pd.DataFrame(adata_all.obsm["X_umap"])
    umap.to_csv(save_dir+"/SIMBA_umap.csv")

    embed = pd.DataFrame(adata_all.obsm["embed"])
    embed.to_csv(save_dir+"/SIMBA_embed.csv")

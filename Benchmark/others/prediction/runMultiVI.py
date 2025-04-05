print("=============Running MultiVI=================")
import numpy as np
import scanpy as sc
import scvi
import seaborn as sns
import torch
import anndata as ad


import sys
data_dir = str(sys.argv[1])
save_dir = str(sys.argv[2])
fraction = 0.1 * int(sys.argv[3])
print(f"paired fraction: {fraction}")

rna = ad.read_h5ad(data_dir+"/rna-pp.h5ad")
atac = ad.read_h5ad(data_dir+"/atac-pp.h5ad")

rna = rna[:,rna.var["highly_variable"]]
atac = atac[:,atac.var["highly_variable"]]


rna.var['modality'] =  "Gene Expression"
atac.var['modality'] =  "Peaks"

paired_idx = np.random.choice(rna.shape[0], size=int(rna.shape[0]*fraction),replace=False)
paired_bcd = rna.obs_names[paired_idx]
adata_paired = sc.concat([rna[paired_bcd], atac[paired_bcd]], axis=1)

adata_rna = rna[~rna.obs_names.isin(paired_bcd)]
adata_atac = atac[~atac.obs_names.isin(paired_bcd)]

adata_mvi = scvi.data.organize_multiome_anndatas(adata_paired, adata_rna, adata_atac)

adata_mvi = adata_mvi[:, adata_mvi.var["modality"].argsort()].copy()
# adata_mvi.var

# sc.pp.filter_genes(adata_mvi, min_cells=int(adata_mvi.shape[0] * 0.01))

scvi.model.MULTIVI.setup_anndata(adata_mvi, batch_key="modality")
model = scvi.model.MULTIVI(
    adata_mvi,
    n_genes=(adata_mvi.var["modality"] == "Gene Expression").sum(),
    n_regions=(adata_mvi.var["modality"] == "Peaks").sum(),
)
model.view_anndata_setup()

model.train()

# MULTIVI_LATENT_KEY = "X_multivi"

# adata_mvi.obsm[MULTIVI_LATENT_KEY] = model.get_latent_representation()
# sc.pp.neighbors(adata_mvi, use_rep=MULTIVI_LATENT_KEY)
# sc.tl.umap(adata_mvi, min_dist=0.2)
# sc.pl.umap(adata_mvi, color="modality")

imputed_expression = model.get_normalized_expression()

imputed_obs = adata_atac.obs_names.map(lambda x: x+"_accessibility")
paired_obs = []
for i in list(imputed_expression.index):
    if(i.split("_")[-1]=="paired"):
        paired_obs.append(i)
if(len(imputed_obs) > 0.48 * imputed_expression.shape[0]):
    paired_rna = ad.AnnData(imputed_expression[imputed_expression.index.isin(paired_obs)])
    paired_rna.obs_names = paired_rna.obs_names.map(lambda x: x.split("_")[0])
else:
    imputed_rna = ad.AnnData(imputed_expression[imputed_expression.index.isin(imputed_obs)])
    paired_rna = ad.AnnData(imputed_expression[imputed_expression.index.isin(paired_obs)])
    imputed_rna.obs_names = imputed_rna.obs_names.map(lambda x: x.split("_")[0])
    paired_rna.obs_names = paired_rna.obs_names.map(lambda x: x.split("_")[0])
    imputed_rna.write(save_dir+"/MultiVI_imputed_rna.h5ad")
paired_rna.write(save_dir+"/MultiVI_paired_rna.h5ad")


imputed_accessibility = model.get_accessibility_estimates()

imputed_obs = adata_atac.obs_names.map(lambda x: x+"_expression")
if(len(imputed_obs)> 0.98 * imputed_accessibility.shape[0]):
    paired_atac = ad.AnnData(imputed_accessibility[imputed_accessibility.index.isin(paired_obs)])
    paired_atac.obs_names = paired_atac.obs_names.map(lambda x: x.split("_")[0])
else:
    imputed_atac = ad.AnnData(imputed_accessibility[imputed_accessibility.index.isin(imputed_obs)])
    paired_atac = ad.AnnData(imputed_accessibility[imputed_accessibility.index.isin(paired_obs)])
    imputed_atac.obs_names = imputed_atac.obs_names.map(lambda x: x.split("_")[0])
    paired_atac.obs_names = paired_atac.obs_names.map(lambda x: x.split("_")[0])
    imputed_atac.write(save_dir+"/MultiVI_imputed_atac.h5ad")
paired_atac.write(save_dir+"/MultiVI_paired_atac.h5ad")

print(f"paired rna shape: {paired_rna.shape[0], paired_rna.shape[1]}")
print(f"paired atac shape: {paired_atac.shape[0], paired_atac.shape[1]}")
print(f"unpaired rna shape: {imputed_rna.shape[0], imputed_rna.shape[1]}")
print(f"unpaired atac shape: {imputed_atac.shape[0], imputed_atac.shape[1]}")





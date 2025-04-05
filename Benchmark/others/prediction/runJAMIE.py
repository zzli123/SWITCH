# %%
print("=============Running JAMIE=================")

import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append("/mnt/datadisk/lizhongzhan/SpaMultiOmics/main/")
import scanpy as sc
from jamie import JAMIE
from jamie.evaluation import *
from jamie.utilities import *
import numpy as np
from sklearn import preprocessing

import sys
data_dir = str(sys.argv[1])
save_dir = str(sys.argv[2])
fraction = 0.1 * int(sys.argv[3])
print(f"paired fraction: {fraction}")


np.random.seed(42)
model_folder = '/mnt/datadisk/lizhongzhan/SpaMultiOmics/SCRIPT/Figure2/JAMIE/output/saved_models/'
image_folder = '/mnt/datadisk/lizhongzhan/SpaMultiOmics/SCRIPT/Figure2/JAMIE/output/output_figures/'
output_folder = '/mnt/datadisk/lizhongzhan/SpaMultiOmics/SCRIPT/Figure2/JAMIE/output/output_data/'


rna = sc.read_h5ad(data_dir+"/rna-pp.h5ad")
atac = sc.read_h5ad(data_dir+"/atac-pp.h5ad")
rna = rna[:,rna.var["highly_variable"]]
atac = atac[:,atac.var["highly_variable"]]

raw_data1 = rna.X
raw_data2 = atac.X.toarray()

data1 = preprocessing.scale(raw_data1, axis=0)
data2 = preprocessing.scale(raw_data2, axis=0)
data1[np.isnan(data1)] = 0  
data2[np.isnan(data2)] = 0
dataset = [data1, data2]

features = [None, None]
for i in range(len(features)):
    if features[i] is None:
        features[i] = np.array([f'Feature {i}' for i in range(dataset[i].shape[1])])
        
train_size = int(.8 * len(data1))
train_idx = np.random.choice(range(len(data1)), train_size, replace=False)
test_idx = np.array(list(set(range(len(data1))) - set(train_idx)))


random_idx = np.random.choice(range(len(dataset[0])), int(fraction * len(dataset[0])), replace=False)
priors = np.zeros(len(dataset[0]))
priors[random_idx] = 1
priors = np.diag(priors)

reduced_dim = 32
kwargs = {
    'output_dim': reduced_dim,
    'epoch_DNN': 10000,
    'min_epochs': 2500,
    'log_DNN': 500,
    'use_early_stop': True,
    'batch_size': 512,
    'pca_dim': 2*[512],
    'dist_method': 'euclidean',
    'loss_weights': [1,1,1,1],
}
kwargs_imp = {k: kwargs[k] for k in kwargs if k != 'dropout'}

size_str, hash_str = hash_kwargs(kwargs, "mouse_embryo", dataset)
prefix = model_folder + 'jm_50---'
model_str = prefix + hash_str + '.h5'
match_str = prefix + size_str + '.npy'

# Instantiate
mr = None
jm = JAMIE(**kwargs, match_result=mr, debug=True)


jm_data = jm.fit_transform(dataset=dataset, P=priors)
# jm.save_model(model_str)

jm_imputed = [jm.modal_predict(dataset[i], i) for i in range(1, -1, -1)]


import anndata as ad
pred_rna = ad.AnnData(jm_imputed[0])
pred_atac = ad.AnnData(jm_imputed[1])
pred_rna.obs_names = atac.obs_names
pred_rna.var_names = rna.var_names
pred_atac.obs_names = rna.obs_names
pred_atac.var_names = atac.var_names

paired_bcd =  atac.obs_names[random_idx]
if(len(paired_bcd)> 0.98 * pred_rna.shape[0]):
    paired_rna = pred_rna.copy()
    paired_atac = pred_atac.copy()
else:
    imputed_rna = pred_rna[~pred_rna.obs_names.isin(paired_bcd)].copy()
    paired_rna = pred_rna[pred_rna.obs_names.isin(paired_bcd)]
    imputed_atac = pred_atac[~pred_atac.obs_names.isin(paired_bcd)]
    paired_atac = pred_atac[pred_atac.obs_names.isin(paired_bcd)]
    imputed_rna.write(save_dir+"/JAMIE_imputed_rna.h5ad")
    imputed_atac.write(save_dir+"/JAMIE_imputed_atac.h5ad")

print(f"paired rna shape: {paired_rna.shape[0], paired_rna.shape[1]}")
print(f"paired atac shape: {paired_atac.shape[0], paired_atac.shape[1]}")
print(f"unpaired rna shape: {imputed_rna.shape[0], imputed_rna.shape[1]}")
print(f"unpaired atac shape: {imputed_atac.shape[0], imputed_atac.shape[1]}")

paired_rna.write(save_dir+"JAMIE_paired_rna.h5ad")
paired_atac.write(save_dir+"JAMIE_paired_atac.h5ad")






print("=============Running scButterfly=================")
from scButterfly.butterfly import Butterfly
import scanpy as sc
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

rna.obs_names = rna.obs_names.map(lambda x: x+"_rna")
atac.obs_names = atac.obs_names.map(lambda x: x+"_atac")

RNA_data = rna.copy()
ATAC_data = atac.copy()

import numpy as np
paired_idx = np.random.choice(rna.shape[0], size=int(rna.shape[0]*fraction),replace=False)
all_idx = list(range(rna.shape[0]))

butterfly = Butterfly()

butterfly.load_data(RNA_data, ATAC_data, paired_idx, all_idx)

butterfly.data_preprocessing(binary_data=False, filter_features=False, fpeaks=None)

chrom_list = []
last_one = ''
for i in range(len(butterfly.ATAC_data_p.var.chrom)):
    temp = butterfly.ATAC_data_p.var.chrom[i]
    if temp[0 : 3] == 'chr':
        if not temp == last_one:
            chrom_list.append(1)
            last_one = temp
        else:
            chrom_list[-1] += 1
    else:
        chrom_list[-1] += 1

# print(chrom_list, end="")

butterfly.augmentation(aug_type=None)

butterfly.construct_model(chrom_list=chrom_list)

butterfly.train_model(batch_size=128)

A2R_predict, R2A_predict = butterfly.test_model(batch_size=10000)

imputed_idx = []
for i in range(rna.shape[0]):
    if(i in paired_idx):
        pass
    else:
        imputed_idx.append(i)

A2R_predict.var_names = rna.var_names.copy()
R2A_predict.var_names = atac.var_names.copy()

paired_atac = R2A_predict[paired_idx,:]
paired_rna = A2R_predict[paired_idx,:]
imputed_atac = R2A_predict[imputed_idx,:]
imputed_rna = A2R_predict[imputed_idx,:]

paired_rna.write(save_dir+"/scButterfly_paired_rna.h5ad")
paired_atac.write(save_dir+"/scButterfly_paired_atac.h5ad")
imputed_rna.write(save_dir+"/scButterfly_imputed_rna.h5ad")
imputed_atac.write(save_dir+"/scButterfly_imputed_atac.h5ad")

print(f"paired rna shape: {paired_rna.shape[0], paired_rna.shape[1]}")
print(f"paired atac shape: {paired_atac.shape[0], paired_atac.shape[1]}")
print(f"unpaired rna shape: {imputed_rna.shape[0], imputed_rna.shape[1]}")
print(f"unpaired atac shape: {imputed_atac.shape[0], imputed_atac.shape[1]}")






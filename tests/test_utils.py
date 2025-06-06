import sys
import numpy as np
sys.path.append("/mnt/datadisk/lizhongzhan/SpaMultiOmics/main/")
from switch import integration_score

def test_integration_score(rna_pp, atac_pp):
    rna_pp.obsm["temp"] =  np.random.random((rna_pp.shape[0], 10))
    atac_pp.obsm["temp"] =  np.random.random((atac_pp.shape[0], 10))
    score = integration_score(adatas=[rna_pp, atac_pp], distance=2, use_rep="temp")
    assert score > 0
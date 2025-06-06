import sys
sys.path.append("/mnt/datadisk/lizhongzhan/SpaMultiOmics/main/")
from switch import SWITCH, preprocess
from itertools import chain
import pytest

@pytest.mark.parametrize("rna_prob", ["NB"])
@pytest.mark.parametrize("atac_prob", ["NB", "POIS"])
def test_SWITCH(
    rna_pp, atac_pp, rna_prob, atac_prob,
):  
    print(rna_pp)
    guidance = preprocess.rna_anchored_guidance_graph(rna_pp, atac_pp, promoter_len=2)
    preprocess.Cal_Spatial_Net(rna_pp, rad_cutoff=1, model="Radius")
    preprocess.Cal_Spatial_Net(atac_pp, rad_cutoff=1, model="Radius")
    preprocess.config_data(
        rna_pp,
        rna_prob,
        use_highly_variable=True,
        use_layer="counts"
    )
    preprocess.config_data(
        atac_pp,
        atac_prob,
        use_highly_variable=True,
    )
    vertices = sorted(guidance.nodes)

    guidance_hvf = guidance.subgraph(chain(
        rna_pp.var.query("highly_variable").index,
        atac_pp.var.query("highly_variable").index
    )).copy()
    model = SWITCH(
        {"rna": rna_pp, "atac": atac_pp}, vertices, latent_dim=2, seed=0
    )
    model.compile()
    model.pretrain({"rna": rna_pp, "atac": atac_pp}, guidance, max_epochs=1, dsc_k=2)
    model.train({"rna": rna_pp, "atac": atac_pp}, guidance, max_epochs=1, dsc_k=2)

    model.pretrain({"rna": rna_pp, "atac": atac_pp}, guidance, max_epochs=1, mini_batch=True, iteration=2, dsc_k=2)
    model.train({"rna": rna_pp, "atac": atac_pp}, guidance, max_epochs=1, mini_batch=True, iteration=2, dsc_k=2)

    rna_pp.obsm["SWITCH"] = model.encode_data("rna", rna_pp)
    atac_pp.obsm["SWITCH"] = model.encode_data("atac", atac_pp)
    graph_embedding1 = model.encode_graph(guidance)

    imputed_rna =  model.impute_data(source_key="atac",target_key="rna", 
                                     source_adata=atac_pp, target_adata=rna_pp,
                                     graph=guidance_hvf)
    imputed_atac =  model.impute_data(source_key="rna",target_key="atac", 
                                     source_adata=rna_pp, target_adata=atac_pp,
                                     graph=guidance_hvf)
    denosied_rna =  model.impute_data(source_key="rna",target_key="rna",
                                      source_adata=rna_pp, target_adata=rna_pp,
                                      graph=guidance_hvf)
    denoised_atac =  model.impute_data(source_key="atac",target_key="atac", 
                                       source_adata=atac_pp, target_adata=atac_pp,
                                       graph=guidance_hvf)

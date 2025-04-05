# %%
library("Seurat", help, pos = 2, lib.loc = NULL)

# %%
args <- commandArgs()
data_dir = args[6]
save_dir = args[7]
n_clus = args[8]
peak_data_name = args[9]

# %%
library(zellkonverter)
if(!file.exists(paste0(data_dir,"/GeneScore.rds"))){
    genescore = readH5AD(paste0(data_dir,"/simba_gs.h5ad"))
    genescore = CreateSeuratObject(counts = genescore@assays@data$X)
    saveRDS(genescore, file = paste0(data_dir,"GeneScore.rds"))
}
if(!file.exists(paste0(data_dir,"/rna.rds"))){
    rna <- readH5AD(paste0(data_dir,"/rna-pp.h5ad"))
    rna <- CreateSeuratObject(counts = rna@assays@data$X)
    saveRDS(rna, file = paste0(data_dir,"rna.rds"))
}

rna <- readRDS(paste0(data_dir,"/rna.rds"))
genescore <- readRDS(paste0(data_dir,"/GeneScore.rds"))

rna$omic <- "rna"
genescore$omic <- "atac"

rna <- NormalizeData(rna, verbose=FALSE)
rna <- FindVariableFeatures(rna, verbose=FALSE, nfeatures=3000)
rna <- ScaleData(rna, verbose=FALSE)

genescore <- NormalizeData(genescore)
genescore <- FindVariableFeatures(genescore, verbose=FALSE, nfeatures=3000)
genescore <- ScaleData(genescore)

Int.anchors <- FindIntegrationAnchors(object.list = list(genescore, rna))
combined <- IntegrateData(anchorset = Int.anchors)
DefaultAssay(combined) <- "integrated"
combined <- ScaleData(combined, verbose = FALSE)
combined <- RunPCA(combined, verbose = FALSE)
# combined <- RunUMAP(combined, reduction = "pca", dims = 1:50)
# combined <- FindNeighbors(combined, reduction = "pca", dims = 1:50)

# combined <- FindClusters(combined)
# print(length(unique((combined$seurat_clusters))))

# %%
embed = combined@reductions$pca@cell.embeddings
# umap <- combined@reductions$umap@cell.embeddings
# cluster <- combined$seurat_clusters

# %%
write.csv(file=paste0(save_dir,"/Seurat_embed.csv"), embed)
# write.csv(file="rep1/Seurat_umap.csv", umap)
# write.csv(file="rep1/Seurat_cluster.csv", cluster)


